"""Read-only originals and aggregate-only duplicate audit for ratio-v1."""
from __future__ import annotations
from collections import Counter, OrderedDict
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from .data import read_rgb, write_json


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


class Sources:
    def __init__(self, manifest, cache_size=24):
        self.manifest = Path(manifest)
        self.records = json.loads(self.manifest.read_text())['records']
        self.cache = OrderedDict()
        self.cache_size = cache_size

    def read(self,index):
        index = int(index)
        if index not in self.cache:
            try:
                pixels = read_rgb(Path(self.records[index]['source_path']))
            except Exception as error:
                # Do not disclose private paths in process logs.
                raise RuntimeError(f'Source ordinal {index}: {type(error).__name__}') from None
            self.cache[index] = pixels
        self.cache.move_to_end(index)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return self.cache[index]


def phash(image):
    gray = cv2.cvtColor(cv2.resize(image,(32,32),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY).astype(np.float32)
    d = cv2.dct(gray)[:8,:8].reshape(-1)[1:]
    return np.packbits(d > np.median(d))


def audit(sources, output):
    """Strong verified perceptual duplicates join exact/224-equal clusters in RAM.

    No identity, grouping key, per-source hash, or original path is persisted.
    Historical dev is retained. Linked training sources are conservatively quarantined.
    """
    n = len(sources.records)
    parent = list(range(n))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def join(i,j):
        parent[root(j)] = root(i)
    exact, legacy = {}, {}
    hashes, thumbnails, shapes, ratios = [], [], [], []
    counts = Counter()
    for i,record in enumerate(sources.records):
        if sha256(record['source_path']) != record['source_sha256']:
            raise ValueError(f'Source ordinal {i} changed since manifest')
        pixels = sources.read(i)
        height,width = pixels.shape[:2]
        if [width,height] != record['original_size']:
            raise ValueError(f'EXIF size mismatch at source ordinal {i}')
        shapes.append((width,height)); ratios.append(width/height)
        digest = hashlib.sha256(np.array(pixels.shape,dtype=np.int64).tobytes()+pixels.tobytes()).digest()
        reduced = cv2.resize(pixels,(224,224),interpolation=cv2.INTER_AREA)
        digest224 = hashlib.sha256(reduced.tobytes()).digest()
        for table,key,label in ((exact,digest,'exact_original_copies'),(legacy,digest224,'equal_224_copies')):
            if key in table:
                join(i,table[key]); counts[label] += 1
            else:
                table[key] = i
        hashes.append(phash(pixels))
        gray = cv2.cvtColor(cv2.resize(pixels,(64,64),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        thumbnails.append(gray)
        if (i+1)%500 == 0:
            print(json.dumps({'phase':'audit','sources_read':i+1,'total':n}),flush=True)
    hashes = np.stack(hashes)
    bitcounts = np.array([int(i).bit_count() for i in range(256)],np.uint8)
    near_pairs = cross_suspects = 0
    suspect_pairs, review_pairs = [], []
    for i in range(n):
        distance = bitcounts[np.bitwise_xor(hashes[i],hashes[i+1:])].sum(axis=1)
        for j in np.flatnonzero(distance <= 4)+i+1:
            if root(i) == root(int(j)):
                continue
            a,b = thumbnails[i], thumbnails[j]
            corr = float(np.corrcoef(a.reshape(-1),b.reshape(-1))[0,1])
            mae = float(np.abs(a-b).mean())
            if abs(np.log(ratios[i]/ratios[j])) < .01 and corr >= .995 and mae <= .02:
                join(i,int(j)); near_pairs += 1
                if sources.records[i]['split'] != sources.records[j]['split']:
                    review_pairs.append((i,int(j),'verified near duplicate'))
            elif corr >= .98:
                suspect_pairs.append((i,int(j)))
                if sources.records[i]['split'] != sources.records[j]['split']:
                    cross_suspects += 1
                    review_pairs.append((i,int(j),'similarity candidate only'))
    cluster_splits = {}
    for i,row in enumerate(sources.records):
        cluster_splits.setdefault(root(i),set()).add(row['split'])
    crossings = sum(len(v)>1 for v in cluster_splits.values())
    strict_dev_roots = {root(i) for i,row in enumerate(sources.records) if row['split']=='val'}
    strict_excluded = {i for i,row in enumerate(sources.records) if row['split']=='train' and root(i) in strict_dev_roots}
    for i,j in suspect_pairs:
        join(i,j)
    dev_roots = {root(i) for i,row in enumerate(sources.records) if row['split']=='val'}
    quarantine = [i for i,row in enumerate(sources.records) if row['split']=='train' and root(i) in dev_roots]
    train = [i for i,row in enumerate(sources.records) if row['split']=='train' and i not in quarantine]
    dev = [i for i,row in enumerate(sources.records) if row['split']=='val']
    result = {'sources':n,'train':len(train),'dev':len(dev),**counts,'verified_near_duplicate_pairs':near_pairs,
              'cross_split_duplicate_clusters':crossings,'cross_split_near_suspects':cross_suspects,
              'historical_train':sum(row['split']=='train' for row in sources.records),'quarantined_train_sources':len(quarantine),
              'strict_duplicate_train_excluded':len(strict_excluded),
              'additional_candidate_train_excluded':len(set(quarantine)-strict_excluded),
              'split_version':'ratio-v1.1-dev478-quarantine',
              'decision_provenance':'main_task_authorized_2026-09-21; retain historical dev; quarantine linked train; no patient identity inference',
              'candidate_rule':'pHash_Hamming<=4 and gray64_corr>=.98; connected closure conservatively quarantined',
              'cross_split_links_remaining':0, 'source_exposures_per_group':80000,
              'mean_source_exposures_per_group':80000/len(train),
              'patient_mapping_available':False,'scope':'exposed_source_image_development_not_independent',
              'manifest_sha256':sha256(sources.manifest),'ratio_target':'EXIF_full_canvas_W/H_including_original_black_borders',
              'mean_train_log_ratio':float(np.mean(np.log(np.array(ratios)[train]))),
              'ratio_min':min(ratios),'ratio_max':max(ratios),'square_canvases':sum(w==h for w,h in shapes),
              'backgrounds':'independent_procedural_no_patient_source', 'originals_verified_unchanged':True,
              'near_duplicate_rule':'pHash_Hamming<=4; aspect_logdiff<.01; gray64_corr>=.995; gray64_MAE<=.02',
              'unknown_patient_overlap_not_excluded':True}
    write_json(Path(output)/'data-audit.json',result)
    write_json(Path(output)/'split.local.json',{'train':train,'dev':dev,'quarantine':quarantine})
    if review_pairs:
        tiles=[]
        for pair_number,(i,j,reason) in enumerate(review_pairs):
            tile=np.concatenate([cv2.resize((thumbnails[k]*255).astype(np.uint8),(192,192)) for k in (i,j)],axis=1)
            tile=cv2.cvtColor(tile,cv2.COLOR_GRAY2BGR)
            cv2.putText(tile,f'{pair_number+1}: {reason}',(4,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1)
            tiles.append(tile)
        for start in range(0,len(tiles),8):
            cv2.imwrite(str(Path(output)/f'duplicate-review-{start//8:02d}.local.jpg'),np.concatenate(tiles[start:start+8],axis=0))
    return result
