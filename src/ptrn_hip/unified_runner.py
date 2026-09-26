"""Fixed-budget local unified PTRN experiment, with sealed scene evaluation.

Images are rebuilt from immutable schedules. Originals are read-only; all caches,
checkpoints, source snapshots and per-view records stay under the ignored run root.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from .data import write_json
from .ratio_data import Sources, sha256
from .ratio_synthesis import letterbox, project

SEED = 20260922
GROUPS = {'S224': (224, 'head', 5000, 64),
          'F224': (224, 'head', 1250, 64),
          'J224': (224, 'last_block', 1250, 16),
          'J384': (384, 'last_block', 1250, 4)}
RECIPE = {'seed': SEED, 'effective_batch': 64, 'head_lr': 1e-4,
          'backbone_lr': 1e-5, 'clip_norm': 1., 'evaluation_every': 250,
          'updates': {g: v[2] for g, v in GROUPS.items()}, 'total_exposures': 560000,
          'source_counts': {'train': 4114, 'dev': 478, 'quarantine': 28},
          'train_views': 80000, 'dev_views': 5258, 'test_views': 4780,
          'photo_geometry': [0.45, 0.35, 0.15, 0.05],
          'photo_occupancy_bins': [[0.30, 0.50], [0.50, 0.75], [0.75, 0.90]],
          'photo_occupancy_probabilities': [0.20, 0.60, 0.20],
          'loss': 'SmoothL1(q_error/(.01*input_diagonal))+.25*SmoothL1(log_ratio_error/.05)',
          'ratio': 'log(edge_ratio(current_predicted_photo_quad)).detach()+raw_delta',
          'BN': 'frozen', 'head_precision': 'float32',
          'joint_precision': 'bfloat16_autocast_float32_loss',
          'view_storage': 'fixed_schedule_and_seed_reconstruction; no_full_photo_library'}


def stamp(path, **record):
    write_json(Path(path), {'time_unix': time.time(), 'pid': os.getpid(), **record})


def disk_check(path, minimum=.75):
    free = shutil.disk_usage(path).free / 2**30
    if free < minimum:
        raise RuntimeError(f'Insufficient free disk ({free:.2f} GiB; reserve {minimum:.2f} GiB)')
    return free


def source_inventory(root):
    root = Path(root)
    paths = sorted([*root.glob('src/**/*.py'), *root.glob('tests/**/*.py'), root/'pyproject.toml'])
    files = {str(p.relative_to(root)): sha256(p) for p in paths if p.is_file()}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {'files': files, 'tree_sha256': digest,
            'git_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()}


def freeze_source(run):
    root = Path.cwd()
    inventory = source_inventory(root)
    record = Path(run)/'source-inventory.json'
    if record.exists():
        if json.loads(record.read_text()) != inventory:
            raise RuntimeError('Training source changed after freeze; create a new run or explicitly repair pre-training evidence')
    else:
        output = Path(run)/'source-snapshot'
        for relative in inventory['files']:
            target = output/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root/relative, target)
        write_json(record, inventory)
    return inventory


def assert_source(run):
    expected = json.loads((Path(run)/'source-inventory.json').read_text())
    if source_inventory(Path.cwd()) != expected:
        raise RuntimeError('Refusing to use source drift during the frozen experiment')
    return expected


def read_schedule(run, split, allow_test=False):
    if split == 'test' and not allow_test:
        raise RuntimeError('Sealed test must not be opened for development')
    directory = Path(run)/('sealed' if split == 'test' else 'schedules')
    return json.loads((directory/f'{split}.json').read_text())


def prepare(args):
    from .unified_data import build_schedule, audit_schedule
    args.run.mkdir(parents=True, exist_ok=True)
    disk_check(args.run, 4.)
    prior = json.loads((args.prior_run/'data-audit.json').read_text())
    if sha256(args.manifest) != prior['manifest_sha256']:
        raise RuntimeError('Manifest differs from the audited historical source partition')
    splits = json.loads((args.prior_run/'split.local.json').read_text())
    if {k: len(v) for k, v in splits.items()} != RECIPE['source_counts']:
        raise RuntimeError('Frozen source counts changed')
    if any(set(splits[a]) & set(splits[b]) for a, b in [('train','dev'),('train','quarantine'),('dev','quarantine')]):
        raise RuntimeError('Source partitions overlap')
    sources = Sources(args.manifest, cache_size=1)
    if len(sources.records) != 4620:
        raise RuntimeError('Original source count changed')
    integrity = args.run/'original-integrity-before.json'
    if not integrity.exists():
        for ordinal, record in enumerate(sources.records):
            try:
                valid = sha256(record['source_path']) == record['source_sha256']
                pixels = sources.read(ordinal)
            except Exception as error:
                raise RuntimeError(f'Source ordinal {ordinal}: {type(error).__name__}') from None
            if not valid or [pixels.shape[1],pixels.shape[0]] != record['original_size']:
                raise RuntimeError(f'Source ordinal {ordinal}: integrity or dimension mismatch')
            if (ordinal+1) % 500 == 0:
                stamp(args.run/'status.json', phase='source_audit', completed=ordinal+1, total=4620)
                print(json.dumps({'phase':'source_audit','completed':ordinal+1,'total':4620}), flush=True)
        write_json(integrity, {'originals_checked':4620, 'all_hashes_and_EXIF_sizes_match':True,
                              'manifest_sha256':sha256(args.manifest)})
    write_json(args.run/'split.local.json', splits)
    audits = {}
    for split, key in [('train','train'),('dev','dev'),('test','dev')]:
        rows = build_schedule(splits[key], split=split, seed=SEED)
        directory = args.run/('sealed' if split == 'test' else 'schedules')
        target = directory/f'{split}.json'
        if target.exists() and json.loads(target.read_text()) != rows:
            raise RuntimeError(f'{split} schedule changed')
        write_json(target, rows)
        audits[split] = audit_schedule(rows)
        expected = RECIPE[{'train':'train_views','dev':'dev_views','test':'test_views'}[split]]
        if len(rows) != expected:
            raise RuntimeError(f'{split} schedule length differs from approved plan')
    write_json(args.run/'recipe.json', RECIPE)
    write_json(args.run/'data-audit.json', {'source_audit':prior, 'schedules':audits,
               'test_scope':'same_exposed_sources_new_sealed_scenes_not_patient_independent',
               'sealed_test_rendered':False, 'actual_training_views_audited':False})
    stamp(args.run/'status.json', phase='prepared', source_counts=RECIPE['source_counts'],
          free_disk_gib=disk_check(args.run), training_started=False)
    print(json.dumps({'phase':'prepared','source_counts':RECIPE['source_counts'],
                      'views':{'train':80000,'dev':5258,'test_sealed':4780}}), flush=True)


def sample_record(sources, record, size):
    from .unified_synthesis import sample
    row = dict(record)
    index = int(row['source_index'])
    source = sources.read(index)
    view = sample(source, index, int(row['view_index']), kind=row['kind'],
                  mode=row.get('mode','front'), occupancy_bin=row.get('occupancy_bin',1),
                  level=row.get('level','clear'), pure_roll=row.get('pure_roll',False),
                  domain=row.get('domain','train'), seed=SEED)
    image = view.clean if row.get('condition') in ('clean','clear') else view.noisy
    pixels, lb = letterbox(image, size)
    q = project(view.quad, lb).astype(np.float32)
    info = {**row, **view.info, 'source_index':index,
            'photo_size':[image.shape[1], image.shape[0]],
            'source_size':view.info['source_size'],
            'source_to_photo':np.asarray(view.info['source_to_photo']).tolist(),
            'quad_photo':np.asarray(view.quad).tolist(),
            'quad_input':q.tolist(), 'log_ratio':float(view.log_ratio),
            'inverse_letterbox':np.linalg.inv(lb).tolist()}
    return pixels, q, np.float32(view.log_ratio), np.linalg.inv(lb).astype(np.float32), info, image


class ViewDataset:
    def __init__(self, manifest, schedule, size, positions=None, legacy=False):
        self.manifest, self.schedule, self.size = str(manifest), schedule, size
        self.positions = list(range(len(schedule))) if positions is None else list(positions)
        self.legacy = legacy
        self.sources = None

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, position):
        index = self.positions[position]
        if self.sources is None:
            self.sources = Sources(self.manifest, cache_size=2)
        pixels,q,ratio,inverse,info,image = sample_record(self.sources,self.schedule[index],self.size)
        old = cv2.resize(image,(224,224),interpolation=cv2.INTER_AREA) if self.legacy else np.zeros((1,1,3),np.uint8)
        return index,pixels,q,ratio,inverse,json.dumps(info),old


def worker_init(_):
    cv2.setNumThreads(1)


def loader(dataset, batch, workers):
    from .model import torch
    return torch.utils.data.DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=workers,
        worker_init_fn=worker_init,pin_memory=True,
        **({'multiprocessing_context':'spawn','persistent_workers':True,'prefetch_factor':2} if workers else {}))


def configure():
    from .model import configure as setup, torch, keras
    cv2.setNumThreads(1)
    environment = setup(False)
    keras.utils.set_random_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError('Approved local CUDA device is unavailable')
    return environment


def preview(args):
    schedule = read_schedule(args.run,'train')
    sources = Sources(args.manifest,cache_size=1)
    chosen = []
    # Include extreme original aspect ratios and every geometry/occupancy pairing.
    splits = json.loads((args.run/'split.local.json').read_text())
    train = splits['train']
    ratios = {i:sources.records[i]['original_size'][0]/sources.records[i]['original_size'][1] for i in train}
    extremes = [min(train,key=ratios.get),max(train,key=ratios.get)]
    seen = set()
    for ordinal,row in enumerate(schedule):
        key = (row['kind'],row.get('mode'),row.get('occupancy_bin'),bool(row.get('pure_roll')))
        if key not in seen:
            chosen.append(ordinal);seen.add(key)
    for source_index in extremes:
        chosen.extend(i for i,r in enumerate(schedule) if r['source_index']==source_index and r['kind']=='photo')
    chosen = list(dict.fromkeys(chosen))
    out = args.run/'preview';out.mkdir(parents=True,exist_ok=True)
    tiles,records = [],[]
    for number,index in enumerate(chosen):
        _,_,_,_,info,image = sample_record(sources,schedule[index],224)
        overlay = image.copy()
        q = np.rint(info['quad_photo']).astype(np.int32)
        cv2.polylines(overlay,[q],True,(0,255,0),2)
        thumb = letterbox(overlay,320)[0]
        label=f"{number+1:02d} {info['kind']} {info.get('mode')} a={info.get('occupancy',1):.2f}"
        cv2.putText(thumb,label,(4,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,180,0),1)
        tiles.append(thumb)
        records.append({'ordinal':number+1,'kind':info['kind'],'mode':info.get('mode'),
                        'occupancy':info.get('occupancy'),'tilt':info.get('tilt'),'roll':info.get('roll')})
    for start in range(0,len(tiles),6):
        group=tiles[start:start+6]
        group += [np.zeros_like(tiles[0])]*(6-len(group))
        sheet=np.concatenate([np.concatenate(group[:3],axis=1),np.concatenate(group[3:],axis=1)],axis=0)
        cv2.imwrite(str(out/f'sheet-{start//6:02d}.local.jpg'),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR))
    write_json(out/'summary.json',{'count':len(records),'records':records,'scope':'authorized_real_source_synthetic_preview_not_real_photo_evaluation'})
    print(json.dumps({'phase':'preview','count':len(records),'sheets':(len(tiles)+5)//6}),flush=True)


def measure_predictions(predictions, metadata, size, thresholds=None, references=None):
    from .unified_evaluation import measure, summarize_rows
    rows=[]
    for index,(prediction,info) in enumerate(zip(predictions,metadata,strict=True)):
        arguments=dict(photo_size=info['photo_size'], source_size=info['source_size'],
            source_to_photo=info['source_to_photo'], inverse_letterbox=info['inverse_letterbox'],
            thresholds=thresholds,source_kind='unknown',condition=info.get('condition','noisy'),
            mode=info.get('mode','front'),geometry=info.get('mode','front') if info['kind']=='photo' else 'preserve',
            kind=info['kind'],preserve_kind=info['kind'] if info['kind']!='photo' else None,
            front_subgroup=('rotated' if info.get('pure_roll') else 'upright') if info['kind']=='photo' and info.get('mode')=='front' else None,
            occupancy=('30-50','50-75','75-90')[int(info['occupancy_bin'])] if info['kind']=='photo' else 'full_canvas',
            source_aspect=('portrait' if info['source_size'][0]/info['source_size'][1]<.8 else
                           'landscape' if info['source_size'][0]/info['source_size'][1]>1.25 else 'near_square'),
            source_index=info['source_index'])
        if references is not None:
            arguments['reference']=references[index]
        rows.append(measure(prediction,np.array(info['quad_input']),float(info['log_ratio']),size,**arguments))
    return summarize_rows(rows),rows


def actual_view_audit(metadata):
    counts=Counter(); bins={mode:Counter() for mode in ['front','mild','medium','hard']}
    source_counts=Counter();source_kinds={}
    appearance_counts=Counter();actual={m:[] for m in bins}
    input_stats={str(size):{'area_fraction':[],'short_edge_pixels':[]} for size in (224,384)}
    for info in metadata:
        counts['kind_'+info['kind']]+=1
        source_counts[info['source_index']]+=1
        source_kinds.setdefault(info['source_index'],set()).add('photo' if info['kind']=='photo' else 'preserve')
        if info['kind']=='photo':
            a=float(info['occupancy'])
            appearance_counts[info['appearance']]+=1
            tilt=float(info['tilt']);roll=float(info['roll']);mode=info['mode']
            bounds={'front':(0.,0.),'mild':(0.,8.),'medium':(8.,15.),'hard':(15.,25.)}
            low,high=bounds[mode]
            if not (tilt==0 if mode=='front' else low<tilt<=high):
                raise RuntimeError('Actual plane tilt violates its geometry stratum')
            if abs(roll)>{'front':12.,'mild':6.,'medium':12.,'hard':20.}[mode]+1e-6:
                raise RuntimeError('Actual roll exceeds its geometry stratum')
            if mode=='front' and ((info['pure_roll'] and abs(roll)<1.-1e-6) or (not info['pure_roll'] and roll!=0)):
                raise RuntimeError('Actual front rotation subgroup is inconsistent')
            if mode=='hard' and info['appearance'] not in ('clear','light'):
                raise RuntimeError('Hard geometry has excessive appearance degradation')
            actual[mode].append((tilt,roll,float(info.get('camera_attempts',1)),float(info['photo_size'][0]/info['photo_size'][1])))
            if not .30-1e-8 <= a <= .90+1e-8:
                raise RuntimeError('Actual rendered occupancy violates hard bound')
            index=0 if a<=.5 else (1 if a<=.75 else 2)
            if index!=int(info['occupancy_bin']):
                raise RuntimeError('Actual rendered occupancy differs from frozen slot')
            bins[info['mode']][index]+=1
            counts['photo_'+info['mode']]+=1
            counts['photo_above_half']+=int(a>.5)
            q=np.array(info['quad_photo']);w,h=info['photo_size']
            if not ((q>=0).all() and (q<=[w-1,h-1]).all()):
                raise RuntimeError('Rendered film corner is outside the photo')
            for size in (224,384):
                scale=(size-1)/max(w-1,h-1)
                input_stats[str(size)]['area_fraction'].append(a*w*h*scale**2/size**2)
                edges=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
                input_stats[str(size)]['short_edge_pixels'].append(float(edges.min()*scale))
    if len(metadata)==80000:
        if len(source_counts)!=4114 or not set(source_counts.values())<={19,20}:
            raise RuntimeError('Full training source coverage/19-or-20 slots violated')
        if any(v!={'photo','preserve'} for v in source_kinds.values()):
            raise RuntimeError('A training source lacks a preserve or photo view')
        if dict(appearance_counts)!={'clear':28800,'light':22400,'medium':9600,'strong':3200}:
            raise RuntimeError('Actual training photo appearance quota violated')
        if counts['photo_above_half']!=51200:
            raise RuntimeError('Actual training large-film quota violated')
        for mode,total in [('front',28800),('mild',22400),('medium',9600),('hard',3200)]:
            if dict(bins[mode])!={0:total//5,1:total*3//5,2:total//5}:
                raise RuntimeError(f'Actual joint occupancy quota violated: {mode}')
    def stats(values):
        return dict(zip(('min','p10','median','p90','max'),map(float,np.quantile(values,[0,.1,.5,.9,1])))) if values else {}
    return {'views':len(metadata),'source_count':len(source_counts),'counts':dict(counts),
            'actual_photo_appearance':dict(appearance_counts),
            'actual_pose':{mode:{name:stats([r[i] for r in rows]) for i,name in enumerate(('tilt','roll','attempts','photo_aspect'))} for mode,rows in actual.items()},
            'letterboxed_film':{size:{name:stats(v) for name,v in values.items()} for size,values in input_stats.items()},
            'actual_photo_occupancy_by_geometry':{k:dict(v) for k,v in bins.items()},
            'source_slots_min':min(source_counts.values()),'source_slots_max':max(source_counts.values()),
            'full_film_visibility_verified':True}


def cache_signature(args,split):
    return {'source_tree_sha256':assert_source(args.run)['tree_sha256'],
            'g0_sha256':sha256(args.g0_checkpoint),'manifest_sha256':sha256(args.manifest),
            'schedule_sha256':sha256(args.run/'schedules'/f'{split}.json'),
            'size':224,'dtype':'float32','features':1920}


def build_feature_cache(args, split, backbone, initial_head, legacy_model=None):
    from .unified_model import feature_predict,forward_features
    from .ratio_model import predict
    from .model import torch
    schedule=read_schedule(args.run,split)
    folder=args.run/'cache'/split;folder.mkdir(parents=True,exist_ok=True)
    signature=cache_signature(args,split)
    binding=folder/'binding.json'
    if binding.exists() and json.loads(binding.read_text())!=signature:
        raise RuntimeError('Feature cache binding changed')
    write_json(binding,signature)
    n=len(schedule)
    feature_path=folder/'features.npy';done_path=folder/'done.npy'
    if feature_path.exists()!=done_path.exists():
        raise RuntimeError('Feature cache data/completion files disagree; refusing partial reconstruction')
    features=np.lib.format.open_memmap(feature_path,mode='r+' if feature_path.exists() else 'w+',dtype=np.float32,shape=(n,1920))
    done=np.lib.format.open_memmap(done_path,mode='r+' if done_path.exists() else 'w+',dtype=np.bool_,shape=(n,))
    journal=folder/'metadata.jsonl'
    metadata={}
    if journal.exists():
        for line in journal.read_text().splitlines():
            record=json.loads(line);metadata[record['cache_index']]=record
    if any(bool(done[i]) and i not in metadata for i in range(n)):
        raise RuntimeError('Feature cache journal is incomplete')
    positions=sorted((i for i in range(n) if not done[i]),key=lambda i:(schedule[i]['source_index'],i))
    data=loader(ViewDataset(args.manifest,schedule,224,positions,legacy=legacy_model is not None),16,args.workers)
    old_predictions={}
    old_path=folder/'legacy-predictions.json'
    if old_path.exists():old_predictions={int(k):v for k,v in json.loads(old_path.read_text()).items()}
    started=time.perf_counter();previous=int(done.sum())
    for batch_number,(indices,pixels,quad,logr,inverse,serialized,legacy_pixels) in enumerate(data):
        f=feature_predict(backbone,pixels.numpy(),'cuda')
        if f.shape!=(len(indices),1920) or not np.isfinite(f).all():
            raise RuntimeError('Invalid cached encoder feature')
        ids=indices.numpy()
        features[ids]=f
        with journal.open('a') as log:
            for index,record in zip(ids,serialized,strict=True):
                info=json.loads(record);info['cache_index']=int(index)
                metadata[int(index)]=info;log.write(json.dumps(info)+'\n')
            log.flush();os.fsync(log.fileno())
        if legacy_model is not None:
            pred=predict(legacy_model,legacy_pixels.numpy(),'cuda')
            for index,value in zip(ids,pred,strict=True):old_predictions[int(index)]=value.tolist()
            write_json(old_path,old_predictions)
        features.flush();done[ids]=True;done.flush()
        completed=int(done.sum())
        if batch_number%20==0 or completed==n:
            elapsed=time.perf_counter()-started
            stamp(args.run/'status.json',phase='feature_cache',split=split,completed=completed,total=n,
                  views_per_second=(completed-previous)/max(elapsed,.001),free_disk_gib=disk_check(args.run))
            print(json.dumps({'phase':'feature_cache','split':split,'completed':completed,'total':n,
                              'views_per_second':round((completed-previous)/max(elapsed,.001),2)}),flush=True)
    ordered=[metadata[i] for i in range(n)]
    write_json(folder/'metadata.json',ordered)
    audit=actual_view_audit(ordered)
    write_json(folder/'realized-distributions.json',audit)
    if split=='train':
        record=json.loads((args.run/'data-audit.json').read_text())
        record.update(actual_training_views_audited=True,realized_train=audit)
        write_json(args.run/'data-audit.json',record)
    if split=='dev':
        predictions=[]
        with torch.no_grad():
            for start in range(0,n,256):
                predictions.append(forward_features(initial_head,torch.as_tensor(np.array(features[start:start+256]),device='cuda')).detach().cpu().numpy())
        g0=np.concatenate(predictions)
        np.save(folder/'G0-predictions.npy',g0)
        summary,rows=measure_predictions(g0,ordered,224)
        write_json(args.run/'G0-D.json',summary)
        write_json(folder/'G0-rows.json',rows)
        if legacy_model is not None:
            from .ratio_evaluation import predicted_quad
            from .ratio_synthesis import UNIT,normalization
            converted=[]
            for i,info in enumerate(ordered):
                if i not in old_predictions:raise RuntimeError('Missing B1.1 development prediction')
                old=predicted_quad(np.array(old_predictions[i]),224,True)
                q_photo=(old+.5)*np.array(info['photo_size'])/224-.5
                q_input=project(q_photo,np.linalg.inv(np.array(info['inverse_letterbox'])))
                q_norm=project(q_input,normalization(224,224))
                matrix=cv2.getPerspectiveTransform(UNIT.astype(np.float32),q_norm.astype(np.float32))
                converted.append(np.r_[matrix.reshape(-1)[:8],0])
            old_summary,_=measure_predictions(np.array(converted),ordered,224)
            write_json(args.run/'B1.1-D.json',old_summary)
    stamp(folder/'complete.json',complete=True,views=n,signature=signature,audit=audit)
    return features,ordered


def cache(args):
    from .unified_model import build_unified
    from .model import build
    environment=configure();inventory=freeze_source(args.run)
    write_json(args.run/'environment.json',environment)
    model,backbone,head=build_unified(args.weights,args.g0_checkpoint,224)
    model.to('cuda')
    build_feature_cache(args,'train',backbone,head)
    legacy,_=build(args.weights);legacy.to('cuda')
    build_feature_cache(args,'dev',backbone,head,legacy)
    stamp(args.run/'status.json',phase='cache_complete',source_tree_sha256=inventory['tree_sha256'])


def cached_split(args,split):
    folder=args.run/'cache'/split
    if not (folder/'complete.json').exists():
        raise RuntimeError(f'{split} features are incomplete')
    signature=cache_signature(args,split)
    if json.loads((folder/'binding.json').read_text())!=signature or json.loads((folder/'complete.json').read_text())['signature']!=signature:
        raise RuntimeError('Completed feature cache binding differs from the current frozen inputs')
    features=np.load(folder/'features.npy',mmap_mode='r')
    metadata=json.loads((folder/'metadata.json').read_text())
    done=np.load(folder/'done.npy',mmap_mode='r')
    if features.shape!=(len(metadata),1920) or not done.all() or len(done)!=len(metadata):
        raise RuntimeError('Completed feature cache shape or completion mismatch')
    return features,metadata


def load_development_pixels(args,size):
    schedule=read_schedule(args.run,'dev')
    images=np.empty((len(schedule),size,size,3),np.uint8)
    metadata=[None]*len(schedule)
    data=loader(ViewDataset(args.manifest,schedule,size),16 if size==224 else 4,args.workers)
    for indices,pixels,quad,logr,inverse,serialized,_ in data:
        for index,image,record in zip(indices.numpy(),pixels.numpy(),serialized,strict=True):
            images[index]=image;metadata[index]=json.loads(record)
        done=int(indices[-1])+1
        if done%400<16 or done==len(schedule):
            stamp(args.run/'status.json',phase='development_inputs',size=size,completed=done,total=len(schedule))
    return images,metadata


def memory():
    from .model import torch
    return {'gpu_allocated_mib':round(torch.cuda.memory_allocated()/2**20,1),
            'gpu_peak_reserved_mib':round(torch.cuda.max_memory_reserved()/2**20,1)}


def config_for(args,group):
    from .unified_model import contract
    inventory=assert_source(args.run)
    size,stage,updates,micro=GROUPS[group]
    result={**RECIPE,'group':group,'input_size':size,'stage':stage,'updates_this_group':updates,
            'microbatch':micro,'accumulation':64//micro,'contract':contract(size),
            'source_sha':inventory['git_base'],'source_tree_sha':inventory['tree_sha256'],
            'manifest_sha256':sha256(args.manifest),
            'initial_g0_sha256':sha256(args.g0_checkpoint),'weights_sha256':sha256(args.weights),
            'schedule_sha256':sha256(args.run/'schedules/train.json'),
            'development_schedule_sha256':sha256(args.run/'schedules/dev.json')}
    if group!='S224':
        selected=json.loads((args.run/'S224'/'best.json').read_text())
        result['stage1_checkpoint']=selected['checkpoint']
        result['stage1_checkpoint_sha256']=sha256(args.run/'S224'/selected['checkpoint'])
    return result


def prediction_batches(model,head,data,cached,size):
    from .model import torch
    from .unified_model import forward_features,predict
    result=[]
    with torch.no_grad():
        batch=256 if cached else (16 if size==224 else 4)
        for start in range(0,len(data),batch):
            if cached:
                p=forward_features(head,torch.as_tensor(np.array(data[start:start+batch]),device='cuda')).detach().cpu().numpy()
            else:p=predict(model,data[start:start+batch],'cuda')
            result.append(p)
    return np.concatenate(result)


def commit_evaluation(folder, step, predictions, result):
    """Publish predictions first; the bound evaluation JSON commits the pair."""
    folder = Path(folder)
    checkpoint = folder/f'step-{step:04d}.pt'
    sidecar = json.loads(checkpoint.with_suffix('.json').read_text())
    checkpoint_sha = sha256(checkpoint)
    if sidecar.get('step') != step or sidecar.get('sha256') != checkpoint_sha or result.get('step') != step:
        raise RuntimeError('Evaluation does not match its saved checkpoint')
    predictions = np.asarray(predictions)
    if predictions.ndim != 2 or predictions.shape[1] != 9 or not len(predictions) or not np.isfinite(predictions).all():
        raise RuntimeError('Invalid development prediction array')
    path = folder/f'predictions-{step:04d}.npy'
    temporary = path.with_suffix('.npy.tmp')
    with temporary.open('wb') as output:
        np.save(output, predictions, allow_pickle=False)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    committed = {**result, 'checkpoint_sha256': checkpoint_sha,
                 'predictions_sha256': sha256(path), 'predictions_shape': list(predictions.shape)}
    write_json(folder/f'evaluation-{step:04d}.json', committed)
    return committed


def completed_evaluations(folder, completed, development_views):
    """Ignore an interrupted latest evaluation, but never silently drop history."""
    folder = Path(folder)
    evaluations = {}
    for step in range(250, completed + 1, 250):
        checkpoint = folder/f'step-{step:04d}.pt'
        record_path = folder/f'evaluation-{step:04d}.json'
        predictions_path = folder/f'predictions-{step:04d}.npy'
        valid = False
        try:
            sidecar = json.loads(checkpoint.with_suffix('.json').read_text())
            result = json.loads(record_path.read_text())
            predictions = np.load(predictions_path, mmap_mode='r', allow_pickle=False)
            checkpoint_sha = sha256(checkpoint)
            valid = (sidecar.get('step') == result.get('step') == step
                     and sidecar.get('sha256') == result.get('checkpoint_sha256') == checkpoint_sha
                     and predictions.shape == (development_views, 9)
                     and result.get('predictions_shape') == [development_views, 9]
                     and result.get('predictions_sha256') == sha256(predictions_path)
                     and np.isfinite(predictions).all())
        except (OSError, ValueError, EOFError):
            pass
        if valid:
            evaluations[step] = result
        elif step != completed:
            raise RuntimeError(f'Incomplete earlier development evaluation at step {step}; restore it before continuing')
    return evaluations


def rebuild_best(folder, evaluations):
    """Best is derived from committed evaluations on every run, including resume."""
    path = Path(folder)/'best.json'
    if not evaluations:
        path.unlink(missing_ok=True)
        return
    best = min(evaluations.values(), key=lambda r: (*r['selection_score'], r['step']))
    write_json(path, {'step': best['step'], 'checkpoint': f"step-{best['step']:04d}.pt",
                     'score': best['selection_score'], 'evaluation': f"evaluation-{best['step']:04d}.json"})


def check_training_manifest(args):
    before = json.loads((args.run/'original-integrity-before.json').read_text())
    if sha256(args.manifest) != before['manifest_sha256']:
        raise RuntimeError('Training manifest differs from the audited original sources')


def resume_checkpoint(folder, updates):
    """Use the last committed optimizer state; retain interrupted writes locally."""
    folder = Path(folder)
    complete, interrupted = [], []
    for checkpoint in sorted(folder.glob('step-*.pt')):
        step = int(checkpoint.stem.removeprefix('step-'))
        if step <= 0 or step > updates or step % 250:
            raise RuntimeError('Checkpoint step is outside the approved training budget')
        sidecar = checkpoint.with_suffix('.json')
        if not sidecar.exists():
            interrupted.append(checkpoint)
            continue
        record = json.loads(sidecar.read_text())
        if record.get('sha256') != sha256(checkpoint) or record.get('step') != step:
            raise RuntimeError('Resume checkpoint SHA or step mismatch')
        complete.append(checkpoint)
    latest = complete[-1] if complete else None
    if latest is not None and any(path.name < latest.name for path in interrupted):
        raise RuntimeError('Incomplete earlier checkpoint; restore its sidecar before continuing')
    interrupted.extend(sorted(folder.glob('step-*.tmp')))
    if interrupted:
        archive = folder/'interrupted'/f'checkpoint-write-{time.time_ns()}'
        archive.mkdir(parents=True)
        for path in interrupted:
            path.rename(archive/path.name)
        write_json(archive/'recovery.json', {
            'preserved_files': [path.name for path in interrupted],
            'resume_checkpoint': latest.name if latest is not None else None,
            'sample_count_basis': 'durable_optimizer_steps_times_64',
            'discarded_updates_may_require_repeated_computation': True})
    return latest


def group_train(args):
    from .model import torch,keras
    from .unified_model import build_unified,load_unified,parameter_groups,losses,tensor_input,forward_features
    from .unified_evaluation import selection_score,admission_gates
    from .ratio_runner import save_checkpoint
    check_training_manifest(args)
    configure();config=config_for(args,args.group)
    size,stage,updates,micro=GROUPS[args.group]
    folder=args.run/args.group;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'config.json').exists() and json.loads((folder/'config.json').read_text())!=config:
        raise RuntimeError('Cannot resume with changed training configuration')
    write_json(folder/'config.json',config)
    checkpoint=resume_checkpoint(folder,updates)
    if args.group=='S224':model,backbone,head=build_unified(args.weights,args.g0_checkpoint,size)
    else:model,backbone,head=load_unified(args.weights,args.g0_checkpoint,args.run/'S224'/config['stage1_checkpoint'],size=size)
    model.to('cuda')
    groups=parameter_groups(model,backbone,head,stage)
    parameters=[p for group in groups for p in group['params']]
    optimizer=torch.optim.Adam(groups)
    completed=0
    if checkpoint is not None:
        expected=json.loads(checkpoint.with_suffix('.json').read_text())
        if sha256(checkpoint)!=expected['sha256']:raise RuntimeError('Resume checkpoint SHA mismatch')
        payload=torch.load(checkpoint,map_location='cuda',weights_only=True)
        if payload['config']!=config or payload['step']!=expected['step']:
            raise RuntimeError('Resume checkpoint configuration or step differs')
        model.load_state_dict(payload['state_dict'],strict=True)
        optimizer.load_state_dict(payload['optimizer']);completed=payload['step']
        torch.set_rng_state(payload['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in payload['cuda_rng']])
        del payload
    bn_before={v.path:v.value.detach().cpu().clone() for layer in backbone.layers if isinstance(layer,keras.layers.BatchNormalization) for v in layer.weights}
    if stage=='head':
        train_features,train_metadata=cached_split(args,'train')
        dev_data,dev_metadata=cached_split(args,'dev')
    else:
        dev_data,dev_metadata=load_development_pixels(args,size)
    baseline=json.loads((args.run/'G0-D.json').read_text())
    from .unified_evaluation import prepare_reference
    references=[prepare_reference(v['photo_size'],v['source_size'],v['source_to_photo'],v['quad_photo']) for v in dev_metadata]
    evaluations=completed_evaluations(folder,completed,len(dev_metadata))
    rebuild_best(folder,evaluations)
    def evaluate(step):
        stamp(args.run/'status.json',phase='evaluating',group=args.group,optimizer_step=step,samples_seen=step*64)
        started=time.perf_counter()
        predictions=prediction_batches(model,head,dev_data,stage=='head',size)
        summary,rows=measure_predictions(predictions,dev_metadata,size,references=references)
        score=list(selection_score(summary,baseline))
        result={'step':step,'summary':summary,'selection_score':score,
                'gates':admission_gates(summary,baseline),'evaluation_seconds':time.perf_counter()-started}
        result=commit_evaluation(folder,step,predictions,result)
        evaluations[step]=result
        rebuild_best(folder,evaluations)
        print(json.dumps({'phase':'evaluation','group':args.group,'step':step,'score':score,
                          'evaluation_seconds':result['evaluation_seconds']}),flush=True)
    if completed and completed not in evaluations:evaluate(completed)
    if completed==updates:
        stamp(folder/'status.json',phase='complete',group=args.group,optimizer_step=completed,samples_seen=completed*64)
        return
    history=folder/'history.jsonl'
    if history.exists():
        records=[json.loads(line) for line in history.read_text().splitlines() if line]
        if any(r['optimizer_step']>completed for r in records):
            history.rename(folder/f'interrupted-history-{time.time_ns()}.jsonl')
            history.write_text(''.join(json.dumps(r)+'\n' for r in records if r['optimizer_step']<=completed))
    if stage=='last_block':
        schedule=read_schedule(args.run,'train')
        training=iter(loader(ViewDataset(args.manifest,schedule,size,range(completed*64,80000)),micro,args.workers))
    start=time.perf_counter()
    for step in range(completed+1,updates+1):
        optimizer.zero_grad(set_to_none=True)
        means=np.zeros(3);step_start=time.perf_counter()
        for k in range(64//micro):
            if stage=='head':
                indices=np.arange((step-1)*64,step*64)%80000
                values=torch.as_tensor(np.array(train_features[indices]),device='cuda')
                infos=[train_metadata[int(i)] for i in indices]
                q=torch.tensor(np.array([v['quad_input'] for v in infos]),device='cuda',dtype=torch.float32)
                lr=torch.tensor([v['log_ratio'] for v in infos],device='cuda',dtype=torch.float32)
                inverse=torch.tensor(np.array([v['inverse_letterbox'] for v in infos]),device='cuda',dtype=torch.float32)
                prediction=forward_features(head,values)
            else:
                _,pixels,q,lr,inverse,_,_=next(training)
                q=q.to('cuda');lr=lr.to('cuda');inverse=inverse.to('cuda')
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    prediction=model(tensor_input(pixels.numpy(),'cuda'),training=False)
            total,lq,lratio=losses(prediction,q,lr,size,inverse_letterbox=inverse)
            (total/(64//micro)).backward()
            means += [float(total.detach()),float(lq.detach()),float(lratio.detach())]
        gradient=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in parameters):raise FloatingPointError('Non-finite updated weights')
        means/=64//micro
        record={'group':args.group,'optimizer_step':step,'samples_seen':step*64,'loss':float(means[0]),
                'quad_loss':float(means[1]),'ratio_loss':float(means[2]),'gradient_norm':gradient,
                'seconds_per_update':time.perf_counter()-step_start,'elapsed_seconds':time.perf_counter()-start,
                'eta_training_seconds':(updates-step)*(time.perf_counter()-start)/(step-completed),**memory()}
        with history.open('a') as log:log.write(json.dumps(record)+'\n')
        stamp(folder/'status.json',phase='training',**record)
        stamp(args.run/'status.json',phase='training',**record)
        if step%10==0 or step==1:print(json.dumps(record),flush=True)
        if step%250==0:
            assert_source(args.run)
            disk_check(args.run)
            for layer in backbone.layers:
                if isinstance(layer,keras.layers.BatchNormalization):
                    for v in layer.weights:
                        if not torch.equal(bn_before[v.path],v.value.detach().cpu()):raise RuntimeError('Frozen BatchNorm changed')
            save_checkpoint(model,optimizer,folder/f'step-{step:04d}.pt',step,stage,config,Counter(samples=step*64))
            evaluate(step)
    stamp(folder/'status.json',phase='complete',group=args.group,optimizer_step=updates,samples_seen=updates*64,BN_unchanged=True)


def integrity_after(args):
    from .unified_geometry import plan_transform,apply_transform
    sources=Sources(args.manifest,cache_size=1)
    before=json.loads((args.run/'original-integrity-before.json').read_text())
    if sha256(args.manifest)!=before['manifest_sha256']:
        raise RuntimeError('Original manifest changed during the experiment')
    for ordinal,record in enumerate(sources.records):
        if sha256(record['source_path'])!=record['source_sha256']:
            raise RuntimeError(f'Original source ordinal {ordinal} changed during the experiment')
        pixels=sources.read(ordinal)
        plan=plan_transform((pixels.shape[1],pixels.shape[0]),None,None,source_kind='reliable_digital')
        preserved=apply_transform(pixels,plan)
        if preserved is not pixels or not np.array_equal(preserved,pixels):
            raise RuntimeError('Reliable original pass-through changed actual image values')
        if (ordinal+1)%500==0:
            print(json.dumps({'phase':'original_integrity_after','completed':ordinal+1,'total':len(sources.records)}),flush=True)
    result={'originals_checked':len(sources.records),'all_hashes_unchanged':True,
            'reliable_source_passthrough_count':len(sources.records),'reliable_source_values_unchanged':True,
            'manifest_sha256':sha256(args.manifest)}
    write_json(args.run/'original-integrity-after.json',result)
    return result


def run_all(args):
    # A fresh child per stage resets CUDA/optimizer/RNG and the PyTorch interop pool.
    phases=[('cache',None),('benchmark',None),*[('train',g) for g in GROUPS],('finalize',None),('audit-after',None)]
    for command,group in phases:
        arguments=[sys.executable,'-m','ptrn_hip.unified_runner',command]
        for name in ('manifest','prior_run','run','weights','g0_checkpoint','real_dir'):
            value=getattr(args,name,None)
            if value is not None:arguments += ['--'+name.replace('_','-'),str(value.resolve())]
        arguments += ['--workers',str(args.workers)]
        if group is not None:arguments += ['--group',group]
        code=subprocess.call(arguments,env={**os.environ,'KERAS_BACKEND':'torch'})
        if code:
            stamp(args.run/'supervisor-status.json',phase='failed',command=command,group=group,returncode=code)
            raise RuntimeError(f'Experiment stopped in {command}/{group}: child exit {code}')
    stamp(args.run/'supervisor-status.json',phase='training_and_evaluation_complete',
          total_formal_exposures=560000,visual_review_requires_observations=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','preview','freeze','cache','benchmark','train','finalize','audit-after','run-all'])
    for name in ('manifest','prior-run','run','weights','g0-checkpoint'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--real-dir',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--group',choices=list(GROUPS),default='S224')
    args=parser.parse_args()
    if '.local' not in args.run.resolve().parts:
        parser.error('All experiment artifacts must stay under an ignored .local directory')
    if args.workers<0:parser.error('workers must be nonnegative')
    args.run.mkdir(parents=True,exist_ok=True)
    try:
        if args.command=='prepare':prepare(args)
        elif args.command=='preview':preview(args)
        elif args.command=='freeze':freeze_source(args.run)
        elif args.command=='cache':cache(args)
        elif args.command=='benchmark':
            from .unified_benchmark import benchmark
            benchmark(args)
        elif args.command=='train':group_train(args)
        elif args.command=='finalize':
            if args.real_dir is None:parser.error('finalize requires --real-dir')
            from .unified_finalize import finalize
            finalize(args)
        elif args.command=='audit-after':integrity_after(args)
        else:run_all(args)
    except Exception as error:
        stamp(args.run/'failure.json',command=args.command,group=args.group,
              exception_type=type(error).__name__,message=str(error))
        raise


if __name__=='__main__':
    main()
