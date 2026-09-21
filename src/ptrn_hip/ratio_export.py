"""Explicit ratio-v1 TorchScript export and unpaired real-photo visual review."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import warnings
import zipfile
import cv2
import numpy as np
from .data import read_rgb,write_json
from .ratio_data import sha256
from .ratio_synthesis import letterbox,project
from .ratio_evaluation import predicted_quad,edge_ratio


def clean_archive(source,target):
    markers=(b'/home/',b'/root/',b'C:\\Users\\')
    removed=0
    with zipfile.ZipFile(source) as original,zipfile.ZipFile(target,'x') as clean:
        for item in original.infolist():
            if item.filename.endswith('.debug_pkl'):
                removed+=1;continue
            content=original.read(item)
            if any(mark in content or mark in item.filename.encode() for mark in markers):
                raise ValueError('Private path found outside debug records')
            clean.writestr(item,content)
    return removed


def load_candidate(weights,checkpoint):
    from .ratio_model import torch,build_ratio
    expected=json.loads(checkpoint.with_suffix('.json').read_text())
    if sha256(checkpoint)!=expected['sha256']:
        raise ValueError('Candidate checkpoint SHA mismatch')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=payload['config']
    if expected['config']!=config:
        raise ValueError('Candidate configuration mismatch')
    model,_=build_ratio(weights,config['input_size'],config['ratio_head'],config['mean_train_log_ratio'])
    model.load_state_dict(payload['state_dict'],strict=True)
    return model,config


def export(args):
    from .model import configure,torch
    from .ratio_model import predict,contract
    versions=configure(False)
    cv2.setNumThreads(1)
    model,config=load_candidate(args.weights,args.checkpoint)
    model.to('cpu')
    size=config['input_size']
    args.output.mkdir(parents=True,exist_ok=False)
    class Wrapped(torch.nn.Module):
        def __init__(self,base):
            super().__init__();self.base=base
            self.register_buffer('mean',torch.tensor([.485,.456,.406]))
            self.register_buffer('std',torch.tensor([.229,.224,.225]))
        def forward(self,pixels):
            return self.base((pixels/255-self.mean)/self.std,training=False)
    wrapped=Wrapped(model).eval()
    with torch.no_grad(),warnings.catch_warnings():
        warnings.simplefilter('ignore',torch.jit.TracerWarning)
        traced=torch.jit.trace(wrapped,torch.zeros((1,size,size,3)),check_trace=True,strict=True)
    metadata={**contract(size,config['ratio_head']),'training_source_sha':config['source_sha'],
              'checkpoint_sha256':sha256(args.checkpoint),'export_source_sha':args.export_source_sha,
              'weights_redistribution_approved':False,'windows_verified':False,'versions':versions}
    raw=args.output/'trace-raw.local.pt';target=args.output/'ptrn-ratio-v1.torchscript.pt'
    torch.jit.save(traced,str(raw),_extra_files={'contract.json':json.dumps(metadata)})
    removed=clean_archive(raw,target)
    extra={'contract.json':''}
    saved=torch.jit.load(str(target),map_location='cpu',_extra_files=extra).eval()
    if json.loads(extra['contract.json'])!=metadata:
        raise ValueError('Export metadata mismatch')
    if not all(torch.equal(value,saved.state_dict()[key]) for key,value in traced.state_dict().items()):
        raise ValueError('Sanitized trace state changed')
    rng=np.random.default_rng(20260921)
    arrays=[rng.integers(0,256,(size,size,3),dtype=np.uint8),np.zeros((size,size,3),np.uint8)]
    originals=[]
    if args.real_dir:
        paths=sorted(p for p in args.real_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
        for path in paths:
            before=sha256(path)
            try:pixels=read_rgb(path)
            except Exception as error:raise RuntimeError(type(error).__name__) from None
            arrays.append(letterbox(pixels,size)[0]);originals.append((path,before))
    maximum=0.;records=[]
    for index,pixels in enumerate(arrays):
        expected=predict(model,pixels[None],'cpu')
        with torch.no_grad():actual=saved(torch.tensor(pixels[None],dtype=torch.float32)).numpy()
        if actual.shape!=(1,9 if config['ratio_head'] else 8) or not np.isfinite(actual).all():
            raise ValueError('Export shape/finiteness mismatch')
        error=float(np.max(np.abs(actual-expected)))
        np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-5)
        maximum=max(maximum,error);records.append({'ordinal':index,'max_abs_error':error})
    performance={}
    pixels=torch.tensor(arrays[0][None],dtype=torch.float32)
    for threads in (2,4):
        torch.set_num_threads(threads)
        timings=[]
        with torch.no_grad():
            for i in range(13):
                started=time.perf_counter();saved(pixels)
                if i>=3:timings.append((time.perf_counter()-started)*1000)
        performance[str(threads)]={'warm_p50_ms':float(np.median(timings)),'warm_p95_ms':float(np.percentile(timings,95)),'measured':10,'warmup':3}
    if any(sha256(p)!=digest for p,digest in originals):
        raise ValueError('Read-only real photo source changed')
    missing_failed=False
    try:torch.jit.load(str(args.output/'intentionally-missing.pt'),map_location='cpu')
    except (ValueError,RuntimeError,OSError):missing_failed=True
    if not missing_failed:raise AssertionError('Missing resource unexpectedly loaded')
    report={'metadata':metadata,'artifact_sha256':sha256(target),'removed_debug_records':removed,
            'state_equal_after_sanitizing':True,'comparison_count':len(records),'max_absolute_output_difference':maximum,
            'records':records,'cpu_forward':performance,'missing_resource_fails':True,
            'real_photo_scope':'unpaired_development_visual_only','real_ratio_accuracy_verified':False}
    write_json(args.output/'export-report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


def visual(args):
    from .model import configure,torch,build
    from .ratio_model import predict
    configure(False);cv2.setNumThreads(1)
    if not args.real_dir:
        raise ValueError('Explicit authorized real-photo directory required')
    model,config=load_candidate(args.weights,args.checkpoint)
    baseline,_=build(args.weights)
    size=config['input_size']
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    paths=sorted(p for p in args.real_dir.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg'})
    if not paths:raise ValueError('No real photo inputs')
    for ordinal,path in enumerate(paths):
        before=sha256(path)
        try:pixels=read_rgb(path)
        except Exception as error:raise RuntimeError(type(error).__name__) from None
        height,width=pixels.shape[:2]
        input_pixels,lb=letterbox(pixels,size)
        prediction=predict(model,input_pixels[None],'cpu')[0]
        legacy_input=cv2.resize(pixels,(224,224),interpolation=cv2.INTER_AREA)
        old=predict(baseline,legacy_input[None],'cpu')[0]
        quad=project(predicted_quad(prediction,size),np.linalg.inv(lb))
        old_quad=(predicted_quad(old,224,True)+.5)*[width/224,height/224]-.5
        overlays=[]
        for name,q in (('B1.1',old_quad),(config['group'],quad)):
            image=pixels.copy()
            cv2.polylines(image,[np.rint(q).astype(np.int32)],True,(0,255,0),max(1,round(width/500)))
            thumb=letterbox(image,512)[0]
            cv2.putText(thumb,name,(10,24),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,180,0),2)
            overlays.append(thumb)
        cv2.imwrite(str(args.output/f'photo-{ordinal+1:03d}.local.jpg'),cv2.cvtColor(np.concatenate(overlays,axis=1),cv2.COLOR_RGB2BGR))
        rows.append({'ordinal':ordinal+1,'candidate_ratio':float(np.exp(prediction[8])) if len(prediction)==9 else float(edge_ratio(quad)),
                     'same_corners_max_edge_ratio':float(edge_ratio(quad)),'B11_max_edge_ratio':float(edge_ratio(old_quad)),
                     'ratio_evidence':'model_estimated','ground_truth_available':False})
        if sha256(path)!=before:raise ValueError('Read-only input changed')
    write_json(args.output/'real-photo-review.json',{'count':len(rows),'scope':'unpaired_development_visual_only',
               'accuracy_claimed':False,'independent_validation':False,'source_sha':config['source_sha'],'records':rows})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['export','visual'])
    parser.add_argument('--weights',type=Path,required=True);parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--real-dir',type=Path)
    parser.add_argument('--export-source-sha',required=True)
    args=parser.parse_args()
    # CPU export process is isolated from the CUDA training process.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    if args.action=='export':export(args)
    else:visual(args)


if __name__=='__main__':main()
