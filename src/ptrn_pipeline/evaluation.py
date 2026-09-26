"""Ordered prediction producer and bounded asynchronous CPU metric consumers."""
import argparse,concurrent.futures as cf,multiprocessing as mp,os,time,json,gc
from pathlib import Path
import numpy as np
from .cache import DiskCache,cached_scene,scene_key,unpack_view,digest
from .telemetry import atomic_json

_DATA=None;_CACHE=None;_INPUTS=None

def init_worker(bundle,backgrounds,cache_root,split='dev',size=224,seed=20260925):
    global _DATA,_CACHE,_INPUTS
    import cv2,torch
    from ptrn_hip.partial_data import PartialDataset
    cv2.setNumThreads(1);torch.set_num_threads(1)
    _DATA=PartialDataset(bundle,backgrounds,split=split,size=size,seed=seed,verify=False)
    name=digest(_DATA.fingerprint)[:16]
    _CACHE=DiskCache(cache_root,'dev-full-'+name,budget=10*2**30)
    _INPUTS=DiskCache(cache_root,'dev-input-'+name,budget=2*2**30)

def prepare(index):
    key,arrays,meta,hit=cached_scene(_DATA,index,_CACHE)
    inputs={k[6:]:v for k,v in arrays.items() if k.startswith('input_')}
    _INPUTS.put(scene_key(_DATA,index,full=False),inputs,{'index':index})
    return {'index':index,'hit':hit,'input_bytes':sum(v.nbytes for v in inputs.values())}

def metric(job):
    from ptrn_hip.partial_evaluation import measure
    from ptrn_hip.ratio_synthesis import project
    index,quad,ratio,baseline_ratio=job
    _,arrays,meta,_=cached_scene(_DATA,index,_CACHE)
    view=unpack_view(arrays,meta);q=project(np.asarray(quad),arrays['input_inverse_letterbox'])
    row=measure(view,q,float(ratio))
    row.update(index=index,ratio_baseline_error=abs(float(baseline_ratio)/view.ratio-1))
    return row

def memory_pressure():
    p=Path('/sys/fs/cgroup/memory.current')
    return p.exists() and int(p.read_text())>48*2**30

def bounded_map(pool,fn,items,limit=24):
    it=iter(items);pending=[]
    for _ in range(limit):
        try:pending.append(pool.submit(fn,next(it)))
        except StopIteration:break
    while pending:
        f=pending.pop(0);yield f.result()
        while memory_pressure():time.sleep(.25)
        try:pending.append(pool.submit(fn,next(it)))
        except StopIteration:pass

def prepare_cache(bundle,backgrounds,cache_root,output,*,workers=8,limit=None):
    from ptrn_hip.partial_data import PartialDataset
    data=PartialDataset(bundle,backgrounds,split='dev');count=min(limit,len(data)) if limit else len(data)
    output=Path(output);output.mkdir(parents=True,exist_ok=True);start=time.time();hits=0
    with cf.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init_worker,
           initargs=(str(bundle),str(backgrounds),str(cache_root))) as pool:
        for n,row in enumerate(bounded_map(pool,prepare,range(count),workers*2),1):
            hits+=row['hit']
            if n%64==0 or n==count:atomic_json(output/'progress.json',{'phase':'cache_prepare','completed':n,'count':count,'hits':hits,'seconds':time.time()-start})
    result={'count':count,'hits':hits,'seconds':time.time()-start,'fingerprint':data.fingerprint}
    atomic_json(output/'complete.json',result);return result

def load_model(bundle,checkpoint):
    import torch
    from ptrn_hip.partial_model import build_partial
    model=build_partial(Path(bundle)/'weights/b11.weights.h5',Path(bundle)/'weights/g0.pt',stage='head').to('cuda')
    cp=torch.load(checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(cp['model'],strict=True)
    return model,cp

def inference(model,items):
    import torch
    # Preserve the original FP32 / TF32 settings. Ratios use torch.exp, as the reference.
    x=torch.as_tensor(np.stack([v['pixels'] for v in items]),device='cuda',dtype=torch.float32)/255
    out=model((x-x.new_tensor([.485,.456,.406]))/x.new_tensor([.229,.224,.225]))
    return {'quad':out['quad'].detach().cpu().numpy(),'log_ratio':out['log_ratio'].detach().cpu().numpy(),
            'log_ratio_baseline':out['log_ratio_baseline'].detach().cpu().numpy(),
            'ratio':out['log_ratio'].exp().detach().cpu().numpy(),'baseline_ratio':out['log_ratio_baseline'].exp().detach().cpu().numpy()}

def predict(bundle,backgrounds,cache_root,checkpoint,output,*,batch=1,metric_workers=4,limit=None,use_graph=False):
    import torch,cv2
    from ptrn_hip.partial_data import PartialDataset,sha256
    torch.set_num_threads(4);cv2.setNumThreads(1)
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Prediction output must be fresh; existing checkpoint results cannot be reused')
    output.mkdir(parents=True,exist_ok=True)
    data=PartialDataset(bundle,backgrounds,split='dev',verify=False)
    from .input_bank import InputBank
    bank=InputBank(data,cache_root,limit)
    model,cp=load_model(bundle,checkpoint)
    from .graph import GraphPredictor
    graph=GraphPredictor(model,batch) if use_graph else None
    assert cp['binding']['development']==data.fingerprint
    count=min(limit,len(data)) if limit else len(data);name=digest(data.fingerprint)[:16]
    inputs_cache=DiskCache(cache_root,'dev-input-'+name,budget=2*2**30)
    start=time.time();predictions=[]
    # CPU metrics run independently from GPU inference and survive its exit.
    command=[os.sys.executable,'-m','ptrn_pipeline.evaluation','metrics','--bundle',str(bundle),'--backgrounds',str(backgrounds),
             '--cache',str(cache_root),'--output',str(output),'--workers',str(metric_workers),'--count',str(count)]
    import subprocess
    with (output/'metrics-console.log').open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    atomic_json(output/'binding.json',{'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'step':cp['step'],
        'development':data.fingerprint,'runtime_code':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},'batch':batch,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
        'prediction_backend':'cuda_graph' if use_graph else 'eager','rebuilt_input_cache_entries':bank.rebuilt,'metric_pid':child.pid,'producer_pid':os.getpid(),'producer_identity':producer_identity(os.getpid()),'count':count})
    try:
        with torch.no_grad():
            for first in range(0,count,batch):
                ids=list(range(first,min(count,first+batch)));items=[]
                for i in ids:
                    items.append(bank[i])
                pred=graph(items) if graph is not None and len(items)==batch else inference(model,items)
                for j,i in enumerate(ids):predictions.append({'index':i,**{k:v[j].tolist() for k,v in pred.items()}})
                # Publish each bounded prediction shard atomically, never an incomplete file.
                if len(predictions)>=128 or first+batch>=count:
                    shard=output/f'pred-{predictions[0]["index"]:06d}.json';atomic_json(shard,predictions);predictions=[]
                    atomic_json(output/'progress.json',{'phase':'inference','completed':ids[-1]+1,'count':count,'seconds':time.time()-start})
                if first%128==0:gc.collect()
        atomic_json(output/'inference-complete.json',{'count':count,'seconds':time.time()-start,'checkpoint_sha256':sha256(checkpoint)})
    except BaseException as error:
        atomic_json(output/'inference-failed.json',{'error':repr(error)});raise
    finally:graph=None;del model;gc.collect();torch.cuda.empty_cache()
    return {'inference_seconds':time.time()-start,'metric_pid':child.pid,'output':str(output)}

def producer_identity(pid):
    try:return Path(f'/proc/{int(pid)}/stat').read_text().split()[21]
    except (OSError,ValueError,IndexError):return None

def assert_producer_alive(output):
    if (output/'inference-complete.json').exists():return
    from .telemetry import read_json
    binding=read_json(output/'binding.json',{})
    if binding.get('producer_pid'):
        actual=producer_identity(binding['producer_pid'])
        if actual is None or actual!=binding.get('producer_identity'):
            raise RuntimeError('Prediction producer exited before completing all samples')


def consume_metrics(bundle,backgrounds,cache_root,output,workers,count):
    import fcntl
    output=Path(output);seen=set();rows={};start=time.time();pending={};failed=None
    lock=(output/'metrics.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        with cf.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init_worker,
             initargs=(str(bundle),str(backgrounds),str(cache_root))) as pool:
            while len(rows)<count:
                assert_producer_alive(output)
                if (output/'inference-failed.json').exists():raise RuntimeError('Prediction producer failed')
                for p in sorted(output.glob('pred-*.json')):
                    if p.name in seen:continue
                    if len(pending)>=256:break
                    chunk=json.loads(p.read_text());seen.add(p.name)
                    for v in chunk:
                        i=v['index']
                        if i in rows or i in pending:raise ValueError('Duplicate predicted sample')
                        pending[i]=pool.submit(metric,(i,v['quad'],v['ratio'],v['baseline_ratio']))
                ready=[i for i,f in pending.items() if f.done()]
                for i in ready:rows[i]=pending.pop(i).result()
                if ready:atomic_json(output/'metrics-progress.json',{'completed':len(rows),'count':count,'pending':len(pending),'seconds':time.time()-start})
                if (output/'inference-complete.json').exists() and not pending and not ready and len(rows)<count:
                    raise ValueError('Incomplete prediction coverage')
                if not ready:time.sleep(.1)
        if sorted(rows)!=list(range(count)):raise ValueError('Missing sample')
        ordered=[rows[i] for i in range(count)];groups={}
        for key in ('kind','integrity','background_family'):
            for val in sorted({r[key] for r in ordered}):
                subset=[r for r in ordered if r[key]==val];groups[f'{key}/{val}']={'count':len(subset),'valid_geometry':sum(r['valid_geometry'] for r in subset)}
        binding=json.loads((output/'binding.json').read_text())
        result={'scope':'exposed development; diagnostic only; no automatic-success policy asserted','count':count,
          'valid_geometry':sum(r['valid_geometry'] for r in ordered),'invalid_geometry':sum(not r['valid_geometry'] for r in ordered),
          'full_development':count==5736,'rows':ordered,'strata':groups,'execution':{**binding,'metrics_seconds':time.time()-start}}
        atomic_json(output/'evaluation.json',result);atomic_json(output/'metrics-complete.json',{'count':count,'seconds':time.time()-start})
    except BaseException as error:atomic_json(output/'metrics-failed.json',{'error':repr(error)});raise

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','predict','metrics']);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--backgrounds',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--batch',type=int,default=1);p.add_argument('--graph',action='store_true');p.add_argument('--strict-fp32',action='store_true');p.add_argument('--checkpoint',type=Path);p.add_argument('--count',type=int);a=p.parse_args()
    if a.strict_fp32:
        import torch
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if a.action=='prepare':print(json.dumps(prepare_cache(a.bundle,a.backgrounds,a.cache,a.output,workers=a.workers,limit=a.count)))
    elif a.action=='predict':
        from .resources import gpu_lease
        with gpu_lease(a.cache.parent):print(json.dumps(predict(a.bundle,a.backgrounds,a.cache,a.checkpoint,a.output,batch=a.batch,metric_workers=a.workers,limit=a.count,use_graph=a.graph)))
    else:consume_metrics(a.bundle,a.backgrounds,a.cache,a.output,a.workers,a.count)
if __name__=='__main__':main()
