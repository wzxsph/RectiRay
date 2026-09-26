"""Checkpoint-compatible training with pre-generated inputs and interleaved validation."""
import argparse,json,os,gc,time,fcntl,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .cache import digest
from .telemetry import atomic_json,read_json
from .training import TrainCacheDataset,DoubleBuffer,train_step
from .async_eval import EvaluationJob

def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--backgrounds',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--resume',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=5000);p.add_argument('--gc-interval',type=int,default=1);p.add_argument('--eval-batch',type=int,default=1);p.add_argument('--eval-current',action='store_true');p.add_argument('--diagnostic-eval-count',type=int);a=p.parse_args()
 from ptrn_hip.partial_data import PartialDataset,sha256
 from ptrn_hip.partial_runner import restore_checkpoint,save_checkpoint,worker_init
 from .evaluation import load_model
 torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=True)
 from .resources import gpu_lease
 lease=gpu_lease(a.cache.parent);lease.__enter__()
 lock=(a.output/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cp=torch.load(a.resume,map_location='cpu',weights_only=True);binding=cp['binding'];assert binding['effective_batch']==64 and binding['stage']=='head'
 data=PartialDataset(a.bundle,a.backgrounds);dev=PartialDataset(a.bundle,a.backgrounds,split='dev')
 assert all(binding[k]==v for k,v in data.fingerprint.items());assert binding['development']==dev.fingerprint
 import ptrn_hip.partial_runner as frozen
 assert binding['code']=={p.name:sha256(p) for p in Path(frozen.__file__).parent.glob('*.py')}
 assert binding['weights']=={n:sha256(a.bundle/'weights'/n) for n in binding['weights']}
 model,_=load_model(a.bundle,a.resume);opt=torch.optim.Adam(model.optimizer_groups('head'));start,cursor=restore_checkpoint(a.resume,model,opt,binding)
 assert start<a.steps and a.gc_interval>0 and a.eval_batch==1
 def equal_state(x,y):
  if isinstance(x,torch.Tensor):return isinstance(y,torch.Tensor) and torch.equal(x.detach().cpu(),y.detach().cpu())
  if isinstance(x,dict):return x.keys()==y.keys() and all(equal_state(x[k],y[k]) for k in x)
  if isinstance(x,(tuple,list)):return len(x)==len(y) and all(equal_state(u,v) for u,v in zip(x,y))
  return x==y
 assert equal_state(model.state_dict(),cp['model']) and equal_state(opt.state_dict(),cp['optimizer'])
 assert torch.equal(torch.get_rng_state(),cp['torch_rng']) and equal_state(torch.cuda.get_rng_state_all(),cp['cuda_rng'])
 atomic_json(a.output/'resume-state-audit.json',{'step':start,'cursor':cursor,'model_optimizer_rng_exact':True,'checkpoint_sha256':sha256(a.resume)})
 runtime={'source_sha256':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},'gc_interval':a.gc_interval,'eval_batch':a.eval_batch,'binding_unchanged':True,'diagnostic_eval_count':a.diagnostic_eval_count}
 atomic_json(a.output/'runtime-protocol.json',runtime);atomic_json(a.output/'frozen-protocol.json',binding)
 from .input_bank import InputBank
 bank=InputBank(dev,a.cache,a.diagnostic_eval_count)
 ds=TrainCacheDataset(data,a.cache);jobs=[];errors=[];pending_checkpoints=[];requested=[];model_bn={k:v.cpu().clone() for k,v in model.state_dict().items() if 'moving_mean' in k or 'moving_variance' in k}
 # Finish generation of a 32-batch runway before the trainer starts consuming it.
 worker_rng=torch.Generator().manual_seed(data.seed)
 warm=DataLoader(ds,generator=worker_rng,sampler=range(cursor,min(a.steps*64,cursor+2048)),batch_size=64,num_workers=12,multiprocessing_context='spawn',worker_init_fn=worker_init)
 for i,_ in enumerate(warm):atomic_json(a.output/'pipeline-progress.json',{'phase':'training_prefill','completed':(i+1)*64,'target':min(a.steps*64-cursor,2048)})
 del warm
 def queue(checkpoint):
  requested.append(checkpoint);pending_checkpoints.append(checkpoint);start_queued()
 def start_queued():
  if not pending_checkpoints or any(j.status() not in ('complete','failed') for j in jobs):return
  checkpoint=pending_checkpoints.pop(0)
  try:
   if buffer is not None:
    try:buffer.future.result()
    except StopIteration:pass
    torch.cuda.synchronize()
   jobs.append(EvaluationJob(a.bundle,a.backgrounds,a.cache,checkpoint,a.output/('eval-'+checkpoint.stem),batch=a.eval_batch,bank=bank,count=a.diagnostic_eval_count))
  except Exception as e:errors.append({'checkpoint':str(checkpoint),'error':repr(e)})
 buffer=None
 if a.eval_current:queue(a.resume)
 workers=None;next_cursor=cursor
 try:
  for step in range(start+1,a.steps+1):
   active=any(j.status() not in ('complete','failed') for j in jobs)
   desired=8 if active else 12
   if buffer is None or (desired!=workers and (step-1)%250==0):
    if buffer is not None:buffer.close();del loader,buffer;gc.collect()
    loader=DataLoader(ds,generator=worker_rng,sampler=range(cursor,a.steps*64),batch_size=64,num_workers=desired,pin_memory=True,multiprocessing_context='spawn',worker_init_fn=worker_init,prefetch_factor=2)
    buffer=DoubleBuffer(loader);workers=desired
   t=time.perf_counter();wait_before=buffer.wait_seconds;batch=next(buffer)
   terms=train_step(model,opt,batch,effective_batch=64,step=step,gc_interval=a.gc_interval,loss_options=binding['loss']);cursor+=64
   row={'step':step,'cursor':cursor,'seconds_including_data':time.perf_counter()-t,'loader_wait_seconds':buffer.wait_seconds-wait_before,'workers':workers,**terms}
   with (a.output/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps(row),flush=True)
   if step%250==0 or step==a.steps:
    assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in model_bn.items()),'BN changed'
    checkpoint=a.output/f'step-{step}.pt';save_checkpoint(checkpoint,model,opt,step=step,cursor=cursor,binding=binding)
    if step in (1250,5000) or step==a.steps:queue(checkpoint)
   for job in jobs:
    if job.status()=='predicting':
     try:job.tick(2)
     except Exception as e:errors.append({'evaluation':str(job.output),'error':repr(e)})
   start_queued()
   atomic_json(a.output/'pipeline-progress.json',{'phase':'training','step':step,'workers':workers,'evaluations':{str(j.output):{'status':j.status(),'predicted':j.cursor} for j in jobs},'evaluation_errors':errors})
 finally:
  if buffer is not None:buffer.close();del buffer,loader;buffer=None;gc.collect()
 # Final training updates are durable; finishing outstanding evaluation is a separate phase.
 del model,opt;gc.collect();torch.cuda.empty_cache()
 while pending_checkpoints or any(j.status() in ('predicting','metrics') for j in jobs):
  start_queued()
  for job in jobs:
   if job.status()=='predicting':
    try:job.tick(32)
    except Exception as e:errors.append({'evaluation':str(job.output),'error':repr(e)})
  atomic_json(a.output/'pipeline-progress.json',{'phase':'final_evaluation','step':a.steps,'evaluations':{str(j.output):{'status':j.status(),'predicted':j.cursor} for j in jobs}})
  time.sleep(.05)
 atomic_json(a.output/'completion.json',{'training_complete':True,'steps':a.steps,'evaluations_complete':not errors and len(jobs)==len(requested) and all(j.status()=='complete' for j in jobs),'evaluation_errors':errors,'effectiveness_verified':False,'full_development_evaluated':a.diagnostic_eval_count is None and not errors and len(jobs)==len(requested) and all(j.status()=='complete' for j in jobs)})
if __name__=='__main__':main()
