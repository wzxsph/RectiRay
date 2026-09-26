"""Discard-only GPU/runtime checks; results and weights are never promoted."""
import argparse,gc,json,time,os,copy
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .evaluation import load_model
from .training import DoubleBuffer,train_step,TrainCacheDataset
from .telemetry import atomic_json

def benchmark(a):
 from ptrn_hip.partial_data import PartialDataset,sha256
 from ptrn_hip.partial_runner import ShuffledRounds,train_step as reference_step
 data=PartialDataset(a.bundle,a.backgrounds,verify=False);cp=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
 model,_=load_model(a.bundle,a.checkpoint);opt=torch.optim.Adam(model.optimizer_groups('head'))
 sequence=ShuffledRounds(data);cpu=[sequence[cp['cursor']+i] for i in range(64)]
 loader=lambda:DataLoader(cpu,batch_size=64,pin_memory=True)
 def reset():
  model.load_state_dict(cp['model']);opt.load_state_dict(copy.deepcopy(cp['optimizer']));torch.manual_seed(20260925);gc.collect();torch.cuda.empty_cache()
 host=next(iter(loader()))
 transport=DoubleBuffer(loader());transported=next(transport)
 assert all(torch.equal(host[k],v.cpu()) for k,v in transported.items()),'Transferred input changed'
 transport.close();del transported,transport;gc.collect();torch.cuda.empty_cache()
 if a.gc_only:
  receipt=json.loads((a.output.parent/'optimizer-cpu-equivalence.json').read_text())
  assert receipt['exact_model_and_optimizer'] and receipt['checkpoint_sha256']==sha256(a.checkpoint)
  atomic_json(a.output/'optimizer-verification.json',{'cpu_exact_receipt':receipt,'gpu_bitwise_equality':'Not asserted: original CUDA backward A/A is nondeterministic; frozen arithmetic reused unchanged'})
 else:
  reset();host=next(iter(loader()));reference_step(model,opt,[host],device='cuda',effective_batch=64,loss_options=cp['binding']['loss'])
  expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  reset();buffer=DoubleBuffer(loader());batch=next(buffer);train_step(model,opt,batch,effective_batch=64,step=1,loss_options=cp['binding']['loss']);buffer.close()
  actual={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  def difference(a,b):return {k:{'max_abs':float((a[k]-b[k]).abs().max()),'changed':int((a[k]!=b[k]).sum())} for k in a if not torch.equal(a[k],b[k])}
  delta=difference(expected,actual)
  if delta:
   reset();reference_step(model,opt,[host],device='cuda',effective_batch=64,loss_options=cp['binding']['loss'])
   repeat={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   atomic_json(a.output/'optimizer-diagnostic.json',{'candidate_differences':delta,'reference_repeat_differences':difference(expected,repeat)})
   raise AssertionError('Optimizer transition changed; A/A diagnostics preserved')
  atomic_json(a.output/'optimizer-equivalence.json',{'exact_parameter_equality':True,'diagnostic_deterministic_algorithms':False,'checkpoint_sha256':sha256(a.checkpoint)})
 reports=[]
 for interval in (1,4,8):
  reset();times=[];allocated=[];failure=None;torch.cuda.reset_peak_memory_stats()
  # One pinned batch reused; isolate garbage collector effects from synthesis variance.
  batch={k:v.to('cuda') for k,v in host.items()}
  try:
   for step in range(1,a.steps+1):
    torch.cuda.synchronize();start=time.perf_counter();train_step(model,opt,batch,effective_batch=64,step=step,gc_interval=interval,loss_options=cp['binding']['loss']);torch.cuda.synchronize()
    times.append(time.perf_counter()-start);allocated.append(torch.cuda.memory_allocated()/2**20)
    if torch.cuda.max_memory_reserved()>22*2**30:raise MemoryError('22GiB peak exceeded')
    if step%25==0:atomic_json(a.output/'progress.json',{'phase':'gc_benchmark','interval':interval,'step':step,'target':a.steps})
  except (torch.OutOfMemoryError,MemoryError) as e:failure=str(e)
  growth=max(allocated[-100:],default=0)-min(allocated[-100:],default=0)
  passed=failure is None and len(times)==a.steps and growth<=128
  reports.append({'interval':interval,'steps':len(times),'mean_seconds':float(np.mean(times[10:] or times)) if times else None,
       'last100_allocation_spread_mib':growth,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'passed':passed,'failure':failure})
  atomic_json(a.output/'gc-benchmark.json',{'steps_requested':a.steps,'results':reports})
  gc.collect();torch.cuda.empty_cache()
 good=[r for r in reports if r['passed']];selected=min(good,key=lambda r:r['mean_seconds'])['interval'] if good else 1
 atomic_json(a.output/'selected-gc.json',{'interval':selected,'qualified':bool(good),'checkpoint_sha256':sha256(a.checkpoint)})

def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--backgrounds',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=500);p.add_argument('--gc-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
 from .resources import gpu_lease
 with gpu_lease(a.output.parent.parent):benchmark(a)
if __name__=='__main__':main()
