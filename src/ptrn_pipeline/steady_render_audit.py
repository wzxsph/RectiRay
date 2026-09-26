"""Three paired trials with persistent CPU workers, avoiding startup-biased claims."""
from pathlib import Path
import argparse,copy,gc,json,time,multiprocessing as mp
import numpy as np
import torch
from torch.utils.data import DataLoader
from ptrn_hip.partial_data import PartialDataset,sha256
from ptrn_hip.partial_runner import worker_init
from .cache import DiskCache,digest,scene_key
from .training import DoubleBuffer,train_step
from .evaluation import load_model
from .gpu_dataset import pregenerate
from .gpu_render import RENDERER
from .resources import gpu_lease
from .telemetry import atomic_json

class ModeDataset:
 def __init__(self,data,cache,mode):self.data=data;self.cache=cache;self.mode=mode
 def __getitem__(self,i):
  item=self.data[i]
  if self.mode.value==1:
   if not self.cache.put(scene_key(self.data,i,full=False),item,{'index':i}):raise RuntimeError('CPU cache quota refusal')
  return item

def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--backgrounds',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=1024);a=p.parse_args()
 if a.count%64 or a.count<128:raise ValueError('Use complete 64-image batches')
 a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
 with gpu_lease(a.cache.parent):
  data=PartialDataset(a.bundle,a.backgrounds,verify=False);model,cp=load_model(a.bundle,a.checkpoint);opt=torch.optim.Adam(model.optimizer_groups('head'))
  round_index,offset=divmod(cp['cursor'],len(data));data.round_index=round_index
  ids=np.random.default_rng(np.random.SeedSequence([data.seed,round_index,188])).permutation(len(data))[offset:offset+a.count].tolist();assert len(ids)==a.count
  cache=DiskCache(a.cache,'train-'+digest(data.fingerprint)[:16],budget=8*2**30)
  mode=mp.get_context('spawn').Value('i',0);dataset=ModeDataset(data,cache,mode)
  loader=DataLoader(dataset,sampler=ids,batch_size=64,num_workers=12,pin_memory=True,persistent_workers=True,multiprocessing_context='spawn',worker_init_fn=worker_init,prefetch_factor=2)
  def train(source,phase):
   model.load_state_dict(cp['model']);opt.load_state_dict(copy.deepcopy(cp['optimizer']));torch.manual_seed(20260925);gc.collect();torch.cuda.empty_cache()
   start=time.perf_counter();buffer=DoubleBuffer(source);n=0
   for step,batch in enumerate(buffer,1):
    train_step(model,opt,batch,effective_batch=64,step=step,gc_interval=4,loss_options=cp['binding']['loss']);n+=64
    atomic_json(a.output/'progress.json',{'phase':phase,'completed':n,'count':a.count})
   buffer.close();torch.cuda.synchronize();seconds=time.perf_counter()-start;del buffer,batch;gc.collect();torch.cuda.empty_cache();return seconds
  reports=[]
  for trial in range(3):
   mode.value=0;online=train(loader,f'online-{trial}')
   for i in ids:cache.path(scene_key(data,i,full=False)).unlink(missing_ok=True)
   mode.value=1;start=time.perf_counter()
   for batch in loader:pass
   prepare_seconds=time.perf_counter()-start;del batch
   start=time.perf_counter();cpu_items=[cache.get(scene_key(data,i,full=False))[0] for i in ids];cpu_read=time.perf_counter()-start
   cpu_train=train(DataLoader(cpu_items,batch_size=64,pin_memory=True),f'cpu-cached-{trial}')
   start=time.perf_counter();gpu,rows=pregenerate(data,ids,a.cache,batch=4);gpu_generate=time.perf_counter()-start
   start=time.perf_counter();gpu_items=[gpu.get(scene_key(data,i,renderer=RENDERER,full=False))[0] for i in ids];gpu_read=time.perf_counter()-start
   gpu_train=train(DataLoader(gpu_items,batch_size=64,pin_memory=True),f'gpu-cached-{trial}')
   photos=[r for r in rows if r['kind']=='photo'];high=sum(r['occupancy']>.5 for r in photos)/len(photos)
   assert high>=.8 and min(r['occupancy'] for r in photos)>=.3
   assert {'complete','light','moderate'}<={r['integrity'] for r in rows}
   assert {'viewer_screen','coco_context','near_color','mixed_color'}<={r['family'] for r in rows}
   unique_sources=len({data.ids[i//data.views] for i in ids})
   atomic_json(a.output/f'data-audit-{trial}.json',{'renderer':RENDERER,'count':len(rows),'unique_sources':unique_sources,'high_occupancy_fraction':high,'rows':rows})
   reports.append({'trial':trial,'count':a.count,'cpu_pool_startup_included':trial==0,'cpu_online_total':online,'cpu_prepare':prepare_seconds,'cpu_cold_total':prepare_seconds+cpu_read+cpu_train,'cpu_hot_total':cpu_read+cpu_train,'gpu_prepare':gpu_generate,'gpu_cold_total':gpu_generate+gpu_read+gpu_train,'gpu_hot_total':gpu_read+gpu_train,'unique_sources':unique_sources})
   atomic_json(a.output/'report.json',{'trials':reports,'checkpoint_sha256':sha256(a.checkpoint),'renderer':RENDERER});del cpu_items,gpu_items;gc.collect();torch.cuda.empty_cache()
  # Compare the warm-worker trials with both parallel online and CPU caching.
  baseline=min(sum(r['cpu_online_total'] for r in reports[1:]),sum(r['cpu_cold_total'] for r in reports[1:]))
  gain=1-sum(r['gpu_cold_total'] for r in reports[1:])/baseline
  atomic_json(a.output/'decision.json',{'total_gain_vs_best_warm_cpu':gain,'speed_criterion_passed':gain>=.1,'data_audit_passed':True,'automatic_default_enabled':False,'scope':'Opt-in renderer; current reference training and fixed development remain unchanged'})
if __name__=='__main__':main()
