"""Versioned real-source rendering audit and three cold/hot paired supply trials."""
import argparse,time,json,gc,copy
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ptrn_hip.partial_data import PartialDataset
from .cache import DiskCache,digest,scene_key
from .gpu_render import RENDERER
from .gpu_dataset import pregenerate
from .telemetry import atomic_json

class DirectCacheData:
 def __init__(self,data,root):self.data=data;self.cache=DiskCache(root,'train-'+digest(data.fingerprint)[:16],budget=8*2**30)
 def __getitem__(self,index):
  key=scene_key(self.data,index,full=False);item=self.cache.get(key)
  if item is not None:return item[0]
  value=self.data[index]
  if not self.cache.put(key,value,{'index':index}):raise RuntimeError('CPU cache quota refusal')
  return value


def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--backgrounds',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=128);p.add_argument('--checkpoint',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
 from .resources import gpu_lease
 lease=gpu_lease(a.cache.parent);lease.__enter__()
 from .evaluation import load_model
 from .training import DoubleBuffer,train_step
 if a.checkpoint is None:a.checkpoint=Path(json.loads((a.output.parent/'benchmark-config.json').read_text())['checkpoint'])
 model,checkpoint=load_model(a.bundle,a.checkpoint);optimizer=torch.optim.Adam(model.optimizer_groups('head'))
 def train(items):
  model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']));torch.manual_seed(20260925);gc.collect();torch.cuda.empty_cache()
  torch.cuda.synchronize();start=time.perf_counter();buffer=DoubleBuffer(items if isinstance(items,DataLoader) else DataLoader(items,batch_size=64,pin_memory=True))
  for step,batch in enumerate(buffer,1):train_step(model,optimizer,batch,effective_batch=64,step=step,loss_options=checkpoint['binding']['loss'])
  buffer.close();torch.cuda.synchronize();seconds=time.perf_counter()-start;del buffer,batch;gc.collect();torch.cuda.empty_cache();return seconds
 from ptrn_hip.partial_runner import worker_init
 data=PartialDataset(a.bundle,a.backgrounds,verify=False);ids=list(range(a.count));trials=[]
 def loader(dataset):return DataLoader(dataset,sampler=ids,batch_size=64,pin_memory=True,num_workers=12,multiprocessing_context='spawn',worker_init_fn=worker_init,prefetch_factor=2)
 cpu=DiskCache(a.cache,'train-'+digest(data.fingerprint)[:16],budget=8*2**30)
 for trial in range(3):
  for i in ids:cpu.path(scene_key(data,i,full=False)).unlink(missing_ok=True)
  # Goldens are used for exactness only, never timed as the parallel online baseline.
  online=[data[i] for i in ids]
  online_total=train(loader(data))
  t=time.perf_counter()
  for _ in loader(DirectCacheData(data,a.cache)):pass
  cpu_prepare=time.perf_counter()-t
  t=time.perf_counter();cached=[cpu.get(scene_key(data,i,full=False))[0] for i in ids];read_seconds=time.perf_counter()-t
  assert all(all(np.array_equal(x[k],y[k]) for k in x) for x,y in zip(online,cached))
  cached_train=train(cached)
  t=time.perf_counter();gpu,rows=pregenerate(data,ids,a.cache,batch=4);torch.cuda.synchronize();gpu_seconds=time.perf_counter()-t
  t=time.perf_counter();gpu_items=[gpu.get(scene_key(data,i,renderer=RENDERER,full=False))[0] for i in ids];gpu_read=time.perf_counter()-t
  gpu_train=train(gpu_items)
  photo=[r for r in rows if r['kind']=='photo'];high=sum(r['occupancy']>.5 for r in photo)/len(photo)
  assert all(r['occupancy']>=.3 for r in photo);assert high>=.8
  assert {'viewer_screen','coco_context','near_color','mixed_color'}<={r['family'] for r in rows}
  assert {'complete','light','moderate'}<={r['integrity'] for r in rows}
  atomic_json(a.output/f'gpu-audit-{trial}.json',{'renderer':RENDERER,'count':len(rows),'photo_high_occupancy_fraction':high,'rows':rows,'frozen_development_unchanged':True})
  trials.append({'trial':trial,'count':len(ids),'cpu_parallel_online_total_seconds':online_total,'cpu_cache_cold_seconds':cpu_prepare+read_seconds,'cpu_cache_hot_seconds':read_seconds,'gpu_pregeneration_seconds':gpu_seconds,'gpu_cache_read_seconds':gpu_read,'gpu_cold_seconds':gpu_seconds+gpu_read,'cpu_cache_exact':True,'cpu_online_training_total':online_total,'cpu_cache_cold_training_total':cpu_prepare+read_seconds+cached_train,'cpu_cache_hot_training_total':read_seconds+cached_train,'gpu_cache_cold_training_total':gpu_seconds+gpu_read+gpu_train,'gpu_cache_hot_training_total':gpu_read+gpu_train,'training_batch':64,'training_steps':len(ids)//64})
  atomic_json(a.output/'render-benchmark.json',{'renderer':RENDERER,'scope':'Paired supply plus actual model updates; updates discarded, GPU renderer is a new data version','default_enabled':False,'trials':trials})
  del online,cached,gpu_items;gc.collect();torch.cuda.empty_cache()
 gain=1-sum(t['gpu_cache_cold_training_total'] for t in trials)/sum(t['cpu_cache_cold_training_total'] for t in trials)
 atomic_json(a.output/'gpu-default-decision.json',{'cold_total_gain_vs_cpu_cache':gain,'speed_criterion_passed':gain>=.1,'default_enabled':False,'reason':'Renderer version change reserved for subsequent explicitly configured runs; reference training and development remain frozen'})
 print(json.dumps(trials))
if __name__=='__main__':main()
