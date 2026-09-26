"""Three matched cold/hot trials, same checkpoint and ordered diagnostic subset."""
from pathlib import Path
import argparse,time,json,gc,concurrent.futures as cf,multiprocessing as mp
import numpy as np
import torch
from ptrn_hip.partial_data import PartialDataset
from ptrn_hip.partial_evaluation import evaluate as reference
from .evaluation import load_model,prepare_cache,init_worker,metric,bounded_map,inference
from .cache import digest,scene_key,DiskCache
from .input_bank import InputBank
from .graph import GraphPredictor
from .telemetry import atomic_json

class First:
 def __init__(self,data,count):self.data=data;self.count=count
 def __len__(self):return self.count
 def render(self,i):return self.data.render(i)

def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--backgrounds',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=256);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(4);data=PartialDataset(a.bundle,a.backgrounds,split='dev',verify=False);model,_=load_model(a.bundle,a.checkpoint);graph=GraphPredictor(model);results=[]
 for trial in range(3):
  # Cache-only deletion to recreate a cold scenario, never inputs or weights.
  for full,budget,name in [(True,10*2**30,'dev-full-'),(False,2*2**30,'dev-input-')]:
   cache=DiskCache(a.cache,name+digest(data.fingerprint)[:16],budget=budget)
   for i in range(a.count):cache.path(scene_key(data,i,full=full)).unlink(missing_ok=True)
  t=time.perf_counter();old=reference(model,First(data,a.count),device='cuda');old_seconds=time.perf_counter()-t
  t=time.perf_counter();prepare_cache(a.bundle,a.backgrounds,a.cache,a.output/f'prepare-{trial}',workers=4,limit=a.count);cold=time.perf_counter()-t
  bank=InputBank(data,a.cache,a.count);times=[];rows=None
  with cf.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(str(a.bundle),str(a.backgrounds),str(a.cache))) as pool:
   for repeat in range(2):
    t=time.perf_counter();pred=[]
    with torch.no_grad():
     for i in range(a.count):
      v=graph([bank[i]]);pred.append((i,v['quad'][0],v['ratio'][0],v['baseline_ratio'][0]))
    rows=list(bounded_map(pool,metric,pred,limit=16));times.append(time.perf_counter()-t)
  errors={k:max(abs(x[k]-y[k]) for x,y in zip(old['rows'],rows) if k in x and k in y) for k in ('corner_error_diagonal','absolute_ratio_error','additional_crop_fraction','extra_background_fraction','visible_grid_p90','visible_grid_max')}
  assert all(v==0 for v in errors.values());assert all(x['valid_geometry']==y['valid_geometry'] for x,y in zip(old['rows'],rows))
  results.append({'trial':trial,'count':a.count,'reference_seconds':old_seconds,'cold_prepare_seconds':cold,
   'cold_pipeline_seconds':cold+times[0],'hot_pipeline_seconds':times[1],'cold_speedup':old_seconds/(cold+times[0]),'hot_speedup':old_seconds/times[1],'exact_metrics':True})
  atomic_json(a.output/'paired-evaluation.json',{'scope':'matched subset; full 5736-row equivalence measured separately','results':results});gc.collect()
 print(json.dumps(results))
if __name__=='__main__':main()
