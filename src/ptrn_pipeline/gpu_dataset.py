"""GPU scene pre-generation with bounded CPU planning/writing and explicit version."""
from pathlib import Path
import concurrent.futures as cf
import numpy as np
from .cache import DiskCache,digest,scene_key
from .gpu_render import plan,render,RENDERER
from ptrn_hip.partial_data import recipe
from ptrn_hip.data import read_rgb

class GPUPlanner:
 def __init__(self,data):self.data=data
 def __call__(self,index):
  if not 0<=index<len(self.data):raise IndexError(index)
  d=self.data;position,view=divmod(index,d.views);source_id=d.ids[position]
  options=recipe(position,view,round_index=d.round_index,seed=d.seed,domain=d.split)
  bg=None
  if options.get('family')=='coco_context':
   rng=np.random.default_rng(np.random.SeedSequence([d.seed,d.round_index,source_id,view,891]));row=d.bg[int(rng.integers(len(d.bg)))];bg=read_rgb(d.safe_path(d.backgrounds,row['path']))
  source=read_rgb(d.safe_path(d.bundle,d.records[source_id]['source_path']))
  for attempt in range(20):
   try:
    p=plan(source,source_id,view+d.round_index*d.views,seed=d.seed+attempt*1000003,domain=d.split,background=bg,**options)
    p.info.update(round=d.round_index,dataset_index=index,geometry_retries=attempt)
    return p
   except RuntimeError:
    if attempt==19:raise

def pregenerate(data,indices,cache_root,*,batch=4,device='cuda'):
 """Own the GPU externally. Writes run off-thread behind event completion."""
 cache=DiskCache(cache_root,'train-gpu-'+digest(data.fingerprint)[:16],budget=8*2**30)
 planner=GPUPlanner(data);rows=[];pending=[]
 def finish(index,item):
  view=item.resolve();arrays=view.input(data.size);key=scene_key(data,index,renderer=RENDERER,full=False)
  if not cache.put(key,arrays,view.info):raise RuntimeError('GPU cache write refused by quota')
  return {'index':index,'key':key,'occupancy':view.info['occupancy'],'missing_fraction':view.info['missing_fraction'],
          'kind':view.info['kind'],'family':view.info['family'],'integrity':view.info['integrity']}
 ids=list(indices)
 with cf.ThreadPoolExecutor(max_workers=2,thread_name_prefix='render-writer') as writers, cf.ThreadPoolExecutor(max_workers=min(8,batch*2),thread_name_prefix='render-planner') as planning:
  futures={j:planning.submit(planner,ids[j]) for j in range(min(len(ids),batch*2))}
  for start in range(0,len(ids),batch):
   chunk=ids[start:start+batch];plans=[futures.pop(j).result() for j in range(start,start+len(chunk))]
   for j in range(start+2*batch,min(len(ids),start+3*batch)):futures[j]=planning.submit(planner,ids[j])
   try:items=render(plans,device)
   except MemoryError:
    # Same scenes at a smaller render batch, never silently replace a difficult image.
    items=[render([p],device)[0] for p in plans]
   for index,item in zip(chunk,items):pending.append(writers.submit(finish,index,item))
   while len(pending)>batch*2:rows.append(pending.pop(0).result())
  rows.extend(f.result() for f in pending)
 return cache,rows
