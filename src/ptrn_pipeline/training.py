"""Lossless rolling training cache, event-safe double buffering, unchanged losses."""
import gc,time,os
from pathlib import Path
import numpy as np
import torch
from .cache import DiskCache,scene_key,digest

class TrainCacheDataset:
    def __init__(self,dataset,cache_root):
        self.data=dataset;self.cache=DiskCache(cache_root,'train-'+digest(dataset.fingerprint)[:16],budget=8*2**30)
        self.permutation={}
    def __getitem__(self,cursor):
        from .evaluation import memory_pressure
        while memory_pressure():time.sleep(.25)
        round_index,offset=divmod(cursor,len(self.data))
        if round_index not in self.permutation:
            self.permutation={round_index:np.random.default_rng(np.random.SeedSequence([self.data.seed,round_index,188])).permutation(len(self.data))}
        index=int(self.permutation[round_index][offset]);self.data.round_index=round_index
        key=scene_key(self.data,index,full=False);cached=self.cache.get(key)
        if cached is not None:return cached[0]
        item=self.data[index];self.cache.put(key,item,{'index':index,'round':round_index,'cursor':cursor})
        return item

class DoubleBuffer:
    """Background H2D staging overlaps the current forward/backward pass."""
    def __init__(self,loader,device='cuda',pinned_budget=2**30):
        from concurrent.futures import ThreadPoolExecutor
        self.iterator=iter(loader);self.device=device;self.stream=torch.cuda.Stream() if device=='cuda' else None
        self.retired=[];self.budget=pinned_budget;self.wait_seconds=0.
        self.h2d_incomplete_handoffs=0;self.h2d_copy_seconds=0.;self.h2d_copy_samples=0
        self._timing_events=[]
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='ptrn-h2d')
        self.future=self.executor.submit(self._stage)
    def _stage(self):
        host=next(self.iterator)
        size=sum(v.nelement()*v.element_size() for v in host.values())
        if size*2>self.budget:raise MemoryError('Pinned double-buffer budget exceeded')
        if self.device=='cuda':
            if not all(v.is_pinned() for v in host.values()):raise ValueError('Loader must pin before transfer')
            with torch.cuda.stream(self.stream):
                start=torch.cuda.Event(enable_timing=True);start.record(self.stream)
                gpu={k:v.to('cuda',non_blocking=True) for k,v in host.items()}
                event=torch.cuda.Event(enable_timing=True);event.record(self.stream)
            return host,gpu,event,start
        return host,host,None,None
    def _collect_completed_transfers(self):
        remaining=[]
        for start,end in self._timing_events:
            if end.query():
                self.h2d_copy_seconds+=start.elapsed_time(end)/1000
                self.h2d_copy_samples+=1
            else:remaining.append((start,end))
        self._timing_events=remaining
    def __iter__(self):return self
    def __next__(self):
        t=time.perf_counter()
        try:host,gpu,event,start=self.future.result()
        except StopIteration:self.close();raise
        self.wait_seconds+=time.perf_counter()-t
        if event:
            if not event.query():self.h2d_incomplete_handoffs+=1
            self._timing_events.append((start,event))
            current=torch.cuda.current_stream();current.wait_event(event)
            for v in gpu.values():v.record_stream(current)
            self.retired.append((host,event));self.retired=[(h,e) for h,e in self.retired if not e.query()]
            self._collect_completed_transfers()
        self.future=self.executor.submit(self._stage)
        return gpu
    def close(self):
        self.executor.shutdown(wait=True,cancel_futures=True)
        if self.device=='cuda':
            for _,event in self.retired:event.synchronize()
            self._collect_completed_transfers()


def train_step(model,optimizer,batch,*,effective_batch,step,gc_interval=1,loss_options=None,device='cuda'):
    """Execute the frozen arithmetic and checks; only schedule its GC callback.

    A function with the same code object and an isolated globals dictionary avoids
    changing the reference module or sharing GC policy with background threads.
    """
    from ptrn_hip.partial_runner import train_step as reference
    from types import FunctionType
    if gc_interval<1:raise ValueError('Positive GC interval required')
    if gc_interval==1:
        implementation=reference
    else:
        class CollectionPolicy:
            @staticmethod
            def collect():return gc.collect() if step%gc_interval==0 else 0
        context=dict(reference.__globals__,gc=CollectionPolicy)
        implementation=FunctionType(reference.__code__,context,reference.__name__,reference.__defaults__,reference.__closure__)
        implementation.__kwdefaults__=reference.__kwdefaults__
    return implementation(model,optimizer,[batch],device=device,effective_batch=effective_batch,loss_options=loss_options)
