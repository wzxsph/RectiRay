"""Pre-generate a deterministic training runway while the GPU works elsewhere."""
import argparse,time,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ptrn_hip.partial_data import PartialDataset
from ptrn_hip.partial_runner import worker_init
from .training import TrainCacheDataset
from .telemetry import atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--backgrounds',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cursor',type=int,required=True);p.add_argument('--count',type=int,required=True);p.add_argument('--workers',type=int,default=12);a=p.parse_args()
 if a.cursor<0 or a.count<1 or not 1<=a.workers<=12:raise ValueError('Invalid bounded prefill')
 torch.set_num_threads(1);a.output.mkdir(parents=True,exist_ok=True);data=PartialDataset(a.bundle,a.backgrounds,verify=False);ds=TrainCacheDataset(data,a.cache)
 loader=DataLoader(ds,sampler=range(a.cursor,a.cursor+a.count),batch_size=64,num_workers=a.workers,multiprocessing_context='spawn',worker_init_fn=worker_init,prefetch_factor=2)
 started=time.time();n=0
 for batch in loader:
  n+=len(batch['pixels']);atomic_json(a.output/'progress.json',{'phase':'ahead_training_cache','completed':n,'count':a.count,'cursor':a.cursor+n,'seconds':time.time()-started})
 atomic_json(a.output/'complete.json',{'prepared':n,'start_cursor':a.cursor,'seconds':time.time()-started,'best_effort_disk_cache_with_online_fallback':True})
if __name__=='__main__':main()
