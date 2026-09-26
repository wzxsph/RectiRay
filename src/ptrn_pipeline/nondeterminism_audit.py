"""Repeated original GPU trajectories; separate nondeterminism from checkpoint reload."""
from pathlib import Path
import argparse,copy,json,gc,time
import torch
from torch.utils.data import DataLoader
from ptrn_hip.partial_data import PartialDataset
from ptrn_hip.partial_runner import train_step as reference
from .training import TrainCacheDataset,DoubleBuffer
from .evaluation import load_model
from .resources import gpu_lease
from .telemetry import atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--backgrounds',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
 with gpu_lease(a.cache.parent):
  model,cp=load_model(a.bundle,a.checkpoint);opt=torch.optim.Adam(model.optimizer_groups('head'));data=TrainCacheDataset(PartialDataset(a.bundle,a.backgrounds,verify=False),a.cache)
  items=[data[i] for i in range(cp['cursor'],cp['cursor']+20*64)];losses=[];states=[]
  for repeat in range(2):
   model.load_state_dict(cp['model']);opt.load_state_dict(copy.deepcopy(cp['optimizer']));torch.set_rng_state(cp['torch_rng']);torch.cuda.set_rng_state_all(cp['cuda_rng']);gc.collect();torch.cuda.empty_cache();values=[]
   buffer=DoubleBuffer(DataLoader(items,batch_size=64,pin_memory=True))
   for i,batch in enumerate(buffer):
    v=reference(model,opt,[batch],device='cuda',effective_batch=64,loss_options=cp['binding']['loss']);values.append(v['total']);atomic_json(a.output/'progress.json',{'phase':'original_AA','repeat':repeat,'completed':i+1,'count':20})
   buffer.close();losses.append(values);states.append({k:v.cpu().clone() for k,v in model.state_dict().items()})
  atomic_json(a.output/'report.json',{'original_unmodified_step_function':True,'steps':20,'max_parameter_difference':max(float((states[0][k]-states[1][k]).abs().max()) for k in states[0]),'losses':losses,'bitwise_equal':all(torch.equal(states[0][k],states[1][k]) for k in states[0]),'scope':'A/A drift does not prove every between-run discrepancy has the same cause'})
if __name__=='__main__':main()
