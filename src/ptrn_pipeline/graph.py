"""Replay the original batch-one CUDA operations without Python dispatch per layer."""
import gc
import numpy as np
import torch

class GraphPredictor:
 def __init__(self,model,batch=1):
  self.model=model;self.batch=batch
  self.raw=torch.zeros((batch,model.size,model.size,3),device='cuda',dtype=torch.uint8)
  self.mean=torch.tensor([.485,.456,.406],device='cuda');self.std=torch.tensor([.229,.224,.225],device='cuda')
  def forward():
   from torch.nn import functional as F
   x=self.raw.float()/255
   features=model.spatial.extract_features((x-self.mean)/self.std)
   logits=model.spatial._spatial_logits(features).float()
   quad=model.spatial.direct(F.adaptive_avg_pool2d(logits,(56,56)).flatten(1)).reshape(-1,4,2)*(model.size-1)
   return quad.float(),features[-1].float().mean((2,3))
  self.stream=torch.cuda.Stream();self.stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(self.stream),torch.no_grad():
   for _ in range(3):self.output=forward()
  torch.cuda.current_stream().wait_stream(self.stream);torch.cuda.synchronize();gc.collect()
  self.graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(self.graph,stream=self.stream),torch.no_grad():self.output=forward()
 def __call__(self,items):
  if len(items)!=self.batch:raise ValueError('Captured batch shape differs')
  self.raw.copy_(torch.as_tensor(np.stack([v['pixels'] for v in items]),device='cuda'))
  self.graph.replay()
  quad,pooled=self.output
  # Keep the original finite-input checks and ratio operations outside capture.
  edges=torch.linalg.vector_norm(torch.roll(quad.detach(),-1,dims=1)-quad.detach(),dim=-1).clamp_min(1e-4)
  baseline=edges[:,[0,2]].amax(1).log()-edges[:,[1,3]].amax(1).log()
  log_ratio=baseline+self.model.ratio_head(quad,pooled)
  output={'quad':quad,'log_ratio':log_ratio,'log_ratio_baseline':baseline,'ratio':log_ratio.exp(),'baseline_ratio':baseline.exp()}
  return {k:v.detach().cpu().numpy().copy() for k,v in output.items()}
