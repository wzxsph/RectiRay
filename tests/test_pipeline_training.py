import numpy as np
import torch
from torch.utils.data import DataLoader
from ptrn_pipeline.training import TrainCacheDataset,DoubleBuffer
from ptrn_hip.partial_runner import ShuffledRounds

class Toy:
 def __init__(self):self.seed=3;self.round_index=0;self.fingerprint={'seed':3,'size':224}
 def __len__(self):return 17
 def __getitem__(self,i):return {'pixels':np.array([i,self.round_index],np.float32)}

def test_cache_order_matches_resume_with_prefetch(tmp_path,monkeypatch):
 monkeypatch.setattr('ptrn_pipeline.training.scene_key',lambda d,i,full:__import__('hashlib').sha256(f'{i}/{d.round_index}'.encode()).hexdigest())
 a=TrainCacheDataset(Toy(),tmp_path);a.cache.reserve=0;b=ShuffledRounds(Toy())
 expected=[b[i] for i in range(11,53)];actual=[a[i] for i in range(11,53)]
 assert all(np.array_equal(x['pixels'],y['pixels']) for x,y in zip(expected,actual))
 resumed=[a[i] for i in range(29,53)];assert all(np.array_equal(x['pixels'],y['pixels']) for x,y in zip(expected[18:],resumed))

def test_double_buffer_cpu_order():
 loader=DataLoader([{'pixels':torch.tensor([i])} for i in range(12)],batch_size=3)
 assert torch.cat([v['pixels'] for v in DoubleBuffer(loader,'cpu')]).flatten().tolist()==list(range(12))

@__import__('pytest').mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_double_buffer_survives_allocator_reuse():
 loader=DataLoader([{'pixels':torch.full((128,128),i,dtype=torch.uint8)} for i in range(20)],batch_size=2,pin_memory=True)
 buffer=DoubleBuffer(loader);seen=[]
 for b in buffer:
  scratch=torch.ones((1024,1024),device='cuda');scratch.mul_(2)
  seen.extend(b['pixels'][:,0,0].cpu().tolist())
 assert seen==list(range(20))
 assert buffer.h2d_copy_samples==10
 assert buffer.h2d_copy_seconds>=0
 assert 0<=buffer.h2d_incomplete_handoffs<=10
