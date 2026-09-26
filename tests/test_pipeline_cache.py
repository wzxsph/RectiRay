import numpy as np
import pytest
import threading
import time
from ptrn_pipeline.cache import DiskCache,digest,encode,decode

def test_exact_roundtrip_preserves_fractional_masks(tmp_path):
    a={'pixels':np.arange(48,dtype=np.uint8).reshape(4,4,3),'mask':np.linspace(0,1,16,dtype=np.float32).reshape(4,4),'quad':np.array([[-2.2,0],[4,0],[4,4],[0,4]],np.float64)}
    c=DiskCache(tmp_path,'dev',budget=100000,reserve=0);k=digest({'round':0})
    assert c.put(k,a,{'missing':True})
    b,m=c.get(k)
    for name in a:assert b[name].dtype==a[name].dtype and np.array_equal(a[name],b[name])
    assert m=={'missing':True}

def test_corrupt_rebuilt_and_namespaced(tmp_path):
    c=DiskCache(tmp_path,'dev',budget=100000,reserve=0);k=digest(1)
    c.put(k,{'a':np.ones(3)},{});c.path(k).write_bytes(b'corrupt')
    assert c.get(k) is None
    assert c.put(k,{'a':np.zeros(3)},{});assert np.array_equal(c.get(k)[0]['a'],np.zeros(3))
    assert DiskCache(tmp_path,'train',reserve=0).get(k) is None

def test_disk_refusal_cannot_delete_original(tmp_path):
    original=tmp_path/'weight.pt';original.write_bytes(b'protected')
    c=DiskCache(tmp_path/'cache','dev',budget=2,reserve=0)
    assert not c.put(digest(1),{'a':np.zeros(50)},{});assert original.read_bytes()==b'protected'
    with pytest.raises(ValueError):c.get('../bad')

def test_checksum_failure():
    encoded=encode({'a':np.zeros(2)},{});broken=encoded[:-1]+bytes([encoded[-1]^1])
    with pytest.raises(ValueError):decode(broken)

def test_disk_reserve_evicts_only_rebuildable_cache(tmp_path,monkeypatch):
 import collections
 import ptrn_pipeline.cache as module
 usage=collections.namedtuple('usage','total used free')
 c=DiskCache(tmp_path/'cache','train-test',budget=10000,reserve=0)
 assert c.put(digest(1),{'x':np.zeros(10)}, {})
 original=tmp_path/'original.png';original.write_bytes(b'keep')
 monkeypatch.setattr(module.shutil,'disk_usage',lambda p:usage(100,90,10));c.reserve=100
 assert not c.put(digest(2),{'x':np.ones(10)}, {})
 assert original.read_bytes()==b'keep'

def test_gpu_version_and_round_bound_into_cache_key(tmp_path):
 from ptrn_pipeline.cache import scene_key
 class Data:fingerprint={'source':'fixture','size':224};round_index=0
 a=Data();one=scene_key(a,0,renderer='torch-photo-v1');a.round_index=1
 assert one!=scene_key(a,0,renderer='torch-photo-v1')
 assert one!=scene_key(Data(),0)
 with pytest.raises(ValueError):DiskCache(tmp_path,'../escape')

def test_training_budget_shared_across_render_versions(tmp_path,monkeypatch):
 import ptrn_pipeline.cache as module
 monkeypatch.setattr(module,'GIB',512)
 a=DiskCache(tmp_path,'train-a',budget=10000,reserve=0);b=DiskCache(tmp_path,'train-gpu-b',budget=10000,reserve=0)
 pixels=np.random.default_rng(1).integers(0,256,1400,dtype=np.uint8)
 assert a.put(digest(1),{'pixels':pixels},{})
 assert b.put(digest(2),{'pixels':pixels},{})
 assert b.put(digest(3),{'pixels':pixels},{})
 assert a.get(digest(1)) is None and b.get(digest(3)) is not None
 assert sum(p.stat().st_size for p in tmp_path.glob('*/*/*.npz'))<=8*512

def test_corrupt_input_bank_rebuilt_before_gpu_work(tmp_path,monkeypatch):
 import ptrn_pipeline.input_bank as module
 class Data:
  fingerprint={'test':1}
  def __len__(self):return 1
 data=Data();key=digest('bank');monkeypatch.setattr(module,'scene_key',lambda *a,**k:key)
 monkeypatch.setattr(module,'cached_scene',lambda *a:(key,{'input_pixels':np.ones((4,4,3),np.uint8),'input_quad':np.zeros((4,2),np.float32)}, {},True))
 bank=module.InputBank(data,tmp_path)
 assert bank.rebuilt==1 and np.array_equal(bank[0]['pixels'],np.ones((4,4,3),np.uint8))

def test_lazy_input_bank_preserves_order_without_blocking_consumer(tmp_path,monkeypatch):
 import ptrn_pipeline.input_bank as module
 gate=threading.Event()
 class Data:
  fingerprint={'test':'lazy'}
  def __len__(self):return 4
 data=Data()
 monkeypatch.setattr(module,'scene_key',lambda d,i,full:digest(i))
 def render(d,i,cache):
  if i==0:assert gate.wait(2)
  return digest(i),{'input_pixels':np.full((2,2,3),i,np.uint8)}, {},False
 monkeypatch.setattr(module,'cached_scene',render)
 bank=module.InputBank(data,tmp_path,eager=False,prefetch=2)
 try:
  assert bank.ready_batch(0,1) is None
  gate.set()
  deadline=time.monotonic()+2
  while bank.prepared<2 and time.monotonic()<deadline:time.sleep(.01)
  assert bank.prepared==2
  assert [v['pixels'][0,0,0] for v in bank.ready_batch(0,2)]==[0,1]
  assert bank.ready_batch(2,1) is None
  bank.acknowledge(2)
  deadline=time.monotonic()+2
  while bank.prepared<4 and time.monotonic()<deadline:time.sleep(.01)
  assert [v['pixels'][0,0,0] for v in bank.ready_batch(2,2)]==[2,3]
  assert bank.rebuilt==4
 finally:
  gate.set();bank.close()

def test_lazy_input_bank_propagates_render_failure(tmp_path,monkeypatch):
 import ptrn_pipeline.input_bank as module
 class Data:
  fingerprint={'test':'error'}
  def __len__(self):return 1
 monkeypatch.setattr(module,'scene_key',lambda *a,**k:digest(1))
 def fail(*args):raise ValueError('broken scene')
 monkeypatch.setattr(module,'cached_scene',fail)
 bank=module.InputBank(Data(),tmp_path,eager=False)
 try:
  with pytest.raises(RuntimeError,match='preparation failed') as error:bank[0]
  assert isinstance(error.value.__cause__,ValueError)
  with pytest.raises(RuntimeError,match='preparation failed'):bank.ready_batch(0,1)
 finally:bank.close()
