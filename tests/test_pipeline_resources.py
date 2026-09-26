import pytest
from ptrn_pipeline.resources import gpu_lease

def test_gpu_owner_is_exclusive_and_releases_after_failure(tmp_path):
 with pytest.raises(RuntimeError):
  with gpu_lease(tmp_path):
   with pytest.raises(BlockingIOError):
    with gpu_lease(tmp_path):pass
   raise RuntimeError('worker failed')
 with gpu_lease(tmp_path):pass
