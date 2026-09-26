"""One GPU owner per pipeline; training and validation share that owner's scheduler."""
from contextlib import contextmanager
from pathlib import Path
import fcntl,os,json,time

@contextmanager
def gpu_lease(root):
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 with (root/'gpu-owner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  lock.seek(0);lock.truncate();lock.write(json.dumps({'pid':os.getpid(),'started':time.time()}));lock.flush()
  try:yield
  finally:fcntl.flock(lock,fcntl.LOCK_UN)
