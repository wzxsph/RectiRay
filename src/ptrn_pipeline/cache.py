"""Lossless bounded cache, hash-sharded files, atomic publication and age-based eviction."""
import hashlib,io,json,os,time,zipfile,fcntl,shutil,sqlite3
from functools import lru_cache
from pathlib import Path
import numpy as np
from .telemetry import atomic_json

GIB=2**30

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def encode(arrays, meta):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        z.writestr('metadata.json',json.dumps(meta,separators=(',',':')))
        for name,value in arrays.items():
            a=io.BytesIO();np.save(a,np.asarray(value),allow_pickle=False);z.writestr(name+'.npy',a.getvalue())
    data=buffer.getvalue();return hashlib.sha256(data).hexdigest().encode()+b'\n'+data

def decode(blob):
    checksum,body=blob.split(b'\n',1)
    if hashlib.sha256(body).hexdigest().encode()!=checksum:raise ValueError('Cache checksum mismatch')
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        meta=json.loads(z.read('metadata.json'))
        arrays={name[:-4]:np.load(io.BytesIO(z.read(name)),allow_pickle=False) for name in z.namelist() if name.endswith('.npy')}
    return arrays,meta

class DiskCache:
    def __init__(self,root,namespace,*,budget=12*GIB,reserve=15*GIB):
        if not namespace or '/' in namespace or namespace in ('.','..'):raise ValueError('Invalid cache namespace')
        self.base=Path(root);self.root=self.base/namespace;self.budget=budget;self.reserve=reserve
        self.root.mkdir(parents=True,exist_ok=True)
    def path(self,key):
        if len(key)!=64 or any(c not in '0123456789abcdef' for c in key):raise ValueError('Invalid cache key')
        return self.root/key[:2]/(key+'.npz')
    def get(self,key):
        p=self.path(key)
        try:
            result=decode(p.read_bytes());os.utime(p,None);return result
        except OSError:return None
        except (ValueError,zipfile.BadZipFile,KeyError):
            return None
    def put(self,key,arrays,meta):
        blob=encode(arrays,meta);p=self.path(key)
        if len(blob)>self.budget:return False
        with (self.base/'.quota.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            with sqlite3.connect(self.base/'index.sqlite') as db:
                db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
                db.execute('CREATE TABLE IF NOT EXISTS entries(path TEXT PRIMARY KEY,namespace TEXT,size INTEGER,stamp REAL)')
                db.execute('CREATE INDEX IF NOT EXISTS namespace ON entries(namespace)')
                db.execute('CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT)')
                if not db.execute("SELECT value FROM state WHERE key='imported'").fetchone():
                    db.executemany('INSERT OR REPLACE INTO entries VALUES(?,?,?,?)',
                        [(str(f.relative_to(self.base)),f.parent.parent.name,f.stat().st_size,f.stat().st_mtime) for f in self.base.glob('*/*/*.npz')])
                    db.execute("INSERT INTO state VALUES('imported','1')")
                relative=str(p.relative_to(self.base));old=db.execute('SELECT size FROM entries WHERE path=?',(relative,)).fetchone()
                old_size=old[0] if old else 0
                own=db.execute('SELECT COALESCE(SUM(size),0) FROM entries WHERE namespace=?',(self.root.name,)).fetchone()[0]
                total=db.execute('SELECT COALESCE(SUM(size),0) FROM entries').fetchone()[0]
                category='train-%' if self.root.name.startswith('train-') else 'dev-%'
                category_limit=8*GIB if category=='train-%' else 12*GIB
                category_size=db.execute('SELECT COALESCE(SUM(size),0) FROM entries WHERE namespace LIKE ?',(category,)).fetchone()[0]
                free=shutil.disk_usage(self.base).free
                while own-old_size+len(blob)>self.budget or category_size-old_size+len(blob)>category_limit or total-old_size+len(blob)>20*GIB or free-len(blob)<self.reserve:
                    local=own-old_size+len(blob)>self.budget
                    query='SELECT path,namespace,size FROM entries WHERE path!=?'
                    params=[relative]
                    if local:query+=' AND namespace=?';params.append(self.root.name)
                    elif category_size-old_size+len(blob)>category_limit:
                        query+=' AND namespace LIKE ?';params.append(category)
                        if category=='dev-%':query+=" AND namespace NOT LIKE 'dev-input-%'"
                    else:query+=" AND namespace NOT LIKE 'dev-input-%'"
                    query+=' ORDER BY stamp LIMIT 1';entry=db.execute(query,params).fetchone()
                    if entry is None:return False
                    name,namespace,n=entry;target=self.base/name
                    if not target.resolve().is_relative_to(self.base.resolve()) or len(Path(name).parts)!=3 or target.suffix!='.npz' or len(target.stem)!=64:raise ValueError('Unsafe cache index')
                    target.unlink(missing_ok=True);db.execute('DELETE FROM entries WHERE path=?',(name,))
                    total-=n;free=shutil.disk_usage(self.base).free
                    if namespace==self.root.name:own-=n
                    if namespace.startswith(category[:-1]):category_size-=n
                p.parent.mkdir(exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
                try:
                    with tmp.open('wb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
                    os.replace(tmp,p)
                    db.execute('INSERT OR REPLACE INTO entries VALUES(?,?,?,?)',(relative,self.root.name,len(blob),time.time()))
                finally:tmp.unlink(missing_ok=True)
        return True

@lru_cache(maxsize=1)
def renderer_hash():
    import ptrn_hip.partial_synthesis as synthesis
    return digest({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(synthesis.__file__).parent.glob('*.py')})

def scene_key(dataset,index,renderer='opencv-v3',full=True):
    return digest({'fingerprint':dataset.fingerprint,'index':index,'round':dataset.round_index,
                   'renderer':renderer,'renderer_code':renderer_hash(),**({'gpu_code':gpu_renderer_hash()} if renderer!='opencv-v3' else {}),'full':full})

def view_arrays(view,size):
    arrays={'photo':view.photo,'quad_photo':view.quad,'film_mask':view.film_mask,
            'screen_mask':view.screen_mask,'transform':view.transform,'ratio':np.array(view.ratio)}
    arrays.update({'input_'+k:v for k,v in view.input(size).items()})
    return arrays,view.info

def unpack_view(arrays,meta):
    from ptrn_hip.partial_synthesis import PartialView
    return PartialView(arrays['photo'],arrays['quad_photo'],arrays['film_mask'],arrays['screen_mask'],
                       arrays['transform'],float(arrays['ratio']),meta)

def cached_scene(dataset,index,cache):
    key=scene_key(dataset,index);item=cache.get(key)
    if item is not None:return key,item[0],item[1],True
    view=dataset.render(index);arrays,meta=view_arrays(view,dataset.size)
    stored=cache.put(key,arrays,meta)
    # A small, separately-budgeted input cache makes missing full scenes cheap to predict.
    return key,arrays,meta,False

@lru_cache(maxsize=1)
def gpu_renderer_hash():
    return digest({name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('gpu_render.py','gpu_dataset.py')})
