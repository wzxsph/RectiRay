"""Checkpoint-bound GPU work interleaved with training; CPU metrics independent."""
from pathlib import Path
import gc,json,os,time,subprocess
import torch
from .cache import DiskCache,scene_key,digest
from .evaluation import load_model,inference,producer_identity
from .telemetry import atomic_json

class EvaluationJob:
    def __init__(self,bundle,backgrounds,cache_root,checkpoint,output,*,batch=1,workers=4,count=None,bank=None,use_graph=True):
        from ptrn_hip.partial_data import PartialDataset,sha256
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.data=PartialDataset(bundle,backgrounds,split='dev',verify=False)
        self.cache=DiskCache(cache_root,'dev-input-'+digest(self.data.fingerprint)[:16],budget=2*2**30)
        from .input_bank import InputBank
        self.count=min(count,len(self.data)) if count else len(self.data)
        self.owns_bank=bank is None
        self.bank=bank if bank is not None else InputBank(self.data,cache_root,self.count,eager=False)
        if len(self.bank)<self.count:raise ValueError('Validation input bank is shorter than requested count')
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
        try:self.model,cp=load_model(bundle,checkpoint)
        finally:torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
        assert cp['binding']['development']==self.data.fingerprint
        from .graph import GraphPredictor
        self.predictor=GraphPredictor(self.model,batch) if use_graph else None
        self.cursor=0;self.batch=batch;self.pending=[];self.started=time.time();self.gpu_seconds=0;self.input_not_ready_ticks=0
        binding={'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'step':cp['step'],'count':self.count,
                 'producer_pid':os.getpid(),'producer_identity':producer_identity(os.getpid()),'prediction_backend':'cuda_graph' if use_graph else 'eager','runtime_code':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},'batch':batch,'development':self.data.fingerprint,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32}
        atomic_json(self.output/'binding.json',binding)
        cmd=[os.sys.executable,'-m','ptrn_pipeline.evaluation','metrics','--bundle',str(bundle),'--backgrounds',str(backgrounds),
             '--cache',str(cache_root),'--output',str(self.output),'--workers',str(workers),'--count',str(self.count)]
        self.log=(self.output/'metrics-console.log').open('w');self.child=subprocess.Popen(cmd,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
    @property
    def gpu_done(self):return self.cursor==self.count
    def tick(self,batches=2):
        """Called between updates: no CPU metric work or synthesis on this path."""
        if self.gpu_done:return
        try:
            with torch.no_grad():
                for _ in range(batches):
                    ids=range(self.cursor,min(self.count,self.cursor+self.batch))
                    items=self.bank.ready_batch(self.cursor,len(ids))
                    if items is None:
                        self.input_not_ready_ticks+=1
                        break
                    t=time.perf_counter();pred=self.predictor(items) if self.predictor is not None and len(items)==self.batch else inference(self.model,items);self.gpu_seconds+=time.perf_counter()-t
                    for j,i in enumerate(ids):self.pending.append({'index':i,**{k:v[j].tolist() for k,v in pred.items()}})
                    self.cursor+=len(items)
                    self.bank.acknowledge(self.cursor)
                    if len(self.pending)>=128 or self.gpu_done:
                        atomic_json(self.output/f'pred-{self.pending[0]["index"]:06d}.json',self.pending);self.pending=[]
                        atomic_json(self.output/'progress.json',{'phase':'inference','completed':self.cursor,'count':self.count,'gpu_seconds':self.gpu_seconds,'seconds':time.time()-self.started,'input_ready':self.bank.prepared,'input_not_ready_ticks':self.input_not_ready_ticks})
                    if self.gpu_done:
                        atomic_json(self.output/'inference-complete.json',{'count':self.count,'gpu_seconds':self.gpu_seconds})
                        if self.owns_bank:self.bank.close()
                        self.predictor=None;del self.model;self.model=None;gc.collect();torch.cuda.empty_cache();break
        except BaseException as e:
            if self.owns_bank:self.bank.close()
            atomic_json(self.output/'inference-failed.json',{'error':repr(e)});raise
    def status(self):
        if (self.output/'evaluation.json').exists():return 'complete'
        if (self.output/'metrics-failed.json').exists() or (self.output/'inference-failed.json').exists():
            if self.owns_bank:self.bank.close()
            if self.model is not None:self.predictor=None;self.model=None;gc.collect();torch.cuda.empty_cache()
            return 'failed'
        if self.child.poll() is not None:
            atomic_json(self.output/'metrics-failed.json',{'error':'Metric process exited without a complete result','exit_code':self.child.returncode})
            if self.owns_bank:self.bank.close()
            self.predictor=None;self.model=None;gc.collect();torch.cuda.empty_cache()
            return 'failed'
        return 'metrics' if self.gpu_done else 'predicting'
