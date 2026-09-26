import json
import time
from concurrent.futures import Future
from pathlib import Path
import pytest
import numpy as np
import ptrn_pipeline.evaluation as evaluation

class ImmediatePool:
 def __init__(self,**kwargs):pass
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def submit(self,fn,arg):
  f=Future()
  try:f.set_result(fn(arg))
  except Exception as e:f.set_exception(e)
  return f

def row(job):return {'index':job[0],'valid_geometry':True,'kind':'keep','integrity':'complete','background_family':'none'}

def setup(tmp_path,indices):
 items=[{'index':i,'quad':[[0,0]]*4,'ratio':1,'baseline_ratio':1} for i in indices]
 (tmp_path/'pred-000000.json').write_text(json.dumps(items));(tmp_path/'binding.json').write_text('{}');(tmp_path/'inference-complete.json').write_text('{}')

def test_out_of_order_sorted_without_dropping(tmp_path,monkeypatch):
 monkeypatch.setattr(evaluation.cf,'ProcessPoolExecutor',ImmediatePool);monkeypatch.setattr(evaluation,'metric',row)
 setup(tmp_path,[2,0,1]);evaluation.consume_metrics('.', '.', '.',tmp_path,2,3)
 result=json.loads((tmp_path/'evaluation.json').read_text());assert [r['index'] for r in result['rows']]==[0,1,2]
 assert result['full_development'] is False

@pytest.mark.parametrize('indices',[[0,0],[0]])
def test_duplicate_or_missing_cannot_complete(tmp_path,monkeypatch,indices):
 monkeypatch.setattr(evaluation.cf,'ProcessPoolExecutor',ImmediatePool);monkeypatch.setattr(evaluation,'metric',row);setup(tmp_path,indices)
 with pytest.raises(ValueError):evaluation.consume_metrics('.','.','.',tmp_path,2,2)
 assert not (tmp_path/'evaluation.json').exists();assert (tmp_path/'metrics-failed.json').exists()

def test_producer_crash_and_metric_crash_are_reported(tmp_path,monkeypatch):
 monkeypatch.setattr(evaluation.cf,'ProcessPoolExecutor',ImmediatePool);setup(tmp_path,[0]);(tmp_path/'inference-failed.json').write_text('{}')
 with pytest.raises(RuntimeError):evaluation.consume_metrics('.','.','.',tmp_path,1,1)
 assert not (tmp_path/'evaluation.json').exists()

def test_killed_producer_detected_without_failure_marker(tmp_path,monkeypatch):
 monkeypatch.setattr(evaluation.cf,'ProcessPoolExecutor',ImmediatePool)
 (tmp_path/'binding.json').write_text(json.dumps({'producer_pid':999999999,'producer_identity':'1'}))
 with pytest.raises(RuntimeError,match='exited before'):evaluation.consume_metrics('.','.','.',tmp_path,1,1)
 assert (tmp_path/'metrics-failed.json').exists();assert not (tmp_path/'metrics-complete.json').exists()

def test_metric_exception_preserved(tmp_path,monkeypatch):
 monkeypatch.setattr(evaluation.cf,'ProcessPoolExecutor',ImmediatePool);setup(tmp_path,[0])
 def broken(job):raise RuntimeError('metric crash')
 monkeypatch.setattr(evaluation,'metric',broken)
 with pytest.raises(RuntimeError,match='metric crash'):evaluation.consume_metrics('.','.','.',tmp_path,1,1)
 assert not (tmp_path/'evaluation.json').exists()

def test_async_eval_skips_unready_input_without_advancing_sample(tmp_path):
 from ptrn_pipeline.async_eval import EvaluationJob
 class Bank:
  prepared=1
  def __init__(self):self.calls=0;self.acked=[]
  def ready_batch(self,start,size):
   self.calls+=1
   return None if self.calls==1 else [{'pixels':np.zeros((2,2,3),np.uint8)}]
  def acknowledge(self,end):self.acked.append(end)
 class Predictor:
  def __call__(self,items):return {'quad':np.zeros((1,4,2),np.float32),'ratio':np.ones(1,np.float32)}
 job=EvaluationJob.__new__(EvaluationJob)
 job.count=2;job.cursor=0;job.batch=1;job.bank=Bank();job.predictor=Predictor()
 job.model=None;job.pending=[];job.output=tmp_path;job.started=time.time()
 job.gpu_seconds=0;job.input_not_ready_ticks=0;job.owns_bank=False
 job.tick(1)
 assert job.cursor==0 and job.pending==[] and job.input_not_ready_ticks==1
 job.tick(1)
 assert job.cursor==1 and job.pending[0]['index']==0 and job.bank.acked==[1]
