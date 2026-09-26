"""Choose the largest evaluation batch meeting existing full metric tolerances."""
import argparse,json,gc,time
from pathlib import Path
import numpy as np
import torch
from .cache import DiskCache,scene_key,digest,cached_scene,unpack_view
from .evaluation import load_model,inference
from .telemetry import atomic_json

def calibrate(bundle,backgrounds,cache_root,checkpoint,output,count=128):
 from ptrn_hip.partial_data import PartialDataset,sha256
 from ptrn_hip.partial_evaluation import measure
 from ptrn_hip.ratio_synthesis import project
 torch.set_num_threads(4)
 data=PartialDataset(bundle,backgrounds,split='dev',verify=False);name=digest(data.fingerprint)[:16]
 inputs=DiskCache(cache_root,'dev-input-'+name,budget=2*2**30);full=DiskCache(cache_root,'dev-full-'+name,budget=10*2**30)
 model,cp=load_model(bundle,checkpoint);assert cp['binding']['development']==data.fingerprint
 assert count<=len(data)
 samples=[inputs.get(scene_key(data,i,full=False))[0] for i in range(count)]
 def metrics(pred):
  result=[]
  for i in range(count):
   _,arrays,meta,_=cached_scene(data,i,full);view=unpack_view(arrays,meta)
   result.append(measure(view,project(pred['quad'][i],samples[i]['inverse_letterbox']),float(pred['ratio'][i])))
  return result
 reports=[]
 with torch.no_grad():
  old=[inference(model,[s]) for s in samples];reference={k:np.concatenate([r[k] for r in old]) for k in old[0]}
  original_metrics=metrics(reference)
  for batch in (32,16,8,1):
   start=time.perf_counter();pieces=[inference(model,samples[i:i+batch]) for i in range(0,count,batch)]
   prediction={k:np.concatenate([r[k] for r in pieces]) for k in pieces[0]};inference_seconds=time.perf_counter()-start
   tested=metrics(prediction)
   quad=float(np.abs(prediction['quad']-reference['quad']).max());ratio=float(np.abs(prediction['log_ratio']-reference['log_ratio']).max())
   difference={key:max((abs(x[key]-y[key]) for x,y in zip(original_metrics,tested) if key in x and key in y),default=0.)
      for key in ('additional_crop_fraction','extra_background_fraction','visible_grid_p90','visible_grid_max')}
   valid=all(x['valid_geometry']==y['valid_geometry'] for x,y in zip(original_metrics,tested))
   passed=quad<.005 and ratio<.0001 and valid and all(v<.001 for v in difference.values())
   reports.append({'batch':batch,'count':count,'quad_max_pixels':quad,'log_ratio_max':ratio,'metrics_max':difference,'validity_equal':valid,'passed':passed,'inference_seconds':inference_seconds})
   gc.collect()
 selected=next((r['batch'] for r in reports if r['passed']),None)
 result={'checkpoint_sha256':sha256(checkpoint),'development':data.fingerprint,'reference':'original FP32, same TF32 flags','cudnn_tf32':torch.backends.cudnn.allow_tf32,
  'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'results':reports,'selected_batch':selected,'full_development_checked':count==len(data)}
 atomic_json(output,result)
 if selected is None:raise RuntimeError('Even original batch1 is not reproducible; do not deploy')
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--backgrounds',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=128);a=p.parse_args()
 print(json.dumps(calibrate(a.bundle,a.backgrounds,a.cache,a.checkpoint,a.output,a.count)))
if __name__=='__main__':main()
