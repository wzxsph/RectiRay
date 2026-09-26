"""Launch a validated deployment profile without changing the frozen model code."""
import argparse,json,os,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--deployment',type=Path,required=True);a,rest=p.parse_known_args()
 config=json.loads(a.deployment.read_text())
 required={'bundle','backgrounds','cache','gc_interval'}
 if not required<=config.keys():raise ValueError('Incomplete deployment profile')
 if config['gc_interval'] not in (1,4,8):raise ValueError('Unsupported collection interval')
 os.environ.update(KERAS_BACKEND='torch',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 command=[sys.executable,'-m','ptrn_pipeline.runner','--bundle',config['bundle'],'--backgrounds',config['backgrounds'],'--cache',config['cache'],'--gc-interval',str(config['gc_interval']),'--eval-batch','1',*rest]
 os.execv(sys.executable,command)
if __name__=='__main__':main()
