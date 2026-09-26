"""Independent ten-second read-only sampler. Never starts/stops training."""
import argparse,json,os,time,subprocess,hashlib
from pathlib import Path

def read_json(path, default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default

def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)

def tail_record(path):
    try:
        with Path(path).open('rb') as f:
            f.seek(0,2);f.seek(max(0,f.tell()-65536));lines=f.read().decode(errors='replace').splitlines()
        for line in reversed(lines):
            try:return json.loads(line)
            except ValueError:pass
    except OSError:pass
    return None

def sample(root):
    import shutil
    s=read_json(root/'training-status.json',{});run=Path(s.get('run',str(root/'missing')))
    row=tail_record(run/'training.jsonl');progress={}
    for p in run.glob('**/*progress.json'):
        value=read_json(p)
        if value is not None:progress[str(p.relative_to(run))]=value
    runtime=root/'pipeline-v1/runtime'
    for pattern in ('*/progress.json','*/pipeline-progress.json','*progress.json'):
        for p in runtime.glob(pattern):
            value=read_json(p)
            if value is not None:progress['pipeline/'+str(p.relative_to(runtime))]=value
    proc=Path('/proc')/str(s.get('pid',0));alive=proc.exists()
    gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
    cgroup=Path('/sys/fs/cgroup');disk=shutil.disk_usage(root)
    job=read_json(runtime/'active-job.json',{});pipeline=read_json(runtime/'status.json',{});job_failed=False
    if job.get('pid') and pipeline.get('status')=='running' and job.get('phase')==pipeline.get('phase'):
        alive=Path('/proc',str(job['pid'])).exists()
        finished=any((runtime/job['phase']/n).exists() for n in ('completion.json','selected-gc.json','gpu-default-decision.json','decision.json','metrics-complete.json'))
        job_failed=not alive and not finished and time.time()-job.get('started',0)>20
    pipeline['active_job_failed']=job_failed
    return {'time':time.time(),'training':s,'last_step':row,'progress':progress,'process_alive':alive,
            'process_identity':(proc/'stat').read_text().split()[21] if alive else None,
            'gpu':gpu.stdout.strip(),'gpu_probe_exit':gpu.returncode,
            'cpu_stat':(cgroup/'cpu.stat').read_text() if (cgroup/'cpu.stat').exists() else None,
            'cpu_max':(cgroup/'cpu.max').read_text() if (cgroup/'cpu.max').exists() else None,
            'memory_bytes':int((cgroup/'memory.current').read_text()) if (cgroup/'memory.current').exists() else None,
            'disk_free_bytes':disk.free,'pipeline':pipeline,'active_job':read_json(runtime/'active-job.json',{})}

class Alerts:
    def __init__(self):self.signature=None;self.changed=time.time();self.active=set();self.run=None
    def save(self,path):atomic_json(path,{'signature':self.signature,'changed':self.changed,'active':sorted(self.active),'run':self.run})
    def restore(self,path):
        state=read_json(path,{})
        for key in ('signature','changed','run'):
            if key in state:setattr(self,key,state[key])
        self.active=set(state.get('active',[]))
    def inspect(self,s):
        now=s['time'];state=s['training'];run=state.get('run');events=[]
        if run!=self.run:self.run=run;self.signature=None;self.changed=now;self.active=set()
        # Exclude wall-clock timestamps: only actual work progress resets the timer.
        progress={k:{x:v.get(x) for x in ('evaluated','completed','count','step')} for k,v in s['progress'].items() if isinstance(v,dict)}
        signature=json.dumps([s['last_step'].get('step') if s['last_step'] else None,progress,state.get('phase'),s.get('pipeline',{}).get('phase')],sort_keys=True)
        if signature!=self.signature:self.changed=now;self.signature=signature
        flags=set()
        if state.get('status')=='failed' or (state.get('status')=='running' and not s['process_alive']):flags.add('process_failed')
        if (state.get('status')=='running' or s.get('pipeline',{}).get('status')=='running') and now-self.changed>=120:flags.add('no_stage_progress_120s')
        if s['disk_free_bytes']<15*2**30:flags.add('disk_below_15GiB')
        if (s['memory_bytes'] or 0)>48*2**30:flags.add('memory_above_48GiB')
        if s.get('pipeline',{}).get('status')=='failed' or s.get('pipeline',{}).get('active_job_failed'):flags.add('pipeline_failed')
        if s.get('pipeline',{}).get('all_required_work_complete'):flags.add('all_work_complete')
        if state.get('status')=='complete':flags.add('training_complete')
        for key in sorted(flags-self.active):events.append({'kind':key,'time':now,'run':run,'phase':state.get('phase')})
        for key in sorted(self.active-flags):
            if key!='training_complete':events.append({'kind':'recovered','previous':key,'time':now,'run':run})
        self.active=flags;return events

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--once',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(a.output/'sampler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    alert=Alerts();alert.restore(a.output/'alert-state.json');previous=read_json(a.output/'latest.json',{})
    while True:
        try:
            s=sample(a.root)
            if previous.get('cpu_stat') and s.get('cpu_stat'):
                def usage(x):return int(dict(line.split() for line in x.splitlines())['usage_usec'])
                elapsed=max(s['time']-previous['time'],.001);quota,period=s['cpu_max'].split();cores=int(quota)/int(period) if quota!='max' else os.cpu_count()
                s['cpu_quota_utilization_percent']=(usage(s['cpu_stat'])-usage(previous['cpu_stat']))/1e6/elapsed/cores*100
            events=alert.inspect(s);atomic_json(a.output/'latest.json',s);previous=s
            with (a.output/'events.jsonl').open('a') as f:
                for event in events:
                    event['id']=hashlib.sha256(json.dumps(event,sort_keys=True).encode()).hexdigest();f.write(json.dumps(event)+'\n')
            alert.save(a.output/'alert-state.json');(a.output/'sampler-error.json').unlink(missing_ok=True)
        except Exception as e:atomic_json(a.output/'sampler-error.json',{'time':time.time(),'error':repr(e)})
        if a.once:break
        time.sleep(10)
if __name__=='__main__':main()
