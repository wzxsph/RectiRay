"""Resumable, sequential frozen G0/G1/G2 experiment. See docs/ratio-v1.md."""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import cv2
import numpy as np
from .data import write_json
from .ratio_data import Sources, audit, sha256
from .ratio_synthesis import SEED, MODES, LEVELS, sample, stream, project, letterbox

GROUPS = {'G0': (224,False,16), 'G1': (224,True,16), 'G2': (384,True,4)}
RECIPE = {'seed':SEED,'updates':1250,'effective_batch':64,'head_updates':150,
          'head_lr':5e-5,'last_block_lr':1e-5,'clip_global_norm':1.,'evaluation_every':250,
          'geometry_probabilities':[.3,.6,.1],'appearance_probabilities':[.45,.35,.15,.05],
          'strong_geometry_strong_appearance':'downgrade_appearance_to_light',
          'common_photo_long_side':768,'BN':'frozen','mixed_precision':'bfloat16_autocast_loss_float32',
          'loss':'SmoothL1(corner_error/(.01*input_diagonal),beta=1)+.25*SmoothL1(logratio_error/.05,beta=1)',
          'evaluation_noisy_probabilities':{'light':7/11,'medium':3/11,'strong':1/11},
          'selection':'G0_min_clean_all_corner_mean; G1_G2_min_clean_mild_ratio_median; earliest_tie',
          'background':'independent_procedural', 'source_reads':'original_EXIF_canvas_no_square_cache'}


def configure():
    from .model import configure as setup, torch
    cv2.setNumThreads(1)
    environment = setup(False)
    if not torch.cuda.is_available():
        raise RuntimeError('This frozen run requires the authorized local CUDA device')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('Frozen bfloat16 smoke requires device support')
    return environment


def worker_init(_):
    cv2.setNumThreads(1)


class TrainingSet:
    def __init__(self,manifest,indices,size,start=0,stop=80000):
        self.manifest,self.indices,self.size = manifest,np.array(indices),size
        self.start,self.stop = start,stop
        self.sources = None
        self.orders = {}

    def __len__(self):
        return self.stop-self.start

    def __getitem__(self,index):
        ordinal = index+self.start
        epoch,position = divmod(ordinal,len(self.indices))
        if epoch not in self.orders:
            self.orders = {epoch:np.random.default_rng(np.random.SeedSequence([SEED,epoch,99])).permutation(self.indices)}
        source_index = int(self.orders[epoch][position])
        if self.sources is None:
            self.sources = Sources(self.manifest,cache_size=2)
        view = sample(self.sources.read(source_index),source_index,ordinal)
        image,label,quad = view.input(self.size)
        return image,quad,label[8],np.array([MODES.index(view.info['mode']),LEVELS.index(view.info['appearance']),LEVELS.index(view.info['requested_appearance']),int(view.info['moire'])],np.int64)


class EvaluationSet:
    def __init__(self,manifest,indices,size,legacy=False):
        self.manifest,self.indices,self.size,self.legacy = manifest,list(indices),size,legacy
        self.sources = None

    def __len__(self):
        return len(self.indices)*3

    def __getitem__(self,index):
        ordinal,mode_index = divmod(index,3)
        source_index = self.indices[ordinal]
        if self.sources is None:
            self.sources = Sources(self.manifest,cache_size=2)
        level = str(stream(source_index,200000+index,3).choice(('light','medium','strong'),p=(7/11,3/11,1/11)))
        view = sample(self.sources.read(source_index),source_index,200000+index,mode=MODES[mode_index],level=level)
        if self.legacy:
            width,height = view.info['common_size']
            clean = cv2.resize(view.clean,(self.size,self.size),interpolation=cv2.INTER_AREA)
            noisy = cv2.resize(view.noisy,(self.size,self.size),interpolation=cv2.INTER_AREA)
            quad = (view.quad+.5)*np.array([self.size/width,self.size/height])-.5
        else:
            clean,_,quad = view.input(self.size,noisy=False)
            noisy,_,quad2 = view.input(self.size,noisy=True)
            np.testing.assert_array_equal(quad,quad2)
        return clean,noisy,quad.astype(np.float32),np.float32(view.log_ratio),mode_index,np.array(view.info['common_size'])


def loader(dataset,batch,workers):
    from .model import torch
    return torch.utils.data.DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=workers,
            worker_init_fn=worker_init,pin_memory=True,**({'multiprocessing_context':'spawn','persistent_workers':True,'prefetch_factor':2} if workers else {}))


def gpu_memory():
    from .model import torch
    return {'allocated_mib':round(torch.cuda.memory_allocated()/2**20,1),
            'peak_allocated_mib':round(torch.cuda.max_memory_allocated()/2**20,1),
            'peak_reserved_mib':round(torch.cuda.max_memory_reserved()/2**20,1)}


def status(path,**record):
    write_json(path,{'time_unix':time.time(),'pid':os.getpid(),**record})


def evaluate(model,args,indices,size,legacy=False,heartbeat=None):
    from .ratio_model import predict
    from .ratio_evaluation import measure,summarize
    rows = {condition:{mode:[] for mode in MODES} for condition in ('clean','noisy')}
    data = loader(EvaluationSet(args.manifest,indices,size,legacy),args.eval_batch,args.workers)
    count = 0
    for clean,noisy,quad,logr,mode,photo_size in data:
        for condition,pixels in (('clean',clean),('noisy',noisy)):
            predictions = predict(model,pixels.numpy(),'cuda')
            for p,q,r,m,ps in zip(predictions,quad.numpy(),logr.numpy(),mode.numpy(),photo_size.numpy(),strict=True):
                rows[condition][MODES[int(m)]].append(measure(p,q,float(r),size,legacy,ps))
        count += len(clean)
        if heartbeat:
            heartbeat(count,len(data.dataset))
    result = {'scope':'exposed_synthetic_source_image_dev_not_independent','views_per_condition':count}
    for condition,groups in rows.items():
        result[condition] = {mode:summarize(values) for mode,values in groups.items()}
        result[condition]['all'] = summarize([r for values in groups.values() for r in values])
    return result


def check_disk(minimum_gib=3.):
    free = shutil.disk_usage('.').free/2**30
    if free < minimum_gib:
        raise RuntimeError(f'Insufficient free disk: {free:.2f} GiB; requires {minimum_gib:.2f}')
    return free


def ensure_source(expected):
    actual = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    if actual != expected or dirty:
        raise RuntimeError('Long run requires the saved, clean, explicitly confirmed source SHA')
    return actual


def read_split(args):
    audit_record = json.loads((args.run/'data-audit.json').read_text())
    if sha256(args.manifest) != audit_record['manifest_sha256']:
        raise RuntimeError('Input manifest changed after source audit')
    splits = json.loads((args.run/'split.local.json').read_text())
    if len(splits['dev']) != 478:
        raise RuntimeError('Frozen evaluation requires 478 dev sources')
    return audit_record,splits


def save_checkpoint(model,optimizer,path,step,stage_name,config,counters):
    from .model import torch
    check_disk(.5)
    payload = {'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
               'step':step,'stage':stage_name,'config':config,'counters':dict(counters),
               'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
    temporary = path.with_suffix('.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)
    write_json(path.with_suffix('.json'),{'sha256':sha256(path),'step':step,'source_sha':config['source_sha'],'config':config})


def group_train(args):
    from .model import torch,keras
    from .ratio_model import build_ratio,stage,tensor_input,losses,contract
    environment = configure()
    audit_record,splits = read_split(args)
    size,with_ratio,default_micro = GROUPS[args.group]
    micro = args.micro or default_micro
    if 64%micro or micro<1:
        raise ValueError('Microbatch must divide effective batch 64')
    source_sha = ensure_source(args.source_sha)
    directory = args.run/args.group
    directory.mkdir(parents=True,exist_ok=True)
    config = {**RECIPE,'group':args.group,'input_size':size,'ratio_head':with_ratio,'microbatch':micro,
              'accumulation':64//micro,'source_sha':source_sha,'initial_weights_sha256':sha256(args.weights),
              'data_audit_sha256':sha256(args.run/'data-audit.json'),'split_sha256':sha256(args.run/'split.local.json'),
              'mean_train_log_ratio':audit_record['mean_train_log_ratio'],'contract':contract(size,with_ratio)}
    config_path = directory/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Resume config changed; cannot silently change experiment')
    write_json(config_path,config)
    write_json(directory/'environment.json',environment)
    model,backbone = build_ratio(args.weights,size,with_ratio,audit_record['mean_train_log_ratio'])
    bn_before = {v.path:v.value.detach().cpu().clone() for layer in backbone.layers if isinstance(layer,keras.layers.BatchNormalization) for v in layer.weights}
    completed,checkpoint,counters = 0,None,Counter()
    paths = sorted(directory.glob('step-*.pt'))
    if paths:
        checkpoint = torch.load(paths[-1],map_location='cuda',weights_only=False)
        if checkpoint['config'] != config:
            raise ValueError('Checkpoint config mismatch')
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        completed = checkpoint['step']; counters.update(checkpoint['counters'])
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint['cuda_rng']])
    history_path = directory/'history.jsonl'
    if history_path.exists():
        history = [json.loads(line) for line in history_path.read_text().splitlines() if line]
        if any(row['optimizer_step']>completed for row in history):
            history_path.rename(directory/f'interrupted-history-{time.time_ns()}.jsonl')
            with history_path.open('w') as log:
                for row in history:
                    if row['optimizer_step']<=completed:
                        log.write(json.dumps(row)+'\n')
    stage_name = 'head' if completed<150 else 'last_block'
    parameters = stage(model,backbone,stage_name)
    optimizer = torch.optim.Adam(parameters,lr=5e-5 if stage_name=='head' else 1e-5)
    if checkpoint and checkpoint['stage']==stage_name:
        optimizer.load_state_dict(checkpoint['optimizer'])
    del checkpoint
    evaluations = {}
    for path in directory.glob('evaluation-*.json'):
        result = json.loads(path.read_text()); evaluations[result['step']] = result
    # Checkpoint precedes evaluation: a crash during dev evaluation is safely resumed here.
    def run_evaluation(step_number):
        status(directory/'status.json',state='evaluating',group=args.group,optimizer_step=step_number,samples_seen=step_number*64)
        def heartbeat(done,total):
            status(directory/'status.json',state='evaluating',group=args.group,optimizer_step=step_number,
                   samples_seen=step_number*64,evaluation_views_done=done,evaluation_views_total=total,**gpu_memory())
        result = {'step':step_number,**evaluate(model,args,splits['dev'],size,heartbeat=heartbeat)}
        write_json(directory/f'evaluation-{step_number:04d}.json',result)
        evaluations[step_number] = result
        score = lambda r:r['clean']['all']['corner_nme']['mean'] if args.group=='G0' else r['clean']['mild']['ratio_error']['median']
        best = min(evaluations.values(),key=lambda r:(score(r),r['step']))
        write_json(directory/'best.json',{'step':best['step'],'checkpoint':f"step-{best['step']:04d}.pt",'score':score(best),'selection':RECIPE['selection']})
        print(json.dumps({'event':'evaluation','group':args.group,**result}),flush=True)
    if completed and completed%250==0 and completed not in evaluations:
        run_evaluation(completed)
    if completed ==1250:
        status(directory/'status.json',state='complete',group=args.group,optimizer_step=completed,samples_seen=80000)
        return
    training = iter(loader(TrainingSet(args.manifest,splits['train'],size,completed*64),micro,args.workers))
    start,time_train,initial_step = time.perf_counter(),0.,completed
    for step_number in range(completed+1,1251):
        if step_number==151:
            stage_name='last_block'
            parameters=stage(model,backbone,stage_name)
            optimizer=torch.optim.Adam(parameters,lr=1e-5)
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        means = np.zeros(3)
        for microstep in range(64//micro):
            pixels,quad,logr,info = next(training)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                prediction = model(tensor_input(pixels.numpy(),'cuda'),training=False)
            loss,quad_loss,ratio_loss = losses(prediction,quad.to('cuda'),logr.to('cuda'),size)
            (loss/(64//micro)).backward()
            means += [float(loss.detach()),float(quad_loss.detach()),float(ratio_loss.detach())]
            for mode,level,requested,moire in info.numpy():
                counters['geometry_'+MODES[int(mode)]]+=1
                counters['appearance_'+LEVELS[int(level)]]+=1
                counters['requested_'+LEVELS[int(requested)]]+=1
                counters['moire']+=int(moire)
            status(directory/'status.json',state='training',group=args.group,optimizer_step=step_number-1,
                   microstep=microstep+1,accumulation=64//micro,samples_seen=(step_number-1)*64+(microstep+1)*micro,
                   elapsed_seconds=time.perf_counter()-start,loss=float(loss.detach()),
                   samples_per_second=(step_number-initial_step-1)*64/time_train if time_train else None,
                   eta_training_seconds=(1250-step_number+1)*time_train/(step_number-initial_step-1) if step_number>initial_step+1 else None,
                   **gpu_memory())
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in parameters):
            raise FloatingPointError('Non-finite weights after optimizer update')
        duration = time.perf_counter()-step_start
        time_train += duration
        means /= 64//micro
        record={'event':'update','group':args.group,'optimizer_step':step_number,'microstep':64//micro,
                'samples_seen':step_number*64,'stage':stage_name,'loss':float(means[0]),'quad_loss':float(means[1]),
                'ratio_loss':float(means[2]),'gradient_norm':float(gradient_norm),'seconds_per_update':duration,
                'samples_per_second':64/duration,'eta_training_seconds':(1250-step_number)*time_train/(step_number-initial_step),
                'elapsed_seconds':time.perf_counter()-start,**gpu_memory()}
        with (directory/'history.jsonl').open('a') as log:
            log.write(json.dumps(record)+'\n')
        status(directory/'status.json',state='training',**record)
        if step_number%10==0 or step_number==1:
            print(json.dumps(record),flush=True)
        if step_number%250==0:
            for layer in backbone.layers:
                if isinstance(layer,keras.layers.BatchNormalization):
                    for v in layer.weights:
                        if not torch.equal(bn_before[v.path],v.value.detach().cpu()):
                            raise ValueError('Frozen BatchNorm state changed')
            save_checkpoint(model,optimizer,directory/f'step-{step_number:04d}.pt',step_number,stage_name,config,counters)
            write_json(directory/'realized-distributions.json',{'count':step_number*64,'counts':dict(counters)})
            run_evaluation(step_number)
    status(directory/'status.json',state='complete',group=args.group,optimizer_step=1250,samples_seen=80000,
           elapsed_seconds=time.perf_counter()-start,BN_unchanged=True,**gpu_memory())


def baseline(args):
    from .model import build
    configure()
    _,splits=read_split(args)
    ensure_source(args.source_sha)
    model,_=build(args.weights)
    def heartbeat(done,total):
        status(args.run/'B1.1-status.json',state='evaluating',evaluation_views_done=done,evaluation_views_total=total)
    result=evaluate(model,args,splits['dev'],224,legacy=True,heartbeat=heartbeat)
    write_json(args.run/'B1.1-evaluation.json',{'frozen':True,'legacy_direct_square_input':True,'source_sha':args.source_sha,**result})
    status(args.run/'B1.1-status.json',state='complete')


def run_all(args):
    ensure_source(args.source_sha)
    check_disk(4.)
    read_split(args)
    command_base=[sys.executable,'-m','ptrn_hip.ratio_runner','--manifest',str(args.manifest),'--weights',str(args.weights),
                  '--run',str(args.run),'--workers',str(args.workers),'--eval-batch',str(args.eval_batch),'--source-sha',args.source_sha]
    status(args.run/'runner-status.json',state='running',phase='B1.1',source_sha=args.source_sha)
    if not (args.run/'B1.1-evaluation.json').exists():
        subprocess.run(command_base+['baseline'],check=True)
    for group in GROUPS:
        command=command_base+['train','--group',group]
        status(args.run/'runner-status.json',state='running',phase=group,source_sha=args.source_sha,command=command)
        subprocess.run(command,check=True)
    from .ratio_evaluation import select
    evaluations={g:[json.loads(p.read_text()) for p in (args.run/g).glob('evaluation-*.json')] for g in GROUPS}
    write_json(args.run/'selection.json',select(evaluations))
    status(args.run/'runner-status.json',state='training_complete_export_pending',source_sha=args.source_sha)


def preview(args):
    sources=Sources(args.manifest,cache_size=2)
    output=args.run/'preview'
    output.mkdir(parents=True,exist_ok=True)
    # Only anonymous ordinals are painted on these local, uncommitted review sheets.
    chosen=[0,len(sources.records)//2,len(sources.records)-1]
    records=[]
    for number,index in enumerate(chosen):
        source=sources.read(index)
        full,crops=[],[]
        views=[]
        for level in LEVELS:
            view=sample(source,index,42,mode='mild',level=level)
            views.append(view)
            overlay=view.noisy.copy()
            cv2.polylines(overlay,[np.rint(view.quad).astype(np.int32)],True,(0,255,0),2)
            for ordinal,q in enumerate(view.quad):
                cv2.putText(overlay,('TL','TR','BR','BL')[ordinal],tuple(np.rint(q).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,80,40),1)
            thumb=letterbox(overlay,384)[0]
            cv2.putText(thumb,level,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,0),1)
            full.append(thumb)
            cx,cy=np.rint(view.quad.mean(0)).astype(int)
            crop=view.noisy[max(0,cy-128):cy+128,max(0,cx-128):cx+128]
            crops.append(crop)
            records.append({'preview':number,'level':level,'logratio':view.log_ratio,**view.info})
        for view in views[1:]:
            np.testing.assert_array_equal(view.quad,views[0].quad)
            assert view.log_ratio==views[0].log_ratio
        cv2.imwrite(str(output/f'levels-{number:02d}.jpg'),cv2.cvtColor(np.concatenate(full,axis=1),cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output/f'crops-1to1-{number:02d}.png'),cv2.cvtColor(np.concatenate(crops,axis=1),cv2.COLOR_RGB2BGR))
        for mode in MODES:
            view=sample(source,index,57,mode=mode,level='light')
            for size in (224,384):
                image,label,quad=view.input(size)
                for k,q in enumerate(quad):
                    cv2.circle(image,tuple(np.rint(q).astype(int)),2,(0,255,0),-1)
                    cv2.putText(image,str(k),tuple(np.rint(q).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,80,0),1)
                cv2.imwrite(str(output/f'geometry-{number:02d}-{mode}-{size}.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    write_json(output/'preview-checks.json',{'labels_identical_across_appearance':True,'raw_ratio_preserved':True,'records':records})


def smoke(args):
    from .model import torch,keras,build
    from .ratio_model import build_ratio,stage,tensor_input,losses,predict
    environment=configure()
    source=Sources(args.manifest,cache_size=2)
    pixels=source.read(0)
    view=sample(pixels,0,123,mode='mild',level='medium')
    # Strict loading and unchanged H prediction across the new 224 wrappers.
    old,_=build(args.weights)
    image,_,_=view.input(224)
    expected=predict(old,image[None],'cuda')
    del old; keras.backend.clear_session(); gc.collect(); torch.cuda.empty_cache()
    results=[]
    for group,(size,ratio,micro) in GROUPS.items():
        model,backbone=build_ratio(args.weights,size,ratio,.228421)
        image,label,quad=view.input(size)
        actual=predict(model,image[None],'cuda')
        if size==224:
            np.testing.assert_array_equal(actual[:,:8],expected)
        if ratio:
            np.testing.assert_allclose(actual[:,8],.228421,atol=1e-7)
        frozen={v.path:v.value.detach().cpu().clone() for layer in backbone.layers if isinstance(layer,keras.layers.BatchNormalization) for v in layer.weights}
        torch.cuda.reset_peak_memory_stats()
        group_result={'group':group,'parameters':model.count_params(),'microbatch':micro,'accumulation':64//micro,'stages':{}}
        for name in ('head','last_block'):
            parameters=stage(model,backbone,name)
            optimizer=torch.optim.Adam(parameters,lr=5e-5 if name=='head' else 1e-5)
            started=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
            for _ in range(64//micro):
                x=np.repeat(image[None],micro,axis=0)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pred=model(tensor_input(x,'cuda'),training=False)
                loss,_,_=losses(pred,torch.tensor(np.repeat(quad[None],micro,axis=0),device='cuda'),torch.full((micro,),float(label[8]),device='cuda'),size)
                (loss/(64//micro)).backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            optimizer.step(); torch.cuda.synchronize()
            group_result['stages'][name]={'loss':float(loss.detach()),'gradient_norm':float(norm),'seconds_per_update':time.perf_counter()-started,'trainable_parameters':sum(p.numel() for p in parameters)}
        for layer in backbone.layers:
            if isinstance(layer,keras.layers.BatchNormalization):
                for v in layer.weights:
                    assert torch.equal(frozen[v.path],v.value.detach().cpu())
        group_result.update(gpu_memory());group_result['BN_unchanged']=True
        results.append(group_result)
        print(json.dumps(group_result),flush=True)
        del model,backbone,optimizer,parameters,pred,loss
        keras.backend.clear_session();gc.collect();torch.cuda.empty_cache()
    write_json(args.run/'smoke.json',{'environment':environment,'strict_B11_H224_bitwise_equal':True,'groups':results})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--eval-batch',type=int,default=4);p.add_argument('--source-sha')
    sub=p.add_subparsers(dest='action',required=True)
    for action in ('audit','preview','smoke','baseline','run-all'):
        sub.add_parser(action)
    train=sub.add_parser('train');train.add_argument('--group',choices=list(GROUPS),required=True);train.add_argument('--micro',type=int)
    args=p.parse_args();args.run.mkdir(parents=True,exist_ok=True)
    try:
        if args.action=='audit':audit(Sources(args.manifest),args.run)
        elif args.action=='preview':preview(args)
        elif args.action=='smoke':smoke(args)
        elif args.action=='baseline':baseline(args)
        elif args.action=='train':group_train(args)
        else:run_all(args)
    except Exception as error:
        location=args.run/args.group if getattr(args,'group',None) else args.run
        # Local diagnostic contains code traceback, never an echoed raw source filename.
        write_json(location/'failure.json',{'time_unix':time.time(),'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        status(location/'status.json',state='failed',exception=type(error).__name__,message=str(error))
        raise


if __name__=='__main__':
    main()
