"""Post-training unpaired visual review; independent of frozen training source."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import cv2
import numpy as np
from .data import read_rgb,write_json
from .geometry import validate_quad
from .ratio_data import sha256
from .ratio_synthesis import letterbox,project
from .ratio_evaluation import predicted_quad,edge_ratio


def rectify_canvas(pixels,quad,ratio=None):
    """Preserve legacy max-edge sizing, or impose learned full-canvas W/H.

    This is a review image only, not a new application adapter or trusted plane.
    All four source-axis corners are sent to matching TL/TR/BR/BL target centers.
    """
    height,width=pixels.shape[:2]
    reasons=validate_quad(quad,(width,height))
    if reasons:
        raise ValueError(','.join(reasons))
    sides=np.linalg.norm(np.roll(quad,-1,axis=0)-quad,axis=1)
    out_width=max(32,round(max(sides[0],sides[2]))+1)
    out_height=max(32,round(max(sides[1],sides[3]))+1)
    if ratio is not None:
        if not np.isfinite(ratio) or not .1<=ratio<=10:
            raise ValueError('invalid_model_estimated_ratio')
        out_width=max(32,round(out_height*ratio))
    if out_width*out_height>24_000_000:
        raise ValueError('unreasonable_output_size')
    destination=np.array([[0,0],[out_width-1,0],[out_width-1,out_height-1],[0,out_height-1]],np.float32)
    h=cv2.getPerspectiveTransform(np.asarray(quad,np.float32),destination)
    result=cv2.warpPerspective(pixels,h,(out_width,out_height),flags=cv2.INTER_CUBIC)
    return result,h


def panel(pixels,title,ratio=None):
    tile=np.full((552,512,3),28,np.uint8)
    if pixels is not None:
        tile[40:552]=letterbox(pixels,512)[0]
    cv2.putText(tile,title,(9,21),cv2.FONT_HERSHEY_SIMPLEX,.48,(245,245,245),1)
    if ratio is not None:
        cv2.putText(tile,f'W/H={ratio:.4f}',(350,21),cv2.FONT_HERSHEY_SIMPLEX,.45,(200,230,255),1)
    return tile


def review(args):
    from .model import configure,build,torch
    from .ratio_model import predict
    from .ratio_export import load_candidate
    configure(False);cv2.setNumThreads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    files=sorted(p for p in args.real_dir.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg'})
    if len(files)!=47:
        raise ValueError('Frozen real-photo review expects 47 authorized images')
    originals=[]
    for path in files:
        originals.append(sha256(path))
    models={'B1.1':build(args.weights)[0]}
    configs={}
    for group in ('G0','G1','G2'):
        models[group],configs[group]=load_candidate(args.weights,args.run/group/'step-1250.pt')
    records=[]
    model_sha={group:sha256(args.run/group/'step-1250.pt') for group in configs}
    for ordinal,path in enumerate(files):
        try:pixels=read_rgb(path)
        except Exception as error:raise RuntimeError(f'Image ordinal {ordinal+1}: {type(error).__name__}') from None
        height,width=pixels.shape[:2]
        quads={};predictions={}
        for group,model in models.items():
            if group=='B1.1':
                input_pixels=cv2.resize(pixels,(224,224),interpolation=cv2.INTER_AREA)
                p=predict(model,input_pixels[None],'cpu')[0]
                q=(predicted_quad(p,224,True)+.5)*[width/224,height/224]-.5
            else:
                size=configs[group]['input_size']
                input_pixels,lb=letterbox(pixels,size)
                p=predict(model,input_pixels[None],'cpu')[0]
                q=project(predicted_quad(p,size),np.linalg.inv(lb))
            quads[group]=q;predictions[group]=p
        entry={'ordinal':ordinal+1,'scope':'unpaired_development_visual_not_accuracy','models':{}}
        tiles=[]
        specifications=[('B1.1','max_edges'),('G0','max_edges'),('G1','max_edges'),('G1','learned_ratio'),('G2','max_edges'),('G2','learned_ratio')]
        for group,ratio_mode in specifications:
            ratio=float(np.exp(predictions[group][8])) if ratio_mode=='learned_ratio' else None
            name=f'{group} {ratio_mode}'
            group_record=entry['models'].setdefault(group,{'quad':quads[group].tolist(),'max_edge_ratio':float(edge_ratio(quads[group])),
                              'learned_ratio':float(np.exp(predictions[group][8])) if len(predictions[group])==9 else None,
                              'ratio_evidence':'model_estimated','outputs':{}})
            try:
                rectified,h=rectify_canvas(pixels,quads[group],ratio)
                filename=f'photo-{ordinal+1:03d}-{group}-{ratio_mode}.local.jpg'
                cv2.imwrite(str(args.output/filename),cv2.cvtColor(rectified,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,94])
                actual_ratio=rectified.shape[1]/rectified.shape[0]
                tiles.append(panel(rectified,name,actual_ratio))
                group_record['outputs'][ratio_mode]={'valid':True,'filename':filename,'size':[rectified.shape[1],rectified.shape[0]],'H':h.tolist()}
            except ValueError as error:
                tiles.append(panel(None,name+' INVALID'))
                group_record['outputs'][ratio_mode]={'valid':False,'reason':str(error)}
        # Fixed order: baseline/G0/G1 edge above G1 learned/G2 edge/G2 learned.
        contact=np.concatenate([np.concatenate(tiles[:3],axis=1),np.concatenate(tiles[3:],axis=1)],axis=0)
        cv2.imwrite(str(args.output/f'contact-{ordinal+1:03d}.local.jpg'),cv2.cvtColor(contact,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,94])
        records.append(entry)
        print(json.dumps({'phase':'real47_rectified_review','completed':ordinal+1,'total':47}),flush=True)
    if any(sha256(path)!=before for path,before in zip(files,originals,strict=True)):
        raise ValueError('Read-only real photo input changed')
    summary={'count':47,'source_images_unchanged':True,'training_source_sha':args.training_source_sha,
             'postprocess_script_sha256':sha256(Path(__file__)),'checkpoint_sha256':model_sha,
             'candidate_eligible':False,'purpose':'failed_research_diagnostic_only',
             'scope':'unpaired_development_visual_only','accuracy_or_independent_validation_claimed':False,
             'active_application_model_changed':False,'records':records}
    write_json(args.output/'review.json',summary)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--real-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--training-source-sha',required=True)
    args=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']='';review(args)


if __name__=='__main__':main()
