"""Opt-in GPU renderer v1. New pixels/version, unchanged reference-plane semantics."""
from dataclasses import dataclass
import concurrent.futures as cf
import numpy as np
import cv2
import torch
from torch.nn import functional as F
from ptrn_hip.ratio_synthesis import pixel_corners,project,APPEARANCE
from ptrn_hip.spatial_synthesis import _camera,_stream,TOPOLOGIES
from ptrn_hip.partial_synthesis import pre_crop,_background,PartialView
from ptrn_hip.partial_geometry import visibility

RENDERER='torch-photo-v1'

@dataclass
class Plan:
    source:np.ndarray
    background:np.ndarray
    screen:np.ndarray
    transform:np.ndarray
    quad:np.ndarray
    ratio:float
    crop:tuple
    info:dict
    seed:int
    level:str

def viewer_background(shape,plane,camera,pw,ph,source_index,view_index,domain,seed):
    width,height=shape;rng=_stream(source_index,view_index,domain,1,seed)
    topology=TOPOLOGIES[int(rng.integers(len(TOPOLOGIES)))];small=rng.integers(35,185,(5,5,3),dtype=np.uint8)
    canvas=cv2.resize(small,(width,height),interpolation=cv2.INTER_CUBIC)
    film=project(pixel_corners(pw,ph),plane);left,top=film.min(0);right,bottom=film.max(0);fw,fh=right-left,bottom-top
    m=rng.uniform(.035,.12,4);sl,st,sr,sb=left-m[0]*fw,top-m[1]*fh,right+m[2]*fw,bottom+m[3]*fh
    def polygon(target,points,color):cv2.fillConvexPoly(target,np.rint(project(np.array(points),camera)).astype(np.int32),color)
    screen=np.zeros((height,width),np.float32);points=[[sl,st],[sr,st],[sr,sb],[sl,sb]];polygon(screen,points,1.)
    if topology=='light_page':canvas=np.clip(canvas.astype(np.int16)+50,0,245).astype(np.uint8);color=(30,34,40)
    elif topology=='dual_panel':
        color=(22,25,30);polygon(canvas,[[sr,st],[sr+.24*fw,st],[sr+.24*fw,sb],[sr,sb]],(205,210,215))
    elif topology=='physical_screen':
        color=(38,42,48);polygon(canvas,[[sl-.1*fw,st-.1*fh],[sr+.1*fw,st-.1*fh],[sr+.1*fw,sb+.1*fh],[sl-.1*fw,sb+.1*fh]],(9,11,15))
    else:color=(16,19,24)
    polygon(canvas,points,color);polygon(canvas,[[sl,st],[sr,st],[sr,top],[sl,top]],(205,208,213) if topology=='light_page' else (62,67,74))
    for fraction in np.linspace(.08,.88,7):
        x=sl+fraction*(sr-sl);polygon(canvas,[[x,st+.18*(top-st)],[x+.025*fw,st+.18*(top-st)],[x+.025*fw,st+.72*(top-st)],[x,st+.72*(top-st)]],(170,178,185))
    return canvas,screen,topology

def missing_crop(q,transform,plane_size,photo_size,target,rng):
    w,h=photo_size;edge=int(rng.integers(4));edges=[edge]
    if rng.random()<.35:edges.append((edge+1)%4)
    low,high=0.,.95;best=None
    for _ in range(30):
        t=(low+high)/2;b=[0,0,w,h]
        for side in edges:
            if side==0:b[0]=min(w-2,int(round(t*(w-1))))
            elif side==1:b[1]=min(h-2,int(round(t*(h-1))))
            elif side==2:b[2]=max(2,w-int(round(t*(w-1))))
            else:b[3]=max(2,h-int(round(t*(h-1))))
        x,y,r,bt=b;shift=np.array([[1,0,-x],[0,1,-y],[0,0,1.]])
        try:vis=visibility(project(q,shift),shift@transform,plane_size,(r-x,bt-y))
        except ValueError:high=t;continue
        error=abs(vis['missing_fraction']-target)
        if best is None or error<best[0]:best=(error,tuple(b),vis)
        if vis['missing_fraction']<target:low=t
        else:high=t
    if best is None or best[0]>.008:raise RuntimeError('Missing quota unsatisfied')
    return best[1]

def plan(source,source_index,view_index,*,seed=20260925,domain='train',kind='photo',integrity='complete',family='viewer_screen',
         source_crop=False,high_occupancy=True,background=None,mode='mild',level='light',pure_roll=False):
    if kind not in ('keep','photo') or integrity not in ('complete','light','moderate') or family not in ('viewer_screen','coco_context','near_color','mixed_color'):
        raise ValueError('Unsupported scene stratum')
    if domain not in ('train','dev') or level not in APPEARANCE or mode not in ('front','mild','medium','hard'):
        raise ValueError('Unsupported scene configuration')
    rng=np.random.default_rng(np.random.SeedSequence([seed,source_index,view_index,913,domain=='dev']))
    plane,crop=pre_crop(source,rng,source_crop);ph,pw=plane.shape[:2]
    if kind=='keep':
        if source_crop or integrity!='complete':raise ValueError('Original must be preserved')
        info={**visibility(pixel_corners(pw,ph),np.eye(3),(pw,ph),(pw,ph)),'kind':'keep','integrity':'complete','family':'none','renderer':RENDERER,'occupancy':1.}
        return Plan(plane,plane,np.ones((ph,pw),np.float32),np.eye(3),pixel_corners(pw,ph),pw/ph,(0,0,pw,ph),info,seed,level)
    transform,q,camera,plane_map,info=_camera(pw,ph,source_index,view_index,domain,seed,mode,1 if high_occupancy else 0,pure_roll)
    w,h=info['photo_size']
    if family=='viewer_screen':bg,screen,topology=viewer_background((w,h),plane_map,camera,pw,ph,source_index,view_index,domain,seed)
    else:bg=_background(plane,rng,(h,w),family,background);screen=np.zeros((h,w),np.float32);topology=None
    bounds=(0,0,w,h)
    if integrity!='complete':bounds=missing_crop(q,transform,(pw,ph),(w,h),float(rng.uniform(.02,.09) if integrity=='light' else rng.uniform(.12,.28)),rng)
    info.update(kind='photo',family=family,integrity=integrity,source_pre_crop=crop,renderer=RENDERER,topology=topology,
                requested_high_occupancy=high_occupancy,domain=domain,mode=mode,source_index=source_index,view_index=view_index)
    return Plan(plane,bg,screen,transform,q,pw/ph,bounds,info,int(rng.integers(2**31)),level)

def warp_batch(plans,device='cuda'):
    """Batch variable source/canvas sizes by padding, without resizing the sources."""
    n=len(plans);hs=max(p.source.shape[0] for p in plans);ws=max(p.source.shape[1] for p in plans)
    ht=max(p.background.shape[0] for p in plans);wt=max(p.background.shape[1] for p in plans)
    if n*(4*hs*ws+8*ht*wt)*4>2*2**30:raise MemoryError('Renderer temporary allocation budget exceeded; reduce render batch')
    pixels=torch.zeros((n,4,hs,ws),device=device)
    for i,p in enumerate(plans):
        h,w=p.source.shape[:2];pixels[i,:3,:h,:w]=torch.tensor(p.source,device=device).permute(2,0,1).float();pixels[i,3,:h,:w]=1
    y,x=torch.meshgrid(torch.arange(ht,device=device),torch.arange(wt,device=device),indexing='ij')
    target=torch.stack((x,y,torch.ones_like(x)),-1).float().reshape(-1,3)
    inverse=torch.as_tensor(np.stack([np.linalg.inv(p.transform) for p in plans]),device=device,dtype=torch.float32)
    coordinates=target[None]@inverse.transpose(1,2);coordinates=coordinates[...,:2]/coordinates[...,2:3]
    grid=(coordinates/coordinates.new_tensor([ws-1,hs-1])*2-1).reshape(n,ht,wt,2)
    return F.grid_sample(pixels,grid,mode='bilinear',padding_mode='zeros',align_corners=True)

def appearance_gpu(photo,screen,seed,level):
    if level=='clear':return photo.clamp(0,255).to(torch.uint8),{'camera':'none','moire':False},None
    rng=np.random.default_rng(seed);prob,amp,noise,lo,hi,blur,quality,light=APPEARANCE[level]
    h,w=screen.shape;y,x=torch.meshgrid(torch.arange(h,device=photo.device),torch.arange(w,device=photo.device),indexing='ij')
    a=photo/255;gamma=float(rng.uniform(lo,hi));a+=(a.pow(gamma)-a)*screen
    a+=((rng.uniform(-light,light)*(x/max(w-1,1)-.5)+rng.uniform(-light,light)*(y/max(h-1,1)-.5))*screen)[None]
    moire=bool(rng.random()<prob)
    if moire:
        texture=torch.zeros_like(screen)
        for low,high in ((2,8),(8,40),(40,160)):
            period=rng.uniform(low,high);angle,phase=rng.uniform(0,2*np.pi,2)
            texture+=(torch.sin((x*np.cos(angle)+y*np.sin(angle))*(2*np.pi/period)+phase)+torch.sin((-x*np.sin(angle)+y*np.cos(angle))*(2*np.pi/rng.uniform(low,high))+phase))/6
        a+=texture[None]*float(rng.uniform(0,amp))*screen
    camera=str(rng.choice(['none','blur','jpeg','noise'],p=[.4,.2,.2,.2]));jpeg=None
    if camera=='noise':
        g=torch.Generator(device=photo.device).manual_seed(seed);a+=torch.randn(a.shape,device=a.device,generator=g)*float(rng.uniform(0,noise)/255)
    a=(a*255).clamp(0,255).to(torch.uint8)
    if camera=='blur':
        sigma=float(rng.uniform(.01,blur));radius=max(1,int(np.ceil(sigma*3)));axis=torch.arange(-radius,radius+1,device=a.device).float();kernel=torch.exp(-axis.square()/(2*sigma*sigma));kernel/=kernel.sum()
        rgb=a[None].float();rgb=F.pad(rgb,(radius,radius,radius,radius),mode='reflect');k=(kernel[:,None]*kernel[None,:]).expand(3,1,-1,-1)
        a=F.conv2d(rgb,k,groups=3)[0].round().clamp(0,255).to(torch.uint8)
    elif camera=='jpeg':jpeg=int(rng.integers(quality,101))
    return a,{'camera':camera,'moire':moire,'gamma':gamma},jpeg

class PendingView:
    def __init__(self,plan,photo,mask,screen,event,jpeg,app):self.plan,self.photo,self.mask,self.screen,self.event,self.jpeg,self.app=plan,photo,mask,screen,event,jpeg,app
    def resolve(self):
        if self.event is not None:self.event.synchronize()
        p=self.plan;photo=self.photo.numpy();mask=self.mask.numpy();screen=self.screen.numpy()
        if self.jpeg is not None:
            ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(photo,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,self.jpeg]);assert ok
            photo=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        x,y,r,b=p.crop;shift=np.array([[1,0,-x],[0,1,-y],[0,0,1.]])
        q=project(p.quad,shift);transform=shift@p.transform
        info={**p.info,**self.app,**visibility(q,transform,(p.source.shape[1],p.source.shape[0]),(r-x,b-y)),
              'occupancy':float(mask[y:b,x:r].mean()),'source_to_photo':transform.tolist(),'post_photo_crop':list(p.crop)}
        if info['kind']=='photo' and info['occupancy']<(.5 if info['requested_high_occupancy'] else .3):raise RuntimeError('Visible occupancy quota failed')
        return PartialView(photo[y:b,x:r].copy(),q,mask[y:b,x:r].copy(),screen[y:b,x:r].copy(),transform,p.ratio,info)

def render(plans,device='cuda'):
    """Returns event-protected pinned host buffers; CPU writers wait only on their item."""
    if not plans:return []
    out=[None]*len(plans);active=[(i,p) for i,p in enumerate(plans) if p.info['kind']!='keep']
    for i,p in enumerate(plans):
        if p.info['kind']=='keep':out[i]=PendingView(p,torch.from_numpy(p.source.copy()),torch.ones(p.source.shape[:2]),torch.ones(p.source.shape[:2]),None,None,{})
    if not active:return out
    warped=warp_batch([p for _,p in active],device)
    for j,(i,p) in enumerate(active):
        h,w=p.background.shape[:2];mask=warped[j,3,:h,:w];bg=torch.as_tensor(p.background,device=device).permute(2,0,1).float()
        screen=torch.as_tensor(p.screen,device=device);photo=(warped[j,:3,:h,:w]+bg*(1-mask)).clamp(0,255).to(torch.uint8).float()
        rgb,app,jpeg=appearance_gpu(photo,screen,p.seed,p.level);rgb=rgb.permute(1,2,0).contiguous()
        arrays=[]
        for value in (rgb,mask.contiguous(),screen):
            host=torch.empty_like(value,device='cpu',pin_memory=device=='cuda');host.copy_(value,non_blocking=device=='cuda');arrays.append(host)
        event=None
        if device=='cuda':event=torch.cuda.Event();event.record()
        out[i]=PendingView(p,*arrays,event,jpeg,app)
    return out
