import numpy as np
import cv2
import pytest
import torch
from ptrn_pipeline.gpu_render import plan,render,warp_batch

def source():
 y,x=np.mgrid[:72,:96];return np.stack((x*2,y*3,(x+y)%255),-1).astype(np.uint8)

def test_keep_is_pixel_exact():
 a=source();p=plan(a,0,0,kind='keep');v=render([p],device='cpu')[0].resolve()
 assert np.array_equal(v.photo,a) and v.info['missing_fraction']==0

@pytest.mark.parametrize('integrity',['complete','light','moderate'])
def test_missing_plane_ratio_and_mask(integrity):
 p=plan(source(),0,2,seed=12,integrity=integrity,family='near_color',level='clear',source_crop=True)
 v=render([p],device='cpu')[0].resolve()
 assert v.ratio==p.source.shape[1]/p.source.shape[0]
 assert v.info['occupancy']>=.5
 missing=v.info['missing_fraction']
 if integrity=='complete':assert missing<1e-5
 elif integrity=='light':assert .012<=missing<=.098
 else:assert .112<=missing<=.288

def test_warp_matches_opencv_on_clean_geometry():
 p=plan(source(),0,3,seed=2,family='near_color',level='clear')
 gpu=warp_batch([p],device='cpu')[0].numpy();h,w=p.background.shape[:2]
 old=cv2.warpPerspective(p.source,p.transform,(w,h),flags=cv2.INTER_LINEAR)
 assert np.abs(gpu[:3].transpose(1,2,0)-old).mean()<1
 mask=cv2.warpPerspective(np.ones(p.source.shape[:2],np.float32),p.transform,(w,h),flags=cv2.INTER_LINEAR)
 assert np.abs(mask-gpu[3]).mean()<.001

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA smoke requires GPU')
def test_cuda_async_buffers_and_multi_family():
 families=['viewer_screen','coco_context','near_color','mixed_color'];a=source()
 ps=[plan(a,0,i+1,seed=4,family=f,background=np.full((100,120,3),127,np.uint8),level='light') for i,f in enumerate(families)]
 pending=render(ps,'cuda');views=[x.resolve() for x in pending]
 assert all(v.photo.dtype==np.uint8 and v.info['occupancy']>=.5 for v in views)
 assert all(x.event is not None for x in pending)
