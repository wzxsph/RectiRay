import cv2
import numpy as np
import pytest
from ptrn_hip.ratio_synthesis import (sample,appearance,stream,letterbox,project,normalization,pixel_corners,UNIT,LEVELS,MODES)
from ptrn_hip.ratio_evaluation import measure,edge_ratio


def fixture():
    image=np.zeros((125,203,3),np.uint8)
    image[12:-12,10:-10]=100
    image[25:105,15:95,0]=220
    image[25:105,115:190,2]=220
    return image


def test_original_ratio_center_span_roundtrip_identity_not_unit():
    source=fixture()
    view=sample(source,3,7,mode='identity')
    assert view.log_ratio==np.log(203/125)
    assert view.log_ratio!=np.log(202/124)
    for size in (224,384):
        image,label,quad=view.input(size)
        h=np.r_[label[:8],1.].reshape(3,3)
        np.testing.assert_allclose(project(UNIT,h),(quad*2/(size-1)-1),atol=2e-7)
        assert not np.allclose(h,np.eye(3))
        np.testing.assert_allclose(project(project(UNIT,h),np.linalg.inv(h)),UNIT,atol=1e-12)
        # Recover the colored left/right canvas using the actual inverse pixel transform.
        destination=pixel_corners(203,125).astype(np.float32)
        inv=cv2.getPerspectiveTransform(quad.astype(np.float32),destination)
        restored=cv2.warpPerspective(image,inv,(203,125))
        assert restored[60,50,0]>restored[60,50,2]+50
        assert restored[60,150,2]>restored[60,150,0]+50


@pytest.mark.parametrize('mode',MODES)
def test_same_common_photo_resolution_and_label_invariance(mode):
    views=[sample(fixture(),13,66,mode=mode,level=level) for level in LEVELS]
    for view in views:
        np.testing.assert_array_equal(view.clean,views[0].clean)
        np.testing.assert_array_equal(view.quad,views[0].quad)
        assert view.log_ratio==views[0].log_ratio
        assert max(view.clean.shape[:2])==768
        for size in (224,384):
            noisy,label,q=view.input(size)
            clean,clean_label,clean_q=view.input(size,noisy=False)
            np.testing.assert_array_equal(label,clean_label)
            np.testing.assert_array_equal(q,clean_q)
        if mode=='hard' and view.info['requested_appearance']=='strong':
            assert view.info['appearance']=='light'
    a=sample(fixture(),13,66,mode=mode,level='light')
    np.testing.assert_array_equal(a.noisy,views[1].noisy)


def test_screen_texture_changes_viewer_not_only_film():
    image=np.full((160,240,3),120,np.uint8)
    mask=np.zeros((160,240),np.float32);mask[10:150,10:230]=1
    # Force deterministic seed with moire and camera none to isolate masking.
    for seed in range(100):
        result,info=appearance(image,mask,np.random.default_rng(seed),'strong')
        if info['moire'] and info['camera']=='none':break
    assert info['moire'] and info['camera']=='none'
    np.testing.assert_array_equal(result[:10],image[:10])
    assert np.any(result[10:30,10:230]!=image[10:30,10:230])
    assert np.any(result[50:100,50:180]!=image[50:100,50:180])


def test_letterbox_is_isotropic_and_center_exact():
    image=np.zeros((107,303,3),np.uint8)
    _,h=letterbox(image,384)
    assert h[0,0]==h[1,1]
    np.testing.assert_allclose(project([[151,53]],h),[[191.5,191.5]])


def test_metrics_separate_ratio_and_corner_sources():
    v=sample(fixture(),2,8,mode='mild')
    _,label,quad=v.input(224)
    exact=measure(label,quad,label[8],224)
    assert exact['ratio_error']==pytest.approx(0,abs=1e-7)
    assert exact['pred_corners_gt_ratio_error']==pytest.approx(0,abs=1e-6)
    wrong=label.copy();wrong[8]+=np.log(1.2)
    error=measure(wrong,quad,label[8],224)
    assert error['ratio_error']==pytest.approx(.2,abs=1e-6)
    assert error['corner_nme']<1e-6
    assert error['pred_corners_gt_ratio_error']<1e-6
    assert error['gt_corners_pred_ratio_error']==pytest.approx(.2,abs=1e-6)


def test_corner_loss_accumulation_matches_full_batch_and_stops_nonfinite():
    from ptrn_hip.ratio_model import torch,losses
    torch.manual_seed(4)
    head=torch.nn.Linear(6,9)
    with torch.no_grad():
        head.weight.mul_(.01)
        head.bias.copy_(torch.tensor([.65,0,0,0,.65,0,0,0,.2]))
    x=torch.randn(64,6)
    quad=torch.tensor([[[40.,40.],[180.,40.],[180.,180.],[40.,180.]]]).repeat(64,1,1)
    target=torch.full((64,),.1)
    full=losses(head(x),quad,target,224)[0];full.backward()
    expected=[p.grad.clone() for p in head.parameters()]
    head.zero_grad()
    for start in range(0,64,16):
        (losses(head(x[start:start+16]),quad[start:start+16],target[start:start+16],224)[0]/4).backward()
    for actual,want in zip(head.parameters(),expected,strict=True):
        torch.testing.assert_close(actual.grad,want,rtol=2e-5,atol=2e-6)
    broken=head(x).detach();broken[0,6]=1.;broken[0,7]=0.
    with pytest.raises(FloatingPointError):losses(broken,quad,target,224)
    broken=head(x).detach();broken[0,0]=float('nan')
    with pytest.raises(FloatingPointError):losses(broken,quad,target,224)


def test_data_order_views_unchanged_by_resolution_and_resume(tmp_path):
    import json
    from PIL import Image
    from ptrn_hip.ratio_runner import TrainingSet
    path=tmp_path/'fixture.png';Image.fromarray(fixture()).save(path)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'records':[{'source_path':str(path)} for _ in range(3)]}))
    a=TrainingSet(manifest,[0,1,2],224,0,100)
    b=TrainingSet(manifest,[0,1,2],384,0,100)
    resumed=TrainingSet(manifest,[0,1,2],224,6,100)
    for i in (0,2,6,7):
        xa,qa,ra,ia=a[i];xb,qb,rb,ib=b[i]
        assert ra==rb
        np.testing.assert_array_equal(ia,ib)
        np.testing.assert_allclose(qa/223,qb/383,atol=1e-7)
        if i>=6:
            for actual,want in zip(resumed[i-6],a[i],strict=True):
                np.testing.assert_array_equal(actual,want)


def test_checkpoint_optimizer_restores_exact_next_update(tmp_path,monkeypatch):
    from ptrn_hip.ratio_model import torch
    import ptrn_hip.ratio_runner as runner
    monkeypatch.setattr(runner,'check_disk',lambda _:100.)
    torch.manual_seed(8)
    model=torch.nn.Linear(3,2)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
    x=torch.randn(5,3)
    def update(m,o):
        o.zero_grad();m(x).square().mean().backward();o.step()
    update(model,optimizer)
    # CPU-only unit test must not require a CUDA device for RNG capture.
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    runner.save_checkpoint(model,optimizer,tmp_path/'step-0250.pt',250,'last_block',{'source_sha':'test'}, {})
    update(model,optimizer)
    restored=torch.nn.Linear(3,2);opt=torch.optim.Adam(restored.parameters(),lr=1e-5)
    saved=torch.load(tmp_path/'step-0250.pt',weights_only=False)
    restored.load_state_dict(saved['state_dict']);opt.load_state_dict(saved['optimizer'])
    update(restored,opt)
    for want,actual in zip(model.parameters(),restored.parameters(),strict=True):
        torch.testing.assert_close(actual,want,rtol=0,atol=0)


def test_ratio_baseline_uses_current_application_maximum_edges():
    quad=np.array([[0.,0.],[100.,0.],[140.,50.],[0.,60.]])
    edges=np.linalg.norm(np.roll(quad,-1,axis=0)-quad,axis=1)
    assert edge_ratio(quad)==pytest.approx(max(edges[0],edges[2])/max(edges[1],edges[3]))
    assert edge_ratio(quad)!=pytest.approx((edges[0]+edges[2])/(edges[1]+edges[3]))


def test_mixed_sign_projective_horizon_fails_explicitly():
    from ptrn_hip.ratio_model import torch,corner_tensor
    with pytest.raises(FloatingPointError,match='behind-plane'):
        corner_tensor(torch.tensor([[1.,0,0,0,1,0,2,0]]),224)


def test_legacy_gt_resize_uses_actual_pixel_centers():
    source_center=np.array([(768-1)/2,(600-1)/2])
    resized=(source_center+.5)*[224/768,224/600]-.5
    np.testing.assert_allclose(resized,[111.5,111.5],atol=1e-12)
    restored=(resized+.5)*[768/224,600/224]-.5
    np.testing.assert_array_equal(restored,source_center)
