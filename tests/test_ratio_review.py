import numpy as np
import pytest
from ptrn_hip.ratio_review import rectify_canvas
from ptrn_hip.ratio_synthesis import project


def test_learned_canvas_ratio_and_source_axes_are_explicit():
    source=np.zeros((120,180,3),np.uint8)
    source[:,:90,0]=240;source[:,90:,2]=240
    q=np.array([[0.,0.],[179.,0.],[179.,119.],[0.,119.]])
    result,h=rectify_canvas(source,q,2.)
    assert result.shape[:2]==(120,240)
    np.testing.assert_allclose(project(q,h),[[0,0],[239,0],[239,119],[0,119]],atol=1e-9)
    assert result[60,40,0]>200 and result[60,200,2]>200


def test_legacy_canvas_matches_maximum_edge_size():
    source=np.zeros((150,200,3),np.uint8)
    q=np.array([[10.,10.],[180.,10.],[190.,120.],[10.,140.]])
    sides=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
    result,_=rectify_canvas(source,q)
    assert result.shape[1]==round(max(sides[0],sides[2]))+1
    assert result.shape[0]==round(max(sides[1],sides[3]))+1


def test_invalid_quad_and_ratio_are_not_silently_rectified():
    source=np.zeros((120,180,3),np.uint8)
    q=np.array([[0.,0.],[179.,0.],[179.,119.],[0.,119.]])
    with pytest.raises(ValueError):rectify_canvas(source,q,float('nan'))
    with pytest.raises(ValueError):rectify_canvas(source,q[::-1],1.5)
