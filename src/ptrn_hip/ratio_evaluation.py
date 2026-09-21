"""Fair common-photo evaluation; source-image dev is never independent validation."""
from __future__ import annotations
from collections import defaultdict
import cv2
import numpy as np
from .ratio_synthesis import UNIT, project, normalization, MODES


def predicted_quad(parameters,size,legacy=False):
    h = np.r_[parameters[:8],1.].reshape(3,3)
    q = project(UNIT,h)
    return (q+1)*(size if legacy else size-1)/2


def edge_ratio(q):
    lengths = np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
    return max(lengths[0],lengths[2])/max(lengths[1],lengths[3],1e-12)


def measure(prediction,truth_quad,log_ratio,size,legacy=False,photo_size=None):
    try:
        q = predicted_quad(prediction,size,legacy)
        # B1.1 square-distorts its input. Undo this before estimating side lengths.
        q_photo = (q+.5)*(np.array(photo_size)/size)-.5 if legacy else q
        truth_ratio = np.exp(log_ratio)
        edges = edge_ratio(q_photo)
        learned = np.exp(prediction[8]) if len(prediction)==9 else edges
        quad32 = q.astype(np.float32)
        denominator = np.c_[UNIT,np.ones(4)] @ np.r_[prediction[:8],1.].reshape(3,3)[2]
        valid = bool(cv2.isContourConvex(quad32) and cv2.contourArea(quad32,oriented=True)>0 and
                     (denominator>0).all() and (q >= 0).all() and (q <= size-1).all())
        distance = np.linalg.norm(q-truth_quad,axis=1)
        corners = float(np.mean(distance)/(size*np.sqrt(2)))
        # Invalid predictions remain in error aggregates; they are never dropped.
        ratio_error = float(np.expm1(abs(np.log(learned/truth_ratio))))
        edge_error = float(np.expm1(abs(np.log(edges/truth_ratio))))
        # Source-axis ratio after rectification: corner and ratio sources separated.
        def shape_error(rectify_quad,out_ratio):
            target = np.array([[0,0],[out_ratio,0],[out_ratio,1],[0,1]],np.float32)
            matrix = cv2.getPerspectiveTransform(rectify_quad.astype(np.float32),target)
            restored = project(truth_quad,matrix)
            observed = edge_ratio(restored)
            return float(np.expm1(abs(np.log(observed/truth_ratio))))
        return {'corner_nme':corners,'invalid':int(not valid),'ratio_error':ratio_error,'edge_ratio_error':edge_error,
                'gt_corners_pred_ratio_error':ratio_error,
                'pred_corners_gt_ratio_error':shape_error(q,truth_ratio),
                'pred_corners_pred_ratio_error':shape_error(q,learned)}
    except (ValueError,cv2.error,FloatingPointError,OverflowError):
        return {'corner_nme':1e6,'invalid':1,'ratio_error':1e6,'edge_ratio_error':1e6,
                'gt_corners_pred_ratio_error':1e6,'pred_corners_gt_ratio_error':1e6,'pred_corners_pred_ratio_error':1e6}


def summarize(rows):
    result = {'count':len(rows),'invalid':sum(r['invalid'] for r in rows)}
    for key in ('corner_nme','ratio_error','edge_ratio_error','gt_corners_pred_ratio_error','pred_corners_gt_ratio_error','pred_corners_pred_ratio_error'):
        values = np.array([r[key] for r in rows])
        if not np.isfinite(values).all():
            raise FloatingPointError('Non-finite evaluation metric')
        result[key] = {'mean':float(values.mean()),'median':float(np.median(values)),'p90':float(np.percentile(values,90))}
    return result


def select(evaluations):
    """One deterministic checkpoint choice per group; then frozen acceptance gates."""
    best = {}
    for group,checkpoints in evaluations.items():
        best[group] = min(checkpoints,key=lambda row:(row['clean']['all']['corner_nme']['mean'] if group=='G0' else row['clean']['mild']['ratio_error']['median'],row['step']))
    reference = best['G0']
    gates = {}
    for group in ('G1','G2'):
        candidate = best[group]
        corner_ok = all(candidate[condition][mode]['corner_nme']['mean'] <= reference[condition][mode]['corner_nme']['mean']*1.05 and
                        candidate[condition][mode]['corner_nme']['p90'] <= reference[condition][mode]['corner_nme']['p90']*1.10 and
                        candidate[condition][mode]['invalid'] <= reference[condition][mode]['invalid']
                        for condition in ('clean','noisy') for mode in MODES)
        mild = candidate['clean']['mild']
        ratio_ok = mild['ratio_error']['median'] <= .03 and mild['ratio_error']['p90'] <= .10 and mild['ratio_error']['median'] <= .8*mild['edge_ratio_error']['median']
        gates[group] = {'corner_gate':corner_ok,'ratio_gate':ratio_ok,'eligible':corner_ok and ratio_ok}
    chosen = 'G1' if gates['G1']['eligible'] else None
    if gates['G2']['eligible'] and best['G2']['clean']['mild']['ratio_error']['p90'] <= .9*best['G1']['clean']['mild']['ratio_error']['p90']:
        # CPU acceptance has no numeric threshold in training protocol; main-task review is required.
        gates['G2']['additional_p90_improvement'] = True
        gates['G2']['cpu_acceptance_pending'] = True
    return {'best_steps':{g:r['step'] for g,r in best.items()},'gates':gates,'provisional_candidate':chosen,
            'active_model':'B1.1','replacement_performed':False,'independent_real_ratio_validated':False,
            'G2_priority_requires_CPU_acceptance':True,
            'selection_rule':'G0 minimum clean all corner mean; G1/G2 minimum clean mild ratio median; tie earlier step'}
