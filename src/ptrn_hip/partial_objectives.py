"""Visible-area supervision without a blanket out-of-photo corner penalty."""
import torch
from torch.nn import functional as F
from .spatial_model import coordinate_loss
from .spatial_objectives import _soft_quad_mask, _invalid_quad_penalty, ratio_residual_loss


def visible_coverage(pred_quad, truth_quad, photo_valid, size):
    if photo_valid.shape != (len(pred_quad), size, size) or not torch.isfinite(photo_valid).all():
        raise ValueError('Expected finite photo-valid mask [N,size,size]')
    if (photo_valid < 0).any() or (photo_valid > 1).any():
        raise ValueError('Invalid photo support')
    predicted = _soft_quad_mask(pred_quad.float(), size, size, 1.5)*photo_valid
    truth = _soft_quad_mask(truth_quad.float(), size, size, 1.5)*photo_valid
    mass = truth.sum((1, 2)).clamp_min(1)
    missed = (F.relu(truth-predicted).sum((1, 2))/mass).mean()
    background = (F.relu(predicted-truth).sum((1, 2))/mass).mean()
    return missed, background


def partial_loss(output, batch, size, *, coverage_weight=1., mask_weight=1., ratio_weight=.25):
    quad, truth = output['quad'].float(), batch['quad'].float()
    coord = coordinate_loss(quad, truth, size)
    missed, background = visible_coverage(quad, truth, batch['photo_valid'].float(), size)
    invalid = _invalid_quad_penalty(quad, size, size)
    target = F.interpolate(batch['mask'].float()[:, None], size=output['mask_logits'].shape[-2:], mode='area')
    mask = F.binary_cross_entropy_with_logits(output['mask_logits'].float(), target)
    ratio = ratio_residual_loss(output['log_ratio'], batch['log_ratio'])
    total = coord+coverage_weight*(3*missed+background+invalid)+mask_weight*mask+ratio_weight*ratio
    if not torch.isfinite(total): raise FloatingPointError('Nonfinite V3 loss; no sample skipped')
    return {'total': total, 'coordinate': coord, 'missed_visible': missed,
            'extra_background': background, 'invalid': invalid, 'mask': mask, 'ratio': ratio}
