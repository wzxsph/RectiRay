"""V3 research candidate: shared spatial decoder, unbounded quad, visible mask.

The mask is auxiliary supervision, not a hard crop or an automatic approval gate.
The ratio module remains an explicit ablation. No old H8/heatmap checkpoint is
accepted as a V3 checkpoint.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .spatial_model import build_spatial, parameter_groups
from .spatial_objectives import RatioResidual
from .partial_geometry import CONTRACT


class PartialModel(nn.Module):
    def __init__(self, spatial, feature_dim=1920):
        super().__init__()
        if spatial.arm != 'spatial_quad':
            raise ValueError('V3 requires unbounded spatial_quad readout')
        self.spatial, self.size = spatial, spatial.size
        self.mask_head = nn.Conv2d(4, 1, 1)
        self.ratio_head = RatioResidual(feature_dim=feature_dim, size=self.size)

    def forward(self, normalized_nhwc):
        features = self.spatial.extract_features(normalized_nhwc)
        logits = self.spatial._spatial_logits(features).float()
        quad = self.spatial.direct(F.adaptive_avg_pool2d(logits, (56, 56)).flatten(1)).reshape(-1, 4, 2)*(self.size-1)
        pooled = features[-1].float().mean((2, 3))
        # Clamped edge LENGTHS only avoid log(0); predicted coordinates are never clipped.
        edges = torch.linalg.vector_norm(torch.roll(quad.detach(), -1, dims=1)-quad.detach(), dim=-1).clamp_min(1e-4)
        baseline = edges[:, [0, 2]].amax(1).log()-edges[:, [1, 3]].amax(1).log()
        return {'quad': quad.float(), 'log_ratio': baseline+self.ratio_head(quad, pooled),
                'log_ratio_baseline': baseline, 'mask_logits': self.mask_head(logits).float()}

    def optimizer_groups(self, stage):
        groups = parameter_groups(self.spatial, stage)
        groups.append({'params': list(self.mask_head.parameters())+list(self.ratio_head.parameters()),
                       'lr': 1e-4, 'name': 'visible_region_and_ratio'})
        return groups

    def contract(self):
        return {'contract': CONTRACT, 'input_size': self.size, 'quad': 'TL/TR/BR/BL input pixel centers, unbounded',
                'ratio': 'full reference plane width/height, not visible output bounding box',
                'mask': 'auxiliary visible film probability on letterbox input',
                'automatic_approval': False}


def build_partial(weights, g0_checkpoint, *, size=224, stage='head'):
    return PartialModel(build_spatial(weights, g0_checkpoint, 'spatial_quad', size=size, stage=stage))


class RawPartial(nn.Module):
    """Explicit V3 CPU export interface: raw RGB -> quad, log ratio, visible mask."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor([.485, .456, .406]))
        self.register_buffer('std', torch.tensor([.229, .224, .225]))

    def forward(self, rgb_nhwc):
        out = self.model((rgb_nhwc.float()/255-self.mean)/self.std)
        return out['quad'], out['log_ratio'], out['mask_logits'].sigmoid()
