"""Frozen DenseNet prefix cache for spatial stage ``last_block`` research.

The cache cuts at ``pool4_pool``. Keras prefix/suffix Functional views share
the original encoder variables; only the frozen prefix runs once per view.
The suffix is always called with ``training=False`` to preserve BatchNorm
statistics while allowing conv5 gradients, matching SpatialModel.forward.
"""
from __future__ import annotations

import torch

from .model import keras
from .spatial_model import TAPS


class FrozenPrefixCache:
    """Cache normalized NHWC images' frozen taps and pool4 tensors.

    ``forward(indices)`` returns corners [batch,4,2] with gradients through
    the original conv5 and head parameters. The cache is valid only while the
    prefix remains frozen and the input views remain unchanged.
    """

    def __init__(self, model, pixels_normalized: torch.Tensor, batch_size: int = 8):
        if getattr(model, 'stage', None) != 'last_block':
            raise ValueError('FrozenPrefixCache_requires_last_block_stage')
        if batch_size < 1:
            raise ValueError('batch_size_must_be_positive')
        if (pixels_normalized.ndim != 4 or tuple(pixels_normalized.shape[1:]) !=
                (model.size, model.size, 3) or len(pixels_normalized) < 1):
            raise ValueError('expected_nonempty_normalized_NHWC_views')
        if not torch.isfinite(pixels_normalized).all():
            raise ValueError('nonfinite_normalized_views')
        self.model = model
        self.size = len(pixels_normalized)
        self.spatial = model.arm.startswith('spatial_')
        encoder = model.encoder
        pool = encoder.get_layer('pool4_pool').output
        prefix_outputs = ([encoder.get_layer(name).output for name in TAPS[:3]] + [pool]
                          if self.spatial else pool)
        tail = (encoder.get_layer(TAPS[3]).output if self.spatial else encoder.output)
        self.prefix = keras.Model(encoder.input, prefix_outputs, name='frozen_prefix_view')
        self.suffix = keras.Model(pool, tail, name='trainable_conv5_suffix_view')
        if self.prefix.trainable_weights:
            raise ValueError('prefix_has_trainable_weights')
        if not self.suffix.trainable_weights:
            raise ValueError('suffix_has_no_trainable_conv5_weights')
        # Building views must never mutate the original model's trainability.
        if not encoder.get_layer(TAPS[3]).trainable:
            raise ValueError('conv5_must_remain_trainable')
        chunks = []
        with torch.no_grad():
            for offset in range(0, self.size, batch_size):
                chunks.append(self.prefix(pixels_normalized[offset:offset + batch_size], training=False))
        if self.spatial:
            self.taps = tuple(torch.cat([part[i].detach() for part in chunks], dim=0)
                              .permute(0, 3, 1, 2).contiguous() for i in range(3))
            self.pool4 = torch.cat([part[3].detach() for part in chunks], dim=0).contiguous()
        else:
            self.taps = ()
            self.pool4 = torch.cat([part.detach() for part in chunks], dim=0).contiguous()
        if self.pool4.requires_grad or any(t.requires_grad for t in self.taps):
            raise AssertionError('frozen_prefix_cache_retained_autograd_graph')

    def _index(self, indices):
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.pool4.device).reshape(-1)
        if len(idx) < 1 or bool((idx < 0).any()) or bool((idx >= self.size).any()):
            raise IndexError('cached_view_index_out_of_range')
        return idx

    def features_for(self, indices):
        """Return the same feature structure as SpatialModel.extract_features."""
        idx = self._index(indices)
        conv5 = self.suffix(self.pool4.index_select(0, idx), training=False)
        if self.spatial:
            return tuple(t.index_select(0, idx) for t in self.taps) + (
                conv5.permute(0, 3, 1, 2).contiguous(),)
        return conv5

    def forward(self, indices):
        return self.model.forward_features(self.features_for(indices))
