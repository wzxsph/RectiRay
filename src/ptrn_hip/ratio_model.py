"""Strict B1.1 initialization, center-span corner loss, frozen BN training."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from .model import keras, torch, build
from .ratio_data import sha256

B11_SHA = '89d48a8c88200aa3c6916f2839556615138d6c5b1c7b4f803cd8f78ce8cc0520'


def build_ratio(checkpoint, size, with_ratio, mean_log_ratio):
    if sha256(checkpoint) != B11_SHA:
        raise ValueError('B1.1 initialization SHA mismatch')
    legacy, backbone = build(Path(checkpoint))
    inputs = keras.Input((size,size,3),name='photo_ratio_v1')
    features = backbone(inputs,training=False)
    h = legacy.layers[-1](features)
    if with_ratio:
        ratio = keras.layers.Dense(1,kernel_initializer='zeros',bias_initializer=keras.initializers.Constant(mean_log_ratio),dtype='float32',name='log_canvas_ratio')(features)
        h = keras.layers.Concatenate(name='h8_logratio_v1')([h,ratio])
    model = keras.Model(inputs,h,name='ptrn_ratio_v1')
    return model,backbone


def stage(model, backbone, name):
    backbone.trainable = name == 'last_block'
    if backbone.trainable:
        for layer in backbone.layers:
            layer.trainable = layer.name.startswith('conv5_') and not isinstance(layer,keras.layers.BatchNormalization)
    parameters = [v.value for v in model.trainable_weights]
    if not parameters:
        raise ValueError('No trainable parameters')
    return parameters


def tensor_input(pixels,device):
    x = torch.as_tensor(np.asarray(pixels).copy(),device=device,dtype=torch.float32)/255
    return (x-torch.tensor([.485,.456,.406],device=device))/torch.tensor([.229,.224,.225],device=device)


def corner_tensor(prediction,size):
    prediction = prediction.float()
    matrix = torch.cat([prediction[:,:8],torch.ones_like(prediction[:,:1])],dim=1).reshape(-1,3,3)
    corners = torch.tensor([[-1.,-1.,1.],[1.,-1.,1.],[1.,1.,1.],[-1.,1.,1.]],device=prediction.device)
    q = torch.matmul(corners,matrix.transpose(1,2))
    # Fail explicitly on singular horizons, including mixed-sign corner depths.
    denominator = q[:,:,2:]
    if not torch.isfinite(q).all() or (denominator<=1e-7).any():
        raise FloatingPointError('Non-finite/singular/behind-plane predicted projective denominator')
    return (q[:,:,:2]/denominator+1)*(size-1)/2


def losses(prediction,quad,log_ratio,size):
    predicted = corner_tensor(prediction,size)
    quad_loss = torch.nn.functional.smooth_l1_loss((predicted-quad.float())/(.01*math.hypot(size,size)),torch.zeros_like(predicted),beta=1.)
    ratio_loss = prediction.new_zeros((),dtype=torch.float32)
    if prediction.shape[1] == 9:
        error = (prediction[:,8].float()-log_ratio.float())/.05
        ratio_loss = torch.nn.functional.smooth_l1_loss(error,torch.zeros_like(error),beta=1.)
    loss = quad_loss+.25*ratio_loss
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite training loss; no batch skipped')
    return loss,quad_loss,ratio_loss


def predict(model,pixels,device):
    with torch.no_grad():
        output = model(tensor_input(pixels,device),training=False).float()
    if not torch.isfinite(output).all():
        raise FloatingPointError('Non-finite evaluation output')
    return output.detach().cpu().numpy()


def contract(size,with_ratio):
    return {'contract':f'ptrn-ratio-v1-rgb{size}-'+('h8-logratio9' if with_ratio else 'h8'),
            'input_shape':[1,size,size,3],'input_dtype':'float32','input_range':'RGB 0..255',
            'output_shape':[1,9 if with_ratio else 8],'homography_direction':'source_normalized_TL_TR_BR_BL_to_input_normalized',
            'homography_layout':'row_major_first8_h33_equals1','normalization':'[-1,1] maps pixel centers 0..size-1',
            'ratio':'exp(output[8]) equals original EXIF full canvas W/H; not physical anatomy' if with_ratio else None,
            'padding':'isotropic center-span letterbox; black; OpenCV warpAffine INTER_LINEAR',
            'source_corners':'(0,0),(W-1,0),(W-1,H-1),(0,H-1); boundaries +/-0.5',
            'rectification':'invert predicted H after undoing input letterbox; destination W/H set separately',
            'ratio_evidence':'model_estimated','independent_real_ratio_validated':False,
            'b11_legacy_normalization':'B1.1 uses size, not size-1; evaluated with original convention and direct square resize'}
