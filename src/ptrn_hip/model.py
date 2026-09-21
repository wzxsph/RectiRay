"""Official PTRN architecture, with strict loading and explicit fine-tuning."""
from __future__ import annotations

import os
import math
from pathlib import Path

os.environ["KERAS_BACKEND"] = "torch"
os.environ.setdefault("OMP_NUM_THREADS", "4")

import keras
import numpy as np
import torch


def configure(mixed: bool = False) -> dict:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    devices = torch.cuda.device_count()
    keras.mixed_precision.set_global_policy("mixed_float16" if mixed and devices else "float32")
    keras.utils.set_random_seed(20260921)
    if devices:
        torch.cuda.reset_peak_memory_stats()
    return {"backend": "torch", "torch": torch.__version__, "keras": keras.__version__, "gpu_count": devices,
            "gpu": torch.cuda.get_device_name() if devices else None, "mixed_precision": bool(mixed and devices)}


def build(checkpoint: Path):
    backbone = keras.applications.DenseNet201(include_top=False, weights=None, pooling="avg")
    inputs = keras.Input((224, 224, 3), name="photo")
    output = keras.layers.Dense(8, activation="linear", dtype="float32", name="homography")(backbone(inputs))
    model = keras.Model(inputs, output, name="ptrn_hip")
    model.load_weights(checkpoint)
    return model, backbone


def set_stage(model, backbone, stage: str, learning_rate: float) -> int:
    if stage not in {"head", "last_block"}:
        raise ValueError(stage)
    backbone.trainable = stage == "last_block"
    if backbone.trainable:
        for layer in backbone.layers:
            layer.trainable = layer.name.startswith("conv5_") and not isinstance(layer, keras.layers.BatchNormalization)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0), loss="mse", jit_compile=False)
    return sum(math.prod(v.shape) for v in model.trainable_variables)


def preprocess(images):
    return keras.applications.densenet.preprocess_input(np.asarray(images, np.float32).copy())


def infer(model, images, batch_size=16):
    with torch.no_grad():
        return np.concatenate([keras.ops.convert_to_numpy(model(preprocess(images[i:i + batch_size]), training=False)) for i in range(0, len(images), batch_size)])
