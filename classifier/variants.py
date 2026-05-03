"""Детерминистская аугментация для тренировки.

6 фиксированных вариантов на сэмпл: hflip × {neutral, brighter, darker}.
Применяется к уже препроцессированному (через `core.leaf_pipeline.transform_leaf`)
RGB-кропу --- никаких поворотов или PCA здесь нет, только зеркало + цветовой jitter.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image


class Variant(NamedTuple):
    hflip: bool
    brightness: float
    contrast: float
    saturation: float


IDENTITY_VARIANT = Variant(False, 1.0, 1.0, 1.0)

VARIANTS: list[Variant] = [
    Variant(False, 1.0, 1.0,  1.0),   # orig, нейтральный
    Variant(False, 1.2, 1.15, 0.9),   # orig, ярче
    Variant(False, 0.8, 0.85, 1.1),   # orig, темнее
    Variant(True,  1.0, 1.0,  1.0),   # hflip, нейтральный
    Variant(True,  1.2, 1.15, 0.9),   # hflip, ярче
    Variant(True,  0.8, 0.85, 1.1),   # hflip, темнее
]


def apply_variant(img: np.ndarray, v: Variant) -> np.ndarray:
    """hflip + brightness/contrast/saturation на RGB."""
    if v == IDENTITY_VARIANT:
        return img
    pil = Image.fromarray(img)
    if v.hflip:
        pil = TF.hflip(pil)
    if v.brightness != 1.0:
        pil = TF.adjust_brightness(pil, v.brightness)
    if v.contrast != 1.0:
        pil = TF.adjust_contrast(pil, v.contrast)
    if v.saturation != 1.0:
        pil = TF.adjust_saturation(pil, v.saturation)
    return np.array(pil)
