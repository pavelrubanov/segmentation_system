"""Детерминистская аугментация для тренировки.

6 фиксированных вариантов на сэмпл: hflip × {neutral, brighter, darker}.
Применяется к уже препроцессированному (через `core.leaf_pipeline.transform_leaf`)
RGB-кропу --- никаких поворотов или PCA здесь нет, только зеркало + цветовой jitter.
"""
from __future__ import annotations

import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image


# Variant = (hflip, brightness, contrast, saturation).
Variant = tuple[bool, float, float, float]

IDENTITY_VARIANT: Variant = (False, 1.0, 1.0, 1.0)

VARIANTS: list[Variant] = [
    (False, 1.0,  1.0,  1.0),   # orig, нейтральный
    (False, 1.2,  1.15, 0.9),   # orig, ярче
    (False, 0.8,  0.85, 1.1),   # orig, темнее
    (True,  1.0,  1.0,  1.0),   # hflip, нейтральный
    (True,  1.2,  1.15, 0.9),   # hflip, ярче
    (True,  0.8,  0.85, 1.1),   # hflip, темнее
]


def apply_variant(img: np.ndarray, variant: Variant) -> np.ndarray:
    """hflip + brightness/contrast/saturation на RGB."""
    hflip, br, co, sa = variant
    if not hflip and br == 1.0 and co == 1.0 and sa == 1.0:
        return img
    pil = Image.fromarray(img)
    if hflip:
        pil = TF.hflip(pil)
    if br != 1.0:
        pil = TF.adjust_brightness(pil, br)
    if co != 1.0:
        pil = TF.adjust_contrast(pil, co)
    if sa != 1.0:
        pil = TF.adjust_saturation(pil, sa)
    return np.array(pil)
