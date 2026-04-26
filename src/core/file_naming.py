"""Конвенция имён файлов crop/mask/vis и Qt-фильтры для их выбора.

Единая точка правды для:
  - сохранения crop/mask (ручная и авто сегментация дают одинаковые имена ---
    результаты можно смешивать в одной папке);
  - попарного сопоставления crop ↔ mask по общему {stem}_*_{idx:04d};
  - Qt-фильтров в диалогах открытия (measure_crops, measure_images);
  - сборщика обучающих данных в classifier (он держит свою копию CROP_INFIX,
    см. оговорку в classifier/mobilenet.py --- classifier остаётся независимым
    пакетом, поэтому не импортирует отсюда).
"""
from __future__ import annotations

CROP_INFIX = "leaf_crop"
MASK_INFIX = "mask"

# Расширения, которые принимаем при чтении (наш формат сохранения --- всегда PNG).
SUPPORTED_INPUT_EXTENSIONS = ("png", "jpg", "jpeg", "bmp", "tif")


def crop_filename(stem: str, idx: int) -> str:
    """Имя crop-файла: 'IMG_4321_leaf_crop_0001.png'."""
    return f"{stem}_{CROP_INFIX}_{idx:04d}.png"


def mask_filename(stem: str, idx: int) -> str:
    """Имя mask-файла: 'IMG_4321_mask_0001.png'."""
    return f"{stem}_{MASK_INFIX}_{idx:04d}.png"


def vis_filename(crop_stem: str) -> str:
    """Имя vis-файла рядом с кропом: '{crop_stem}_vis.png'.
    На вход --- стем (имя без расширения) crop-файла."""
    return f"{crop_stem}_vis.png"


def is_crop_filename(filename: str) -> bool:
    """True, если имя файла соответствует crop-конвенции."""
    return f"_{CROP_INFIX}_" in filename


def crop_file_filter() -> str:
    """Строка фильтра для Qt QFileDialog --- только crop-файлы."""
    patterns = " ".join(f"*_{CROP_INFIX}_*.{ext}" for ext in SUPPORTED_INPUT_EXTENSIONS)
    return f"Leaf crops ({patterns})"


def image_file_filter() -> str:
    """Строка фильтра для Qt QFileDialog --- любые изображения."""
    patterns = " ".join(f"*.{ext}" for ext in SUPPORTED_INPUT_EXTENSIONS)
    return f"Images ({patterns})"
