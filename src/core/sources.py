"""Источники кропов для конвейера измерений.

Каждый источник --- callable: путь к исходному файлу превращает в
итератор готовых `CropItem`'ов. Какие-то источники дополнительно пишут на
диск свои промежуточные артефакты (raw crop, mask), но снаружи это видно
только через side-папки, переданные в конструктор.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

from PIL import Image

from .file_naming import crop_filename, mask_filename
from .io import read_rgb
from .leaf_pipeline import auto_segment
from .measurement_pipeline import CropItem

if TYPE_CHECKING:
    from classifier import LeafClassifier
    from .predictor import Predictor


def disk_source(path: str) -> Iterable[CropItem]:
    """Один файл --- один уже препроцессированный кроп.
    `name` берём из стема файла, чтобы он совпадал с тем, как кроп был сохранён."""
    yield CropItem(name=Path(path).stem, crop=read_rgb(path))


@dataclass
class AutoSegmentSource:
    """Один файл --- 0..N кропов: SAM AMG + классификатор + фильтр good_leaf.

    По пути сохраняет очищенную маску и raw crop в side-папки. Папки создаются
    в `__post_init__`, чтобы вызывающий код не дублировал mkdir.
    """
    predictor: "Predictor"
    classifier: "LeafClassifier"
    crops_dir: Path
    masks_dir: Path

    def __post_init__(self) -> None:
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, path: str) -> Iterable[CropItem]:
        stem = Path(path).stem
        image = read_rgb(path)
        pairs = auto_segment(image, self.predictor)
        if not pairs:
            return
        crops = [c for _, c in pairs]
        masks = [m for m, _ in pairs]
        labels = self.classifier.classify_batch(crops)

        idx = 0
        for j, (cls, _) in enumerate(labels):
            if cls != "good_leaf":
                continue
            idx += 1
            crop_fname = crop_filename(stem, idx)
            Image.fromarray(crops[j]).save(self.crops_dir / crop_fname)
            Image.fromarray(masks[j]).save(self.masks_dir / mask_filename(stem, idx))
            yield CropItem(name=Path(crop_fname).stem, crop=crops[j])
