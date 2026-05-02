"""Сгенерировать константу EXPECTED для test_biologist_segmentation.py.

Запускается вручную из корня проекта:
    cd segmentation_system
    python -m tests._generate_segmentation_golden

Скрипт прогоняет auto_segment на каждом jpg из fixtures/biologist/ и печатает
готовый dict для копирования в EXPECTED тест-файла. Запускать после любого
сознательного изменения алгоритма сегментации (диф в git покажет что
поменялось --- решаешь регрессия это или улучшение).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.io import read_rgb                # noqa: E402
from core.leaf_pipeline import auto_segment  # noqa: E402
from core.paths import resource_path         # noqa: E402
from core.predictor import Predictor         # noqa: E402  (требует mobile_sam)

FIXTURES = Path(__file__).parent / "fixtures" / "biologist"


def _features(mask: np.ndarray) -> tuple[int, float, float]:
    """(area_px, centroid_x, centroid_y) бинарной маски uint8 0/255."""
    binary = mask > 0
    ys, xs = np.where(binary)
    return int(binary.sum()), float(xs.mean()), float(ys.mean())


def main() -> None:
    predictor = Predictor(str(resource_path("models/mobile_sam.pt")))
    print("EXPECTED: dict[int, list[tuple[int, float, float]]] = {")
    for img_path in sorted(FIXTURES.glob("*.jpg"), key=lambda p: int(p.stem)):
        img_id = int(img_path.stem)
        image = read_rgb(img_path)
        pairs = auto_segment(image, predictor)
        masks_feat = [_features(m) for m, _ in pairs]
        print(f"    {img_id}: [")
        for area, cx, cy in masks_feat:
            print(f"        ({area}, {cx:.1f}, {cy:.1f}),")
        print("    ],")
    print("}")


if __name__ == "__main__":
    main()
