"""Сверяем auto_segment + classifier с ручной разметкой биолога.

Для трёх типов листьев (branch, capillifolium, girgensohnii) лежат снимки
и набор масок-эталонов на каждый снимок. Прогоняем полную цепочку,
жадно матчим по IoU, считаем FP/FN — пороги MAX_FP/MAX_FN ловят регрессию,
но дают небольшой бюджет на пограничные случаи.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.io import read_rgb
from core.leaf_pipeline import auto_segment
from core.paths import resource_path

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent / "fixtures" / "biologist" / "segmentation"
LEAF_TYPES = ("branch", "capillifolium", "girgensohnii")
SAM_WEIGHTS = resource_path("models/mobile_sam.pt")
IMAGE_EXTS = ("jpg", "jpeg", "png", "tif", "tiff", "bmp")

IOU_THRESHOLD = 0.85
MAX_FP = 2
MAX_FN = 2


def _list_cases() -> list[tuple[str, str]]:
    """Все (leaf_type, image_name), для которых есть снимок + хотя бы одна маска."""
    cases = []
    for leaf_type in LEAF_TYPES:
        type_dir = FIXTURES / leaf_type
        if not type_dir.is_dir():
            continue
        images = sorted(p for ext in IMAGE_EXTS for p in type_dir.glob(f"*.{ext}"))
        for img_path in images:
            stem = img_path.stem
            if any(type_dir.glob(f"{stem}_mask_*.png")):
                cases.append((leaf_type, stem))
    return cases


def _find_image(type_dir: Path, image_name: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = type_dir / f"{image_name}.{ext}"
        if p.exists():
            return p
    return None


def _load_expected_masks(type_dir: Path, image_name: str) -> list[np.ndarray]:
    # Эталоны сохранены приложением как image*mask (RGB, чёрный фон), не бинарные.
    return [
        np.array(Image.open(p).convert("RGB")).sum(axis=2) > 0
        for p in sorted(type_dir.glob(f"{image_name}_mask_*.png"))
    ]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def _greedy_match(actual: list[np.ndarray], expected: list[np.ndarray]) -> list[tuple[int, int, float]]:
    candidates = [
        (_iou(a, e), ai, ei)
        for ai, a in enumerate(actual)
        for ei, e in enumerate(expected)
        if a.shape == e.shape
    ]
    candidates.sort(reverse=True)

    used_a, used_e, matches = set(), set(), []
    for iou, ai, ei in candidates:
        if ai in used_a or ei in used_e:
            continue
        used_a.add(ai)
        used_e.add(ei)
        matches.append((ai, ei, iou))
    return matches


@pytest.fixture(scope="session")
def predictor():
    if not SAM_WEIGHTS.exists():
        pytest.skip(f"нет весов SAM: {SAM_WEIGHTS}")
    from core.predictor import Predictor
    return Predictor(str(SAM_WEIGHTS))


@pytest.fixture(scope="session")
def classifiers() -> dict[str, object]:
    from classifier import get_classifier
    bb = str(resource_path("models/mobilenet_v3_small.pth"))
    out = {}
    for leaf_type in LEAF_TYPES:
        path = resource_path(f"models/model_{leaf_type}.pkl")
        if path.exists():
            out[leaf_type] = get_classifier(str(path), bb)
    return out


@pytest.fixture(scope="session")
def aggregated_results(predictor, classifiers) -> list[dict]:
    """Гоняем всю цепочку на всех кейсах, считаем FP/FN на случай."""
    cases = _list_cases()
    if not cases:
        pytest.skip(f"нет фикстур в {FIXTURES}")

    results = []
    for leaf_type, image_name in cases:
        if leaf_type not in classifiers:
            continue
        type_dir = FIXTURES / leaf_type
        img_path = _find_image(type_dir, image_name)
        if img_path is None:
            continue

        image = read_rgb(img_path)
        pairs = auto_segment(image, predictor)
        expected = _load_expected_masks(type_dir, image_name)

        if not pairs:
            actual = []
        else:
            crops = [c for _, c in pairs]
            masks = [m for m, _ in pairs]
            labels = classifiers[leaf_type].classify_batch(crops)
            actual = [
                masks[i] > 127
                for i, (cls, _) in enumerate(labels)
                if cls == "good_leaf"
            ]

        matches = _greedy_match(actual, expected)
        good_actual = {ai for ai, _, iou in matches if iou >= IOU_THRESHOLD}
        good_expected = {ei for _, ei, iou in matches if iou >= IOU_THRESHOLD}

        results.append({
            "case": f"{leaf_type}/{image_name}",
            "n_actual": len(actual),
            "n_expected": len(expected),
            "fp": len(actual) - len(good_actual),
            "fn": len(expected) - len(good_expected),
        })
    return results


def test_total_false_positives_within_budget(aggregated_results):
    total_fp = sum(r["fp"] for r in aggregated_results)
    if total_fp <= MAX_FP:
        return
    details = "\n".join(
        f"  {r['case']}: FP={r['fp']} из {r['n_actual']} good_leaf"
        for r in aggregated_results if r["fp"] > 0
    )
    pytest.fail(f"всего FP={total_fp} (бюджет ≤{MAX_FP}):\n{details}")


def test_total_false_negatives_within_budget(aggregated_results):
    total_fn = sum(r["fn"] for r in aggregated_results)
    if total_fn <= MAX_FN:
        return
    details = "\n".join(
        f"  {r['case']}: FN={r['fn']} из {r['n_expected']} эталонов"
        for r in aggregated_results if r["fn"] > 0
    )
    pytest.fail(f"всего FN={total_fn} (бюджет ≤{MAX_FN}):\n{details}")
