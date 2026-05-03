"""Категория 2: точность связки auto_segment + classifier на ручной разметке.

Тестируется три типа листьев независимо --- по своей модели классификатора
и своему набору снимков:

    fixtures/biologist/segmentation/<leaf_type>/N.jpg
    fixtures/biologist/segmentation/<leaf_type>/N_mask_0001.png
    fixtures/biologist/segmentation/<leaf_type>/N_mask_0002.png
    ...

где ``<leaf_type>`` --- ``branch``, ``capillifolium`` или ``girgensohnii``,
а маски размечают ВСЕ листья на снимке (иначе precision-тест сочтёт
правильно найденный соседний лист за ложное срабатывание).

Для каждого случая прогоняется полная цепочка ``auto_segment`` (SAM AMG +
dedupe + pca_crop) + классификатор соответствующего типа (отбор
``good_leaf``). Результат сравнивается с эталонными масками по IoU:

* **Recall** --- каждая эталонная маска должна найтись среди
  ``good_leaf`` с IoU ≥ ``IOU_THRESHOLD``.
* **Precision** --- каждая ``good_leaf`` маска должна совпасть с какой-то
  эталонной с IoU ≥ ``IOU_THRESHOLD``.

Без файлов или без модели соответствующего типа --- тест skip'ается.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.io import read_rgb
from core.leaf_pipeline import auto_segment
from core.paths import resource_path

FIXTURES = Path(__file__).parent / "fixtures" / "biologist" / "segmentation"
LEAF_TYPES = ("branch", "capillifolium", "girgensohnii")
SAM_WEIGHTS = resource_path("models/mobile_sam.pt")

IOU_THRESHOLD = 0.85


# ── Утилиты ──────────────────────────────────────────────────────────────────


def _list_cases() -> list[tuple[str, int]]:
    """Все (leaf_type, img_id), для которых есть jpg + хотя бы одна маска."""
    cases = []
    for leaf_type in LEAF_TYPES:
        type_dir = FIXTURES / leaf_type
        if not type_dir.is_dir():
            continue
        for jpg in sorted(type_dir.glob("*.jpg")):
            try:
                n = int(jpg.stem)
            except ValueError:
                continue
            if any(type_dir.glob(f"{n}_mask_*.png")):
                cases.append((leaf_type, n))
    return cases


def _load_expected_masks(type_dir: Path, img_id: int) -> list[np.ndarray]:
    return [
        np.array(Image.open(p).convert("L")) > 127
        for p in sorted(type_dir.glob(f"{img_id}_mask_*.png"))
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


# ── Фикстуры (session: модели грузим один раз на весь прогон) ────────────────


@pytest.fixture(scope="session")
def predictor():
    if not SAM_WEIGHTS.exists():
        pytest.skip(f"нет весов SAM: {SAM_WEIGHTS}")
    from core.predictor import Predictor
    return Predictor(str(SAM_WEIGHTS))


@pytest.fixture(scope="session")
def classifiers() -> dict[str, object]:
    """Все доступные классификаторы по типам. Загружаются один раз."""
    from classifier import get_classifier
    out = {}
    for leaf_type in LEAF_TYPES:
        path = resource_path(f"models/model_{leaf_type}.pkl")
        if path.exists():
            out[leaf_type] = get_classifier(str(path))
    return out


_CASES = _list_cases()


@pytest.fixture(
    scope="module",
    params=_CASES if _CASES else [pytest.param(None, marks=pytest.mark.skip(reason="нет фикстур"))],
    ids=lambda p: f"{p[0]}-{p[1]}" if p else "no-fixtures",
)
def system_output(request, predictor, classifiers):
    leaf_type, img_id = request.param
    if leaf_type not in classifiers:
        pytest.skip(f"нет модели для типа {leaf_type}")

    type_dir = FIXTURES / leaf_type
    image = read_rgb(type_dir / f"{img_id}.jpg")
    pairs = auto_segment(image, predictor)
    expected = _load_expected_masks(type_dir, img_id)

    if not pairs:
        return leaf_type, img_id, [], expected

    crops = [c for _, c in pairs]
    masks = [m for m, _ in pairs]
    labels = classifiers[leaf_type].classify_batch(crops)
    good_masks = [
        masks[i] > 127
        for i, (cls, _) in enumerate(labels)
        if cls == "good_leaf"
    ]
    return leaf_type, img_id, good_masks, expected


# ── Тесты ────────────────────────────────────────────────────────────────────


def test_recall_all_expected_leaves_found(system_output):
    """Каждый эталонный лист должен быть в good_leaf с IoU ≥ IOU_THRESHOLD."""
    leaf_type, img_id, actual, expected = system_output
    matches = _greedy_match(actual, expected)
    best_per_expected = {ei: iou for _, ei, iou in matches}

    failures = []
    for ei in range(len(expected)):
        iou = best_per_expected.get(ei, 0.0)
        if iou < IOU_THRESHOLD:
            failures.append(
                f"  эталон #{ei + 1:02d}: лучший IoU={iou:.3f} (порог {IOU_THRESHOLD})"
            )
    assert not failures, (
        f"{leaf_type}/{img_id}: пропущено {len(failures)} из {len(expected)} "
        f"эталонных листьев (found {len(actual)} good_leaf):\n" + "\n".join(failures)
    )


def test_precision_no_false_positives(system_output):
    """Каждая good_leaf-маска должна соответствовать эталонной с IoU ≥ IOU_THRESHOLD."""
    leaf_type, img_id, actual, expected = system_output
    matches = _greedy_match(actual, expected)
    best_per_actual = {ai: iou for ai, _, iou in matches}

    failures = []
    for ai in range(len(actual)):
        iou = best_per_actual.get(ai, 0.0)
        if iou < IOU_THRESHOLD:
            failures.append(
                f"  good_leaf #{ai + 1:02d}: лучший IoU с эталонами={iou:.3f}"
            )
    assert not failures, (
        f"{leaf_type}/{img_id}: {len(failures)} из {len(actual)} good_leaf "
        f"не совпали ни с одной эталонной маской (expected {len(expected)}):\n"
        + "\n".join(failures)
    )
