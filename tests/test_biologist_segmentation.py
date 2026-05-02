"""Категория 2: точность связки auto_segment + classifier на ручной разметке.

Эталон --- бинарные PNG-маски ВСЕХ листьев на снимке, размеченные вручную
через интерактивную сегментацию приложения. Имена по конвенции системы:
``N_mask_0001.png``, ``N_mask_0002.png``, ... (любое количество).

Тест прогоняет полную цепочку, которой пользуется приложение в режиме
``measure_images``: ``auto_segment`` (SAM AMG → dedupe → pca_crop) +
классификатор (отбор ``good_leaf``). Результирующие маски сравниваются
с эталоном по IoU:

* **Recall** --- каждая эталонная маска должна быть найдена среди
  ``good_leaf`` с IoU ≥ ``IOU_THRESHOLD``. Иначе система пропустила лист.
* **Precision** --- каждая ``good_leaf`` маска должна соответствовать
  какой-то эталонной с IoU ≥ ``IOU_THRESHOLD``. Иначе мусор/дубль
  просочился сквозь классификатор.

Тип классификатора задаётся через переменную окружения ``LEAF_TYPE``
(по умолчанию ``branch``). Без файлов в ``fixtures/biologist/`` или без
весов SAM/классификатора --- тесты skip'аются.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.io import read_rgb
from core.leaf_pipeline import auto_segment
from core.paths import resource_path

FIXTURES = Path(__file__).parent / "fixtures" / "biologist" / "segmentation"

IOU_THRESHOLD = 0.85
LEAF_TYPE = os.environ.get("LEAF_TYPE", "branch")
SAM_WEIGHTS = resource_path("models/mobile_sam.pt")
CLF_WEIGHTS = resource_path(f"models/model_{LEAF_TYPE}.pkl")


# ── Утилиты ──────────────────────────────────────────────────────────────────


def _list_image_ids() -> list[int]:
    """Все N.jpg, для которых есть хотя бы одна эталонная маска."""
    ids = []
    for p in sorted(FIXTURES.glob("*.jpg")):
        try:
            n = int(p.stem)
        except ValueError:
            continue
        if any(FIXTURES.glob(f"{n}_mask_*.png")):
            ids.append(n)
    return ids


def _load_expected_masks(img_id: int) -> list[np.ndarray]:
    out = []
    for p in sorted(FIXTURES.glob(f"{img_id}_mask_*.png")):
        out.append(np.array(Image.open(p).convert("L")) > 127)
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def _greedy_match(actual: list[np.ndarray], expected: list[np.ndarray]) -> list[tuple[int, int, float]]:
    """Жадный матчинг 1-к-1 по убыванию IoU. Возвращает [(ai, ei, iou), ...]."""
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


# ── Фикстуры (session: грузим модели один раз) ───────────────────────────────


@pytest.fixture(scope="session")
def predictor():
    if not SAM_WEIGHTS.exists():
        pytest.skip(f"нет весов SAM: {SAM_WEIGHTS}")
    from core.predictor import Predictor
    return Predictor(str(SAM_WEIGHTS))


@pytest.fixture(scope="session")
def classifier():
    if not CLF_WEIGHTS.exists():
        pytest.skip(f"нет весов классификатора: {CLF_WEIGHTS}")
    from classifier import get_classifier
    return get_classifier(str(CLF_WEIGHTS))


@pytest.fixture(scope="module", params=_list_image_ids() or [pytest.param(None, marks=pytest.mark.skip(reason="нет фикстур"))])
def system_output(request, predictor, classifier):
    """Прогон полной цепочки: auto_segment + classifier. Возвращает маски good_leaf."""
    img_id = request.param
    image = read_rgb(FIXTURES / f"{img_id}.jpg")
    pairs = auto_segment(image, predictor)
    if not pairs:
        return img_id, [], _load_expected_masks(img_id)

    crops = [c for _, c in pairs]
    masks = [m for m, _ in pairs]
    labels = classifier.classify_batch(crops)
    good_masks = [
        masks[i] > 127
        for i, (cls, _) in enumerate(labels)
        if cls == "good_leaf"
    ]
    return img_id, good_masks, _load_expected_masks(img_id)


# ── Тесты ────────────────────────────────────────────────────────────────────


def test_recall_all_expected_leaves_found(system_output):
    """Каждый эталонный лист должен быть в good_leaf с IoU ≥ IOU_THRESHOLD."""
    img_id, actual, expected = system_output
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
        f"img {img_id}: пропущено {len(failures)} из {len(expected)} эталонных листьев "
        f"(found {len(actual)} good_leaf):\n" + "\n".join(failures)
    )


def test_precision_no_false_positives(system_output):
    """Каждая good_leaf-маска должна соответствовать эталонной с IoU ≥ IOU_THRESHOLD."""
    img_id, actual, expected = system_output
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
        f"img {img_id}: {len(failures)} из {len(actual)} good_leaf не совпали "
        f"ни с одной эталонной маской (expected {len(expected)}):\n" + "\n".join(failures)
    )
