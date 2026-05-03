"""Категория 2: точность связки auto_segment + classifier на ручной разметке.

Тестируется три типа листьев независимо --- по своей модели классификатора
и своему набору снимков:

    fixtures/biologist/segmentation/<leaf_type>/<name>.jpg
    fixtures/biologist/segmentation/<leaf_type>/<name>_mask_0001.png
    fixtures/biologist/segmentation/<leaf_type>/<name>_mask_0002.png
    ...

где ``<leaf_type>`` --- ``branch``, ``capillifolium`` или ``girgensohnii``,
а маски размечают ВСЕ листья на снимке.

Для каждого случая прогоняется полная цепочка ``auto_segment`` (SAM AMG +
dedupe + pca_crop) + классификатор соответствующего типа (отбор
``good_leaf``). Эталонные маски и финальные ``good_leaf`` сопоставляются
1-к-1 жадно по убыванию IoU. Дальше две проверки:

* **Ничего не пропущено**: для каждой эталонной маски нашлась ``good_leaf``
  с IoU ≥ ``IOU_THRESHOLD``.
* **Ничего лишнего**: каждая ``good_leaf`` совпала с какой-то эталонной
  с IoU ≥ ``IOU_THRESHOLD`` (никаких артефактов, дублей, мусора, который
  классификатор не отсеял).

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


IMAGE_EXTS = ("jpg", "jpeg", "png", "tif", "tiff", "bmp")


def _list_cases() -> list[tuple[str, str]]:
    """Все (leaf_type, image_name), для которых есть снимок + хотя бы одна маска.
    `image_name` --- stem файла (без расширения), может быть произвольной строкой."""
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
    return [
        np.array(Image.open(p).convert("L")) > 127
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
    leaf_type, image_name = request.param
    if leaf_type not in classifiers:
        pytest.skip(f"нет модели для типа {leaf_type}")

    type_dir = FIXTURES / leaf_type
    img_path = _find_image(type_dir, image_name)
    image = read_rgb(img_path)
    pairs = auto_segment(image, predictor)
    expected = _load_expected_masks(type_dir, image_name)

    if not pairs:
        return leaf_type, image_name, [], expected

    crops = [c for _, c in pairs]
    masks = [m for m, _ in pairs]
    labels = classifiers[leaf_type].classify_batch(crops)
    good_masks = [
        masks[i] > 127
        for i, (cls, _) in enumerate(labels)
        if cls == "good_leaf"
    ]
    return leaf_type, image_name, good_masks, expected


# ── Тесты ────────────────────────────────────────────────────────────────────


def test_no_missed_leaves(system_output):
    """Ни один эталонный лист не должен быть пропущен системой."""
    leaf_type, image_name, actual, expected = system_output
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
        f"{leaf_type}/{image_name}: пропущено {len(failures)} из {len(expected)} "
        f"эталонных листьев (выдано {len(actual)} good_leaf):\n" + "\n".join(failures)
    )


def test_no_extra_masks(system_output):
    """Среди good_leaf не должно быть масок, которые ни с чем не совпадают."""
    leaf_type, image_name, actual, expected = system_output
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
        f"{leaf_type}/{image_name}: {len(failures)} из {len(actual)} good_leaf "
        f"не совпали ни с одной эталонной маской (эталонов всего {len(expected)}):\n"
        + "\n".join(failures)
    )
