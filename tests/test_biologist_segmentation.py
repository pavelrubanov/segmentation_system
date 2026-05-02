"""Категория 2: регрессионные тесты `auto_segment` на фиксированных снимках.

Это НЕ accuracy test --- ground truth от биолога на маски нет, эталон
сгенерирован самим текущим алгоритмом и закомпилирован в EXPECTED как
константа. Тест ловит **регрессии**: если рефакторинг в auto_segment / SAM /
dedupe / pca_crop сломает результат, площади и центры найденных масок
разъедутся с эталоном.

Сравнение --- по площади (rel ±2%) и центру масс (abs ±5 px) для каждой маски.
Маски сопоставляются по близости площади (greedy), порядок в списке EXPECTED
не имеет значения. Дополнительно требуется совпадение количества масок.

Если эталон пустой (`EXPECTED == {}`) или нет файла на диске --- skip. Чтобы
сгенерировать эталон:

    cd segmentation_system
    python -m tests._generate_segmentation_golden
    # → скопировать вывод в EXPECTED ниже
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.io import read_rgb
from core.leaf_pipeline import auto_segment
from core.paths import resource_path

# Predictor импортируем лениво в фикстуре --- import-time он тянет mobile_sam,
# а в CI без модельных весов и без mobile_sam пакет не установлен. Так модуль
# собирается безопасно, а сами тесты всё равно skip'аются по EXPECTED.

FIXTURES = Path(__file__).parent / "fixtures" / "biologist"

AREA_REL_TOL = 0.02      # ±2% площади
CENTROID_ABS_TOL = 5.0   # ±5 px по каждой координате


# Эталон: для каждого изображения --- список (area_px, centroid_x, centroid_y)
# по всем маскам, которые auto_segment выдал на момент генерации.
# Сгенерировать → tests/_generate_segmentation_golden.py.
EXPECTED: dict[int, list[tuple[int, float, float]]] = {
    # 1: [(234567, 1234.5, 567.8), (198765, 2345.1, 890.2)],
    # ...
}


@pytest.fixture(scope="session")
def predictor():
    from core.predictor import Predictor
    return Predictor(str(resource_path("models/mobile_sam.pt")))


def _features(mask: np.ndarray) -> tuple[int, float, float]:
    binary = mask > 0
    ys, xs = np.where(binary)
    return int(binary.sum()), float(xs.mean()), float(ys.mean())


def _match_by_area(
    actual: list[tuple[int, float, float]],
    expected: list[tuple[int, float, float]],
) -> list[tuple[int, int]]:
    """Жадный матчинг actual ↔ expected по близости площади."""
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for ai, (a_area, _, _) in enumerate(actual):
        best_ei, best_dist = -1, float("inf")
        for ei, (e_area, _, _) in enumerate(expected):
            if ei in used:
                continue
            dist = abs(a_area - e_area) / e_area
            if dist < best_dist:
                best_ei, best_dist = ei, dist
        if best_ei >= 0:
            used.add(best_ei)
            pairs.append((ai, best_ei))
    return pairs


@pytest.fixture(scope="module", params=list(EXPECTED) if EXPECTED else [])
def segmented(request, predictor):
    img_id = request.param
    img_path = FIXTURES / f"{img_id}.jpg"
    if not img_path.exists():
        pytest.skip(f"нет файла: {img_path.name}")
    image = read_rgb(img_path)
    pairs = auto_segment(image, predictor)
    return img_id, [_features(m) for m, _ in pairs]


@pytest.mark.skipif(not EXPECTED, reason="EXPECTED пуст --- сгенерируй через _generate_segmentation_golden.py")
def test_mask_count_stable(segmented):
    img_id, actual = segmented
    expected = EXPECTED[img_id]
    assert len(actual) == len(expected), (
        f"img {img_id}: число масок got={len(actual)} expected={len(expected)}"
    )


@pytest.mark.skipif(not EXPECTED, reason="EXPECTED пуст --- сгенерируй через _generate_segmentation_golden.py")
def test_mask_features_stable(segmented):
    img_id, actual = segmented
    expected = EXPECTED[img_id]
    if len(actual) != len(expected):
        pytest.skip("число масок разъехалось --- см. test_mask_count_stable")

    failures = []
    for ai, ei in _match_by_area(actual, expected):
        a_area, a_cx, a_cy = actual[ai]
        e_area, e_cx, e_cy = expected[ei]
        rel_area = abs(a_area - e_area) / e_area
        cx_diff = abs(a_cx - e_cx)
        cy_diff = abs(a_cy - e_cy)
        if rel_area > AREA_REL_TOL or cx_diff > CENTROID_ABS_TOL or cy_diff > CENTROID_ABS_TOL:
            failures.append(
                f"  {ai}↔{ei}: area={a_area} vs {e_area} ({rel_area:.2%}); "
                f"centroid=({a_cx:.1f},{a_cy:.1f}) vs ({e_cx:.1f},{e_cy:.1f}) "
                f"Δ=({cx_diff:.1f},{cy_diff:.1f})"
            )
    assert not failures, f"img {img_id}:\n" + "\n".join(failures)
