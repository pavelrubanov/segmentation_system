"""Категория 1: точность метрик на ручных замерах биолога (9 листов).

Эталон --- длина листа и 7 поперечных ширин в пикселях, измеренные биологом
вручную. Маски пользователь делает сам через интерактивную сегментацию
приложения и кладёт в `fixtures/biologist/` рядом с исходными jpg.

Это НЕ тест точности сегментации (маска готовая, пришла извне), а тест
точности измерительной части: `transform_leaf` + `measure_leaf`.

Если папки fixtures/biologist/ нет или нет конкретного файла --- тест skip'нется.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.io import read_rgb
from core.leaf_measure import build_fractions, measure_leaf
from core.leaf_pipeline import transform_leaf


FIXTURES = Path(__file__).parent / "fixtures" / "biologist"
FRACTIONS = build_fractions(7)  # (1/8, 2/8, ..., 7/8)

REL_TOL_LENGTH = 0.03
REL_TOL_WIDTH = 0.05
ABS_TOL_PX = 15.0


# Эталон от биолога. Длина и 7 поперечных ширин в пикселях.
# `widths_base_to_apex` --- слева направо: основание листа → апекс.
# В нашем pca_crop апекс наверху (fraction=0.125 ≈ апекс), поэтому при
# сравнении биологический список реверсится.
EXPECTED: dict[int, dict] = {
    1: {"length": 1106.20, "widths_base_to_apex": (119.94, 230.59, 348.86, 459.95, 512.64, 516.11, 474.64)},
    2: {"length":  703.21, "widths_base_to_apex": (469.05, 533.11, 525.14, 530.16, 540.13, 556.40, 564.26)},
    3: {"length": 1131.98, "widths_base_to_apex": (518.93, 602.70, 623.46, 579.88, 438.83, 293.44, 165.19)},
    4: {"length": 1446.20, "widths_base_to_apex": (789.44, 880.91, 918.69, 924.13, 912.01, 903.78, 890.94)},
    5: {"length": 1992.23, "widths_base_to_apex": (668.54, 797.53, 848.25, 625.51, 767.19, 429.46, 205.57)},
    6: {"length":  766.85, "widths_base_to_apex": (217.70, 336.88, 414.27, 483.84, 520.52, 528.76, 549.57)},
    7: {"length": 1680.43, "widths_base_to_apex": (1032.76, 992.25, 969.15, 944.05, 915.82, 818.46, 627.21)},
    8: {"length":  829.17, "widths_base_to_apex": (261.73, 365.31, 432.05, 470.35, 514.39, 574.28, 592.15)},
    9: {"length":  408.18, "widths_base_to_apex": (309.14, 324.10, 311.38, 285.40, 273.30, 273.18, 271.78)},
}


def _close(got: float, expected: float, rel_tol: float) -> bool:
    return abs(got - expected) <= max(rel_tol * expected, ABS_TOL_PX)


@pytest.fixture(scope="module", params=list(EXPECTED))
def measured(request):
    """Измерить один лист и закэшировать результат на модуль."""
    img_id = request.param
    img_path = FIXTURES / f"{img_id}.jpg"
    mask_path = FIXTURES / f"{img_id}_mask_0001.png"
    if not img_path.exists() or not mask_path.exists():
        pytest.skip(f"нет файла: {img_path.name} или {mask_path.name}")

    image = read_rgb(img_path)
    mask = np.array(Image.open(mask_path).convert("L"))
    img_masked = image.copy()
    img_masked[mask == 0] = 0
    crop = transform_leaf(img_masked)
    return img_id, measure_leaf(crop, fractions=FRACTIONS)


def test_length_matches_biologist(measured):
    img_id, metrics = measured
    expected = EXPECTED[img_id]["length"]
    assert _close(metrics.length, expected, REL_TOL_LENGTH), (
        f"img {img_id}: длина got={metrics.length:.1f} "
        f"expected={expected:.1f} diff={abs(metrics.length - expected):.1f} px"
    )


def test_widths_match_biologist(measured):
    img_id, metrics = measured
    biologist = EXPECTED[img_id]["widths_base_to_apex"]
    apex_to_base = list(reversed(biologist))   # → порядок системных fractions

    failures = []
    for frac, exp_w in zip(FRACTIONS, apex_to_base):
        got = metrics.widths[frac].width
        if not _close(got, exp_w, REL_TOL_WIDTH):
            failures.append(
                f"  @{frac:.3f}: got={got:.1f} expected={exp_w:.1f} "
                f"diff={abs(got - exp_w):.1f} px"
            )
    assert not failures, f"img {img_id}:\n" + "\n".join(failures)
