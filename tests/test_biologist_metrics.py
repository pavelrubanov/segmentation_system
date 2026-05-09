"""Сверяем measure_leaf на канонических кропах с ручными замерами биолога.

Кроп уже после transform_leaf (PCA, апекс наверху). Сегментацию не трогаем.
Файлы — fixtures/biologist/metrics/{N}_leaf_crop_0001.png, без файла skip.
"""
from pathlib import Path

import pytest

from core.io import read_rgb
from core.leaf_measure import build_fractions, measure_leaf


pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent / "fixtures" / "biologist" / "metrics"
FRACTIONS = build_fractions(7)

REL_TOL_LENGTH = 0.15
REL_TOL_WIDTH = 0.12
ABS_TOL_PX = 15.0


EXPECTED: dict[str, dict] = {
    "1": {"length": 1106.20, "widths": (119.94, 230.59, 348.86, 459.95, 512.64, 516.11, 474.64)},
    "2": {"length":  703.21, "widths": (469.05, 533.11, 525.14, 530.16, 540.13, 556.40, 564.26)},
    "3": {"length": 1131.98, "widths": (165.19, 293.44, 438.83, 579.88, 623.46, 602.70, 518.93)},
    "4": {"length": 1446.20, "widths": (789.44, 880.91, 918.69, 924.13, 912.01, 903.78, 890.94)},
    "5": {"length": 1992.23, "widths": (205.57, 429.46, 625.51, 767.19, 848.25, 797.53, 668.54)},
    "6": {"length":  766.85, "widths": (549.57, 528.76, 520.52, 483.84, 414.27, 336.88, 217.70)},
    "7": {"length": 1680.43, "widths": (1032.76, 992.25, 969.15, 944.05, 915.82, 818.46, 627.21)},
    "8": {"length":  829.17, "widths": (261.73, 365.31, 432.05, 470.35, 514.39, 574.28, 592.15)},
    "9": {"length":  408.18, "widths": (271.78, 273.18, 273.30, 285.40, 311.38, 324.10, 309.14)},
}


def _close(got: float, expected: float, rel_tol: float) -> bool:
    return abs(got - expected) <= max(rel_tol * expected, ABS_TOL_PX)


@pytest.fixture(scope="module", params=list(EXPECTED))
def measured(request):
    leaf_name = request.param
    crop_path = FIXTURES / f"{leaf_name}_leaf_crop_0001.png"
    if not crop_path.exists():
        pytest.skip(f"нет канонического кропа: {crop_path.name}")
    crop = read_rgb(crop_path)
    return leaf_name, measure_leaf(crop, fractions=FRACTIONS)


def test_length_matches_biologist(measured):
    leaf_name, metrics = measured
    expected = EXPECTED[leaf_name]["length"]
    assert _close(metrics.length, expected, REL_TOL_LENGTH), (
        f"лист {leaf_name}: длина got={metrics.length:.1f} "
        f"expected={expected:.1f} diff={abs(metrics.length - expected):.1f} px"
    )


def test_widths_match_biologist(measured):
    leaf_name, metrics = measured
    biologist = EXPECTED[leaf_name]["widths"]

    failures = []
    for frac, exp_w in zip(FRACTIONS, biologist):
        got = metrics.widths[frac].width
        if not _close(got, exp_w, REL_TOL_WIDTH):
            failures.append(
                f"  @{frac:.3f}: got={got:.1f} expected={exp_w:.1f} "
                f"diff={abs(got - exp_w):.1f} px"
            )
    assert not failures, f"лист {leaf_name}:\n" + "\n".join(failures)
