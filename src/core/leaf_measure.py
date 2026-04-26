"""Измерение длины, ширины и поперечных сечений листа.

Вход --- RGB-кроп из `core.leaf_pipeline.transform_leaf()`:
  - главная ось вертикальна (строки = длина, столбцы = ширина),
  - апекс вверху,
  - чёрный фон вне листа.

Поэтому ничего поворачивать тут не надо: длина = высота bbox, ширина =
максимум по строкам, поперечные сечения --- чтение пикселей в выбранных строках.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SectionWidth:
    """Ширина листа в одном поперечном сечении."""
    width: float                            # в единицах unit
    p0: Optional[tuple[float, float]]       # левый край сечения (x, y)
    p1: Optional[tuple[float, float]]       # правый край (x, y)
    center: tuple[float, float]             # центр сечения


@dataclass
class LeafMetrics:
    """Результат измерения листа."""
    length: float
    width: float
    unit: str                                  # "px", "мм", "см", ...
    widths: dict[float, SectionWidth]          # доля длины: ширина в этом сечении


def _crop_to_mask(crop: np.ndarray) -> np.ndarray:
    """RGB-кроп с чёрным фоном вернуть как bool-маску листа."""
    return crop.sum(axis=2) > 0


def measure_leaf(
    crop: np.ndarray,
    fractions: Sequence[float] = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875),
    unit: str = "px",
    scale: float = 1.0,
) -> LeafMetrics:
    """Длина = высота bbox, ширина = его ширина. На каждой доле длины меряем
    расстояние от крайнего левого до крайнего правого пикселя листа.
    `scale` --- единиц на пиксель (1.0 = вернуть пиксели).
    """
    mask = _crop_to_mask(crop)
    if not mask.any():
        raise ValueError("Кроп пуст (нет пикселей листа)")

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    length_px = y1 - y0 + 1
    width_px  = x1 - x0 + 1

    widths: dict[float, SectionWidth] = {}
    for frac in fractions:
        row = int(round(y0 + frac * (y1 - y0)))
        row = min(max(row, 0), mask.shape[0] - 1)
        row_xs = np.where(mask[row])[0]
        if row_xs.size == 0:
            widths[frac] = SectionWidth(
                width=0.0, p0=None, p1=None,
                center=((x0 + x1) / 2.0, float(row)),
            )
            continue
        left, right = int(row_xs[0]), int(row_xs[-1])
        w_px = right - left + 1
        widths[frac] = SectionWidth(
            width=w_px * scale,
            p0=(float(left),  float(row)),
            p1=(float(right), float(row)),
            center=((left + right) / 2.0, float(row)),
        )

    return LeafMetrics(
        length=length_px * scale,
        width=width_px * scale,
        unit=unit,
        widths=widths,
    )


def build_fractions(n: int) -> tuple[float, ...]:
    """n равномерных долей внутри (0, 1) для поперечных сечений."""
    return tuple(i / (n + 1) for i in range(1, n + 1))


def column_names_for(fractions: Sequence[float], unit: str) -> list[str]:
    """Имена колонок для CSV/XLSX: image, length_<unit>, width_<unit>, width_<frac>_<unit>...
    Без поля 'crop' --- его добавляет export-модуль, если нужна миниатюра.
    """
    cols = ["image", f"length_{unit}", f"width_{unit}"]
    cols += [f"width_{f:.3f}_{unit}" for f in fractions]
    return cols


def save_measurement_visualization(
    crop: np.ndarray,
    metrics: LeafMetrics,
    out_path: str,
    dpi: int = 150,
) -> None:
    """Рисует кроп с осью длины и поперечными сечениями, пишет PNG."""
    u = metrics.unit
    mask = _crop_to_mask(crop)
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x_mid = (int(xs.min()) + int(xs.max())) / 2.0

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(crop)

    ax.plot([x_mid, x_mid], [y0, y1], lw=1.5, color="cyan", label="ось длины")

    for sec in metrics.widths.values():
        if sec.p0 is None or sec.p1 is None:
            continue
        ax.plot([sec.p0[0], sec.p1[0]], [sec.p0[1], sec.p1[1]], lw=1, color="red")
        ax.text(sec.center[0], sec.center[1] - 3, f"{sec.width:.2f} {u}",
                fontsize=8, ha="center", va="bottom", color="red")

    ax.set_title(f"Длина={metrics.length:.2f} {u} | Ширина={metrics.width:.2f} {u}")
    ax.set_xlim(0, crop.shape[1] - 1)
    ax.set_ylim(crop.shape[0] - 1, 0)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.axis("off")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
