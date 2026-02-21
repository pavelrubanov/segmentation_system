"""Измерение метрик листа по бинарной маске."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ── Результат ────────────────────────────────────────────────────────────────


@dataclass
class SectionWidth:
    """Ширина листа в одном поперечном сечении."""
    width_px: float
    width_mm: Optional[float]
    p0: Optional[tuple[float, float]]   # начало отрезка (x, y)
    p1: Optional[tuple[float, float]]   # конец отрезка (x, y)
    center: tuple[float, float]         # центр сечения на главной оси


@dataclass
class LeafMetrics:
    """Результат измерения листа."""
    length_px: float
    width_px: float
    length_mm: Optional[float]
    width_mm: Optional[float]

    contour: np.ndarray                     # (N, 2) int  — (x, y)
    widths: dict[float, SectionWidth]       # fraction → ширина

    # PCA-данные (для визуализации)
    mean_xy: tuple[float, float]
    axis_u: tuple[float, float]             # единичный вектор длины
    axis_v: tuple[float, float]             # единичный вектор ширины (перпенд.)
    length_endpoints: tuple[tuple[float, float], tuple[float, float]]
    width_endpoints: tuple[tuple[float, float], tuple[float, float]]


# ── Измерение ────────────────────────────────────────────────────────────────


def measure_leaf(
    mask: np.ndarray,
    fractions: Sequence[float] = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875),
    mm_per_px: Optional[float] = None,
) -> LeafMetrics:
    """
    Измеряет лист по бинарной маске (uint8, 0/255).

    Метод: PCA → главная ось → проекции (длина/ширина) →
           поперечные сечения на заданных фракциях.

    Ширины берутся от первого до последнего пикселя маски
    вдоль перпендикуляра (поддержка прерывных контуров).
    """
    # 1. Контур (наибольший внешний)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Контур не найден в маске")
    contour = max(contours, key=cv2.contourArea)

    # 2. PCA: главная ось (u) и перпендикуляр (v)
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(axis=0)

    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    u = eigvecs[:, np.argmax(eigvals)].astype(np.float32)
    u /= np.linalg.norm(u) + 1e-12
    v = np.array([-u[1], u[0]], dtype=np.float32)

    # 3. Проекции → длина / ширина (Feret)
    proj_u = (pts - mean) @ u
    proj_v = (pts - mean) @ v
    umin, umax = float(proj_u.min()), float(proj_u.max())
    vmin, vmax = float(proj_v.min()), float(proj_v.max())

    length_px = umax - umin
    width_px = vmax - vmin

    # 4. Ширины по фракциям (прерывные линии — от первого до последнего пикселя)
    widths = _measure_widths(mask, mean, u, v, umin, umax, width_px, fractions, mm_per_px)

    # 5. Концы осей (для визуализации)
    p_len = (_to_tuple(mean + u * umin), _to_tuple(mean + u * umax))
    p_wid = (_to_tuple(mean + v * vmin), _to_tuple(mean + v * vmax))

    return LeafMetrics(
        length_px=length_px,
        width_px=width_px,
        length_mm=length_px * mm_per_px if mm_per_px else None,
        width_mm=width_px * mm_per_px if mm_per_px else None,
        contour=contour.reshape(-1, 2),
        widths=widths,
        mean_xy=_to_tuple(mean),
        axis_u=(float(u[0]), float(u[1])),
        axis_v=(float(v[0]), float(v[1])),
        length_endpoints=p_len,
        width_endpoints=p_wid,
    )


# ── Визуализация ─────────────────────────────────────────────────────────────


def save_measurement_visualization(
    mask: np.ndarray,
    metrics: LeafMetrics,
    out_path: str,
    dpi: int = 150,
) -> None:
    """Сохраняет визуализацию измерения в PNG."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mask, cmap="gray")

    # контур
    ax.plot(metrics.contour[:, 0], metrics.contour[:, 1], lw=1, label="контур")

    # оси PCA
    (x0, y0), (x1, y1) = metrics.length_endpoints
    ax.plot([x0, x1], [y0, y1], lw=1, label="ось длины")
    (x0, y0), (x1, y1) = metrics.width_endpoints
    ax.plot([x0, x1], [y0, y1], lw=1, label="ось ширины")

    # сечения
    for sec in metrics.widths.values():
        if sec.p0 is None or sec.p1 is None:
            continue
        ax.plot([sec.p0[0], sec.p1[0]], [sec.p0[1], sec.p1[1]], lw=1)
        label = f"{sec.width_px:.1f}px"
        if sec.width_mm is not None:
            label += f" / {sec.width_mm:.2f}mm"
        ax.text(*sec.center, label, fontsize=10, ha="center", va="bottom", color="red")

    # заголовок
    title = f"Длина={metrics.length_px:.1f}px | Ширина={metrics.width_px:.1f}px"
    if metrics.length_mm is not None:
        title = (
            f"Длина={metrics.length_px:.1f}px ({metrics.length_mm:.2f}mm) | "
            f"Ширина={metrics.width_px:.1f}px ({metrics.width_mm:.2f}mm)"
        )
    ax.set_title(title)
    ax.set_xlim(0, mask.shape[1] - 1)
    ax.set_ylim(mask.shape[0] - 1, 0)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.axis("off")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ── Вспомогательные ──────────────────────────────────────────────────────────


def _measure_widths(
    mask: np.ndarray,
    mean: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    umin: float,
    umax: float,
    width_px: float,
    fractions: Sequence[float],
    mm_per_px: Optional[float],
) -> dict[float, SectionWidth]:
    """Измеряет ширину листа на каждой фракции вдоль главной оси."""
    h, w = mask.shape[:2]
    half_span = max(10.0, width_px / 2 + 5)
    n_samples = int(max(50, 2 * half_span * 4))
    ts = np.linspace(-half_span, half_span, n_samples, dtype=np.float32)

    result: dict[float, SectionWidth] = {}

    for frac in fractions:
        center = mean + u * (umin + frac * (umax - umin))

        # Сканируем вдоль перпендикуляра v
        coords = center[None, :] + ts[:, None] * v[None, :]
        xi = np.rint(coords[:, 0]).astype(int)
        yi = np.rint(coords[:, 1]).astype(int)

        in_bounds = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        inside = np.zeros(n_samples, dtype=bool)
        inside[in_bounds] = mask[yi[in_bounds], xi[in_bounds]] > 0

        hits = np.where(inside)[0]
        if hits.size == 0:
            result[frac] = SectionWidth(0.0, None, None, None, _to_tuple(center))
            continue

        # От первого до последнего пикселя маски (прерывные линии учтены)
        t0, t1 = float(ts[hits[0]]), float(ts[hits[-1]])
        wpx = abs(t1 - t0)
        p0 = center + v * t0
        p1 = center + v * t1

        result[frac] = SectionWidth(
            width_px=wpx,
            width_mm=wpx * mm_per_px if mm_per_px else None,
            p0=_to_tuple(p0),
            p1=_to_tuple(p1),
            center=_to_tuple(center),
        )

    return result


def _to_tuple(arr: np.ndarray) -> tuple[float, float]:
    return (float(arr[0]), float(arr[1]))

