# leaf_measure.py
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple, Any

import numpy as np
import cv2
import matplotlib.pyplot as plt


def _largest_external_contour(bin_u8: np.ndarray) -> np.ndarray:
    # OpenCV: findContours may return 2 or 3 values depending on version
    res = cv2.findContours(bin_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = res[0] if len(res) == 2 else res[1]
    if not contours:
        raise ValueError("No contours found in mask")

    # Choose by area (robust if there is noise)
    areas = [cv2.contourArea(c) for c in contours]
    c = contours[int(np.argmax(areas))]
    return c


def _approx_polygon(contour: np.ndarray, eps_frac: float = 0.01) -> np.ndarray:
    per = cv2.arcLength(contour, True)
    eps = max(1.0, eps_frac * per)
    poly = cv2.approxPolyDP(contour, eps, True)
    return poly


def _pca_axes_from_mask(bin_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      mean_xy: (2,) float
      u: (2,) unit vector along length (principal axis)
      v: (2,) unit vector perpendicular to u
    """
    ys, xs = np.nonzero(bin_u8)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)  # (N,2) in (x,y)
    mean = pts.mean(axis=0)

    X = pts - mean
    cov = np.cov(X.T)  # 2x2
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    order = np.argsort(eigvals)[::-1]
    u = eigvecs[:, order[0]].astype(np.float32)
    u /= (np.linalg.norm(u) + 1e-12)

    # Perpendicular (right-handed)
    v = np.array([-u[1], u[0]], dtype=np.float32)
    v /= (np.linalg.norm(v) + 1e-12)

    return mean, u, v


def _proj_range(pts_xy: np.ndarray, mean_xy: np.ndarray, axis_uv: np.ndarray) -> Tuple[float, float]:
    proj = (pts_xy - mean_xy) @ axis_uv
    return float(proj.min()), float(proj.max())


def _measure_width_on_section(
    bin_u8: np.ndarray,
    center_xy: np.ndarray,
    v: np.ndarray,
    half_span: float,
    samples_per_px: float = 4.0,
) -> Dict[str, Any]:
    """
    Samples along a line: center + t*v, t in [-half_span, +half_span].
    Finds the longest contiguous 'inside-mask' segment and returns its endpoints.
    """
    h, w = bin_u8.shape[:2]
    n = int(max(50, 2 * half_span * samples_per_px))
    ts = np.linspace(-half_span, half_span, n).astype(np.float32)

    coords = center_xy[None, :] + ts[:, None] * v[None, :]  # (n,2) float
    xs = np.rint(coords[:, 0]).astype(np.int32)
    ys = np.rint(coords[:, 1]).astype(np.int32)

    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if not np.any(valid):
        return {
            "width_px": 0.0,
            "p0": None,
            "p1": None,
            "center": (float(center_xy[0]), float(center_xy[1])),
        }

    xs_v = xs[valid]
    ys_v = ys[valid]
    ts_v = ts[valid]
    inside = (bin_u8[ys_v, xs_v] > 0).astype(np.uint8)

    idx = np.where(inside > 0)[0]
    if idx.size == 0:
        return {
            "width_px": 0.0,
            "p0": None,
            "p1": None,
            "center": (float(center_xy[0]), float(center_xy[1])),
        }

    # Find contiguous runs in idx
    breaks = np.where(np.diff(idx) > 1)[0]
    run_starts = np.r_[0, breaks + 1]
    run_ends = np.r_[breaks, idx.size - 1]

    best_len = -1
    best_s = best_e = None
    for rs, re in zip(run_starts, run_ends):
        s = idx[rs]
        e = idx[re]
        run_len = e - s
        if run_len > best_len:
            best_len = run_len
            best_s, best_e = s, e

    t0 = float(ts_v[best_s])
    t1 = float(ts_v[best_e])
    width_px = abs(t1 - t0)  # because v is unit

    p0 = center_xy + v * t0
    p1 = center_xy + v * t1

    return {
        "width_px": width_px,
        "p0": (float(p0[0]), float(p0[1])),
        "p1": (float(p1[0]), float(p1[1])),
        "center": (float(center_xy[0]), float(center_xy[1])),
    }


def measure_leaf_pca(
    bin_u8: np.ndarray,
    fractions: Iterable[float] = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875),
    mm_per_px: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Prototype leaf measurement by binary mask.

    Method:
      - external contour via findContours
      - polygon via approxPolyDP (Douglas–Peucker)
      - PCA axis from mask pixels
      - length/width = projection range (Feret/calliper)
      - widths at fractions = intersection with perpendicular section line (sampled)

    Returns dict:
      contour, polygon,
      length_px, width_px, length_mm, width_mm,
      widths_by_fraction: {f: {width_px, width_mm, p0, p1, center}}
      plus PCA axes endpoints for debug/vis.
    """

    contour = _largest_external_contour(bin_u8)
    polygon = _approx_polygon(contour, eps_frac=0.01)

    mean_xy, u, v = _pca_axes_from_mask(bin_u8)

    # Use all foreground pixels for projections (stable)
    ys, xs = np.nonzero(bin_u8)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)

    umin, umax = _proj_range(pts, mean_xy, u)
    vmin, vmax = _proj_range(pts, mean_xy, v)

    length_px = float(umax - umin)
    width_px = float(vmax - vmin)

    length_mm = (length_px * mm_per_px) if mm_per_px else None
    width_mm = (width_px * mm_per_px) if mm_per_px else None

    # Main axis endpoints for visualization
    p_len0 = mean_xy + u * umin
    p_len1 = mean_xy + u * umax

    # Width axis endpoints (through mean)
    p_w0 = mean_xy + v * vmin
    p_w1 = mean_xy + v * vmax

    # Section measurements
    widths_by_fraction: Dict[float, Dict[str, Any]] = {}
    half_span = max(10.0, (width_px / 2.0) + 5.0)  # simple margin

    for f in fractions:
        ff = float(f)
        s = umin + ff * (umax - umin)
        center = mean_xy + u * s

        sec = _measure_width_on_section(
            bin_u8=bin_u8,
            center_xy=center,
            v=v,
            half_span=half_span,
            samples_per_px=4.0,
        )
        if mm_per_px is not None:
            sec["width_mm"] = sec["width_px"] * mm_per_px
        else:
            sec["width_mm"] = None

        widths_by_fraction[ff] = sec

    out = {
        "contour": contour.reshape(-1, 2),          # (N,2) int, (x,y)
        "polygon": polygon.reshape(-1, 2),          # (M,2) int, (x,y)
        "length_px": length_px,
        "width_px": width_px,
        "length_mm": length_mm,
        "width_mm": width_mm,
        "widths_by_fraction": widths_by_fraction,
        "pca": {
            "mean_xy": (float(mean_xy[0]), float(mean_xy[1])),
            "u": (float(u[0]), float(u[1])),
            "v": (float(v[0]), float(v[1])),
            "axis_length_p0": (float(p_len0[0]), float(p_len0[1])),
            "axis_length_p1": (float(p_len1[0]), float(p_len1[1])),
            "axis_width_p0": (float(p_w0[0]), float(p_w0[1])),
            "axis_width_p1": (float(p_w1[0]), float(p_w1[1])),
        },
    }

    return out


def _draw_measurement(ax, bin_u8, result, mm_per_px, image=None):
    """Общая отрисовка результатов измерения на axes."""
    contour = result["contour"]
    polygon = result["polygon"]
    pca = result["pca"]

    if image is None:
        ax.imshow(bin_u8, cmap="gray")
    else:
        ax.imshow(image if image.ndim == 3 else image, cmap=None if image.ndim == 3 else "gray")
        ax.imshow(bin_u8, cmap="gray", alpha=0.25)

    ax.plot(contour[:, 0], contour[:, 1], linewidth=1, label="contour")
    if polygon.shape[0] >= 2:
        poly_closed = np.vstack([polygon, polygon[0]])
        ax.plot(poly_closed[:, 0], poly_closed[:, 1], linewidth=1, label="polygon")

    x0, y0 = pca["axis_length_p0"]
    x1, y1 = pca["axis_length_p1"]
    ax.plot([x0, x1], [y0, y1], linewidth=1, label="PCA length axis")

    wx0, wy0 = pca["axis_width_p0"]
    wx1, wy1 = pca["axis_width_p1"]
    ax.plot([wx0, wx1], [wy0, wy1], linewidth=1, linestyle="--", label="PCA width axis")

    for f, sec in result["widths_by_fraction"].items():
        if sec["p0"] is None or sec["p1"] is None:
            continue
        (sx0, sy0), (sx1, sy1) = sec["p0"], sec["p1"]
        ax.plot([sx0, sx1], [sy0, sy1], linewidth=1)
        cx, cy = sec["center"]
        wpx = sec["width_px"]
        wtxt = f"{wpx:.1f}px / {sec['width_mm']:.2f}mm" if mm_per_px else f"{wpx:.1f}px"
        ax.text(cx, cy, wtxt, fontsize=10, ha="center", va="bottom", color="red")

    Lpx, Wpx = result["length_px"], result["width_px"]
    if mm_per_px:
        title = f"Length={Lpx:.1f}px ({result['length_mm']:.2f}mm) | Width={Wpx:.1f}px ({result['width_mm']:.2f}mm)"
    else:
        title = f"Length={Lpx:.1f}px | Width={Wpx:.1f}px"

    ax.set_title(title)
    ax.set_xlim(0, bin_u8.shape[1] - 1)
    ax.set_ylim(bin_u8.shape[0] - 1, 0)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.axis("off")


def save_measurement_visualization(bin_u8, result, out_path, mm_per_px=None, image=None, dpi=150):
    """Сохраняет визуализацию измерения в PNG-файл."""
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_measurement(ax, bin_u8, result, mm_per_px, image)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
