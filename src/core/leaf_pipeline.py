"""Препроцессинг листа после сегментации: чистка маски, PCA-канонизация, кроп.

Точка входа для автосегментации --- `auto_segment`. Для одной маски (например,
из ручной сегментации) --- `transform_leaf`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .predictor import Predictor

DOWNSCALE: int = 2  # на уменьшенной копии гоняем тяжёлые шаги (AMG, NMS, PCA)


def preprocess_mask(mask: np.ndarray) -> np.ndarray:
    """Оставить крупнейшую компоненту маски и залить дыры."""
    binary = np.ascontiguousarray((mask > 0).astype(np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary * 255

    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros(binary.shape, dtype=np.uint8)
    cv2.drawContours(filled, [largest], -1, 1, cv2.FILLED)
    return filled * 255


def preprocess_img_masked(img_masked: np.ndarray) -> np.ndarray:
    """То же что preprocess_mask, но для RGB --- зануляем пиксели вне чистой маски."""
    mask = (img_masked.sum(axis=2) > 0).astype(np.uint8) * 255
    processed = preprocess_mask(mask)
    result = img_masked.copy()
    result[processed == 0] = 0
    return result


def _axis_bbox(img_masked: np.ndarray) -> np.ndarray:
    """Тесный axis-aligned bbox по ненулевым пикселям (без поворота)."""
    mask = img_masked.sum(axis=2) > 0
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return img_masked[y0:y1, x0:x1]


def pca_crop(img_masked: np.ndarray) -> np.ndarray:
    """Поворот по PCA: главная ось вертикальна, узкий конец (апекс) --- вверху."""
    mask = img_masked.sum(axis=2) > 0
    ys, xs = np.where(mask)
    if len(ys) < 8:
        return _axis_bbox(img_masked)

    pts = np.column_stack([xs, ys]).astype(np.float64)
    cx, cy = pts.mean(axis=0)
    cov = np.cov(pts.T)
    _, eigvecs = np.linalg.eigh(cov)
    main_axis = eigvecs[:, -1]
    angle_deg = np.degrees(np.arctan2(main_axis[1], main_axis[0]))
    rotate_by = angle_deg - 90.0

    M = cv2.getRotationMatrix2D((float(cx), float(cy)), rotate_by, 1.0)

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    corners = np.array([[x0, x1, x1, x0],
                        [y0, y0, y1, y1],
                        [1,  1,  1,  1]], dtype=np.float64)
    rotated = M @ corners
    rx0, rx1 = float(rotated[0].min()), float(rotated[0].max())
    ry0, ry1 = float(rotated[1].min()), float(rotated[1].max())
    M[0, 2] -= rx0
    M[1, 2] -= ry0

    new_w = max(1, int(np.ceil(rx1 - rx0)))
    new_h = max(1, int(np.ceil(ry1 - ry0)))
    rotated_img = cv2.warpAffine(
        img_masked, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )

    # Если масса листа в верхней половине --- значит апекс снизу, перевернём.
    rot_mask = rotated_img.sum(axis=2) > 0
    mid = rotated_img.shape[0] // 2
    if rot_mask[:mid, :].sum() > rot_mask[mid:, :].sum():
        rotated_img = rotated_img[::-1, :, :].copy()

    return _axis_bbox(rotated_img)


def transform_leaf(img_masked: np.ndarray) -> np.ndarray:
    """Сырой img_masked: чистая маска, поворот по PCA, обрезка до bbox.
    Используется и в MasksBuffer, и в auto_segment."""
    return pca_crop(preprocess_img_masked(img_masked))


def _mask_bbox(b: np.ndarray) -> tuple[int, int, int, int] | None:
    """(y0, y1, x0, x1), границы включительно. None для пустой маски."""
    ys, xs = np.where(b)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _bbox_intersect(a, b) -> tuple[int, int, int, int] | None:
    """Пересечение двух bbox в формате _mask_bbox; None если не пересекаются."""
    y0 = max(a[0], b[0]); y1 = min(a[1], b[1])
    x0 = max(a[2], b[2]); x1 = min(a[3], b[3])
    if y1 < y0 or x1 < x0:
        return None
    return y0, y1, x0, x1


def dedupe_and_transform_leaf(
    image: np.ndarray,
    masks: list[np.ndarray],
    iou_threshold: float = 0.9,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Чистим маски (preprocess_mask), выкидываем дубли по IoU, делаем pca_crop.

    На вход --- бинарные маски [H, W] uint8 0/255 (например, от SAM.predict_all).
    NMS-оптимизация: до полного IoU проверяем пересечение bbox'ов; если
    пересекаются --- `logical_and` считается только на области пересечения,
    а union --- по формуле |A| + |B| - |A∩B|. На 4K-кадре это критично для
    скорости.

    Возвращает [(cleaned_mask, crop), ...] в порядке убывания площади:
      cleaned_mask --- бинарная маска uint8 0/255 в координатах исходной картинки;
      crop --- RGB-кроп после pca_crop, апекс вверху.
    """
    if not masks:
        return []

    cleaned = [preprocess_mask((m > 0).astype(np.uint8) * 255) for m in masks]
    bool_masks = [c > 0 for c in cleaned]
    areas = [int(b.sum()) for b in bool_masks]
    bboxes = [_mask_bbox(b) for b in bool_masks]
    order = sorted(range(len(masks)), key=lambda i: -areas[i])

    kept_idx: list[int] = []
    for i in order:
        if areas[i] == 0:
            continue
        ok = True
        for j in kept_idx:
            ibb = _bbox_intersect(bboxes[i], bboxes[j])
            if ibb is None:
                continue
            y0, y1, x0, x1 = ibb
            inter = int(np.logical_and(
                bool_masks[i][y0:y1 + 1, x0:x1 + 1],
                bool_masks[j][y0:y1 + 1, x0:x1 + 1],
            ).sum())
            if inter == 0:
                continue
            union = areas[i] + areas[j] - inter
            if inter / union > iou_threshold:
                ok = False
                break
        if ok:
            kept_idx.append(i)

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i in kept_idx:
        bb = bboxes[i]
        if bb is None:
            continue
        # Достаём только окрестность маски, чтобы не копировать весь кадр.
        y0, y1, x0, x1 = bb
        region = image[y0:y1 + 1, x0:x1 + 1].copy()
        region[~bool_masks[i][y0:y1 + 1, x0:x1 + 1]] = 0
        out.append((cleaned[i], pca_crop(region)))
    return out


def auto_segment(
    image: np.ndarray,
    predictor: "Predictor",
    iou_threshold: float = 0.9,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Полный конвейер автосегментации: SAM AMG, дедупликация, PCA-кроп.

    Тяжёлые шаги (AMG, preprocess_mask, IoU NMS, pca_crop) идут на копии,
    уменьшенной в DOWNSCALE раз --- на 4K-кадре даёт ускорение примерно
    на порядок (см. _claude/experiments/auto_segment_optimization.md).
    Выжившие маски и кропы апсэмплятся обратно к full-res через cv2.resize:
    повторно делать pca_crop на full-res смысла нет --- ориентация уже найдена,
    а классификатору хватает интерполированного разрешения.

    Возвращает [(cleaned_mask, crop), ...] в координатах исходного image.
    """
    h, w = image.shape[:2]
    if DOWNSCALE > 1:
        small = cv2.resize(
            image, (w // DOWNSCALE, h // DOWNSCALE),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = image

    raw_masks = predictor.predict_all(small)
    pairs_small = dedupe_and_transform_leaf(small, raw_masks, iou_threshold)

    if DOWNSCALE == 1:
        return pairs_small

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for small_mask, small_crop in pairs_small:
        full_mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        full_mask = (full_mask > 127).astype(np.uint8) * 255
        sh, sw = small_crop.shape[:2]
        full_crop = cv2.resize(
            small_crop, (sw * DOWNSCALE, sh * DOWNSCALE),
            interpolation=cv2.INTER_LINEAR,
        )
        out.append((full_mask, full_crop))
    return out
