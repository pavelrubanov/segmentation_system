"""core/leaf_pipeline.py: чистка маски, PCA-кроп, дедупликация, auto_segment."""
from unittest.mock import MagicMock

import numpy as np

from core.leaf_pipeline import (
    auto_segment,
    dedupe_and_transform_leaf,
    pca_crop,
    preprocess_mask,
    transform_leaf,
)


def _mask(h, w, *blobs):
    # blobs — кортежи (y0, y1, x0, x1) для прямоугольников.
    m = np.zeros((h, w), dtype=np.uint8)
    for y0, y1, x0, x1 in blobs:
        m[y0:y1, x0:x1] = 255
    return m


def _rgb_from_mask(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask > 0] = (255, 255, 255)
    return rgb


class TestPreprocessMask:
    def test_returns_uint8_binary(self):
        out = preprocess_mask(_mask(50, 50, (10, 30, 10, 30)))
        assert out.dtype == np.uint8
        assert set(np.unique(out)).issubset({0, 255})

    def test_keeps_largest_component(self):
        m = _mask(50, 50, (5, 25, 5, 25), (40, 43, 40, 43))
        out = preprocess_mask(m)
        assert out[40:43, 40:43].sum() == 0
        assert out[5:25, 5:25].sum() > 0

    def test_fills_holes(self):
        m = _mask(60, 60, (5, 55, 5, 55))
        m[20:30, 20:30] = 0
        out = preprocess_mask(m)
        assert out[25, 25] == 255

    def test_empty_mask(self):
        out = preprocess_mask(np.zeros((20, 20), dtype=np.uint8))
        assert out.shape == (20, 20)
        assert out.sum() == 0

    def test_shape_preserved(self):
        out = preprocess_mask(_mask(80, 50, (10, 20, 10, 20)))
        assert out.shape == (80, 50)


class TestPcaCrop:
    def test_already_vertical_stays_vertical(self):
        out = pca_crop(_rgb_from_mask(_mask(120, 80, (10, 110, 30, 50))))
        h, w = out.shape[:2]
        assert h >= w

    def test_horizontal_rotated_to_vertical(self):
        out = pca_crop(_rgb_from_mask(_mask(120, 120, (50, 70, 10, 110))))
        h, w = out.shape[:2]
        assert h > w

    def test_apex_ends_up_at_top(self):
        # Треугольник: широкое основание сверху, узкий апекс снизу — pca_crop
        # обязан перевернуть, чтобы апекс оказался наверху.
        h, w = 100, 80
        m = np.zeros((h, w), dtype=np.uint8)
        for y in range(h):
            half = max(1, int((h - y) * 0.35))
            cx = w // 2
            m[y, max(0, cx - half):min(w, cx + half)] = 255

        out = pca_crop(_rgb_from_mask(m))
        out_mask = out.sum(axis=2) > 0
        mid = out_mask.shape[0] // 2
        # Масса внизу = апекс наверху.
        assert out_mask[mid:].sum() > out_mask[:mid].sum()

    def test_too_few_pixels_falls_back_to_axis_bbox(self):
        # < 8 пикселей → срабатывает fallback на _axis_bbox без warpAffine.
        m = np.zeros((30, 30), dtype=np.uint8)
        m[10, 10] = m[10, 11] = m[11, 10] = 255
        out = pca_crop(_rgb_from_mask(m))
        assert out.shape[0] >= 1 and out.shape[1] >= 1
        assert (out.sum(axis=2) > 0).any()

    def test_tight_crop_rows_have_pixels(self):
        out = pca_crop(_rgb_from_mask(_mask(100, 100, (10, 90, 30, 70))))
        active = out.sum(axis=2) > 0
        assert active[0].any() or active[-1].any()


class TestDedupeAndTransformLeaf:
    def test_empty_input(self):
        assert dedupe_and_transform_leaf(np.zeros((50, 50, 3), dtype=np.uint8), []) == []

    def test_two_identical_dedupe(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        m = _mask(100, 100, (20, 80, 20, 80))
        out = dedupe_and_transform_leaf(img, [m, m.copy()])
        assert len(out) == 1

    def test_disjoint_kept(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        m1 = _mask(100, 100, (5, 25, 5, 25))
        m2 = _mask(100, 100, (60, 80, 60, 80))
        assert len(dedupe_and_transform_leaf(img, [m1, m2])) == 2

    def test_low_iou_kept(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        m1 = _mask(100, 100, (10, 50, 10, 50))
        m2 = _mask(100, 100, (40, 80, 40, 80))
        assert len(dedupe_and_transform_leaf(img, [m1, m2], iou_threshold=0.9)) == 2

    def test_high_iou_keeps_larger(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        big = _mask(100, 100, (10, 90, 10, 90))
        small = _mask(100, 100, (12, 88, 12, 88))
        out = dedupe_and_transform_leaf(img, [small, big], iou_threshold=0.9)
        assert len(out) == 1
        kept_mask = out[0][0]
        assert (kept_mask > 0).sum() > 5000

    def test_returns_uint8_binary_mask_and_rgb_crop(self):
        img = np.zeros((80, 80, 3), dtype=np.uint8)
        out = dedupe_and_transform_leaf(img, [_mask(80, 80, (10, 70, 10, 70))])
        cleaned, crop = out[0]
        assert cleaned.dtype == np.uint8
        assert set(np.unique(cleaned)).issubset({0, 255})
        assert crop.ndim == 3 and crop.shape[2] == 3


class TestTransformLeaf:
    def test_combined_preprocess_and_pca(self):
        img = np.zeros((150, 150, 3), dtype=np.uint8)
        img[40:120, 30:80] = 200
        img[70:80, 50:60] = 0
        out = transform_leaf(img)
        assert out.ndim == 3 and out.shape[2] == 3
        assert (out.sum(axis=2) > 0).any()


class TestAutoSegment:
    def test_upscales_to_full_resolution(self):
        h, w = 400, 400
        image = np.zeros((h, w, 3), dtype=np.uint8)
        image[100:300, 150:250] = 200

        # predict_all внутри auto_segment получает картинку 200×200 (DOWNSCALE=2).
        small_mask = _mask(h // 2, w // 2, (50, 150, 75, 125))
        predictor = MagicMock()
        predictor.predict_all = MagicMock(return_value=[small_mask])

        out = auto_segment(image, predictor)
        assert len(out) == 1
        full_mask, full_crop = out[0]
        assert full_mask.shape == (h, w)
        assert full_mask.dtype == np.uint8
        assert set(np.unique(full_mask)).issubset({0, 255})
        assert full_crop.ndim == 3 and full_crop.shape[2] == 3

    def test_empty_returns_empty(self):
        predictor = MagicMock()
        predictor.predict_all = MagicMock(return_value=[])
        assert auto_segment(np.zeros((200, 200, 3), dtype=np.uint8), predictor) == []
