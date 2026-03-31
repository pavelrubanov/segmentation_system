"""Тесты модуля core/leaf_measure.py."""
import numpy as np
import pytest

from core.leaf_measure import (
    SectionWidth,
    _to_tuple,
    measure_leaf,
    save_measurement_visualization,
)


# ── Вспомогательные функции ───────────────────────────────────────────────────


def make_rect_mask(h: int, w: int) -> np.ndarray:
    """Прямоугольная маска (все пиксели = 255)."""
    return np.full((h, w), 255, dtype=np.uint8)


def make_ellipse_mask(h: int, w: int) -> np.ndarray:
    """Эллиптическая маска, вписанная в прямоугольник h×w."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx = h // 2, w // 2
    ry, rx = h // 2 - 2, w // 2 - 2
    Y, X = np.ogrid[:h, :w]
    mask[((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1] = 255
    return mask


# ── measure_leaf ──────────────────────────────────────────────────────────────


class TestMeasureLeafBasic:

    def test_empty_mask_raises(self):
        """Пустая маска → ValueError."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="Контур не найден"):
            measure_leaf(mask)

    def test_rectangle_length_and_width(self):
        """Горизонтальный прямоугольник 100×20: длина=99, ширина=19."""
        mask = make_rect_mask(h=20, w=100)
        m = measure_leaf(mask)
        assert abs(m.length_px - 99) < 0.5
        assert abs(m.width_px - 19) < 0.5

    def test_length_ge_width(self):
        """Длина всегда ≥ ширины."""
        mask = make_rect_mask(h=30, w=150)
        m = measure_leaf(mask)
        assert m.length_px >= m.width_px

    def test_square_length_equals_width(self):
        """Для квадрата длина ≈ ширина (симметрия PCA)."""
        mask = make_rect_mask(h=100, w=100)
        m = measure_leaf(mask)
        assert abs(m.length_px - m.width_px) < 1.0

    def test_small_mask_does_not_raise(self):
        """Маленькая маска 5×15 не вызывает исключения."""
        mask = make_rect_mask(h=5, w=15)
        m = measure_leaf(mask)
        assert m.length_px > 0


class TestMeasureLeafMm:

    def test_no_mm_per_px_gives_none(self):
        """Без mm_per_px поля _mm = None."""
        mask = make_rect_mask(h=20, w=100)
        m = measure_leaf(mask)
        assert m.length_mm is None
        assert m.width_mm is None

    def test_mm_conversion_length(self):
        """length_mm = length_px × mm_per_px."""
        mask = make_rect_mask(h=20, w=100)
        scale = 0.1
        m = measure_leaf(mask, mm_per_px=scale)
        assert m.length_mm is not None
        assert abs(m.length_mm - m.length_px * scale) < 1e-6

    def test_mm_conversion_width(self):
        """width_mm = width_px × mm_per_px."""
        mask = make_rect_mask(h=20, w=100)
        scale = 0.1
        m = measure_leaf(mask, mm_per_px=scale)
        assert m.width_mm is not None
        assert abs(m.width_mm - m.width_px * scale) < 1e-6


class TestMeasureLeafFractions:

    def test_default_seven_fractions(self):
        """По умолчанию возвращаются 7 сечений."""
        mask = make_rect_mask(h=20, w=200)
        m = measure_leaf(mask)
        assert len(m.widths) == 7

    def test_default_fractions_keys(self):
        """Ключи widths совпадают со стандартными дробями."""
        mask = make_rect_mask(h=20, w=200)
        m = measure_leaf(mask)
        expected = {0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875}
        assert set(m.widths.keys()) == expected

    def test_custom_fractions_keys(self):
        """Кастомные дроби → ровно эти ключи."""
        mask = make_rect_mask(h=20, w=200)
        fracs = (0.25, 0.5, 0.75)
        m = measure_leaf(mask, fractions=fracs)
        assert set(m.widths.keys()) == set(fracs)

    def test_single_fraction(self):
        """Одна дробь — один элемент в widths."""
        mask = make_rect_mask(h=20, w=200)
        m = measure_leaf(mask, fractions=(0.5,))
        assert len(m.widths) == 1
        assert 0.5 in m.widths


class TestMeasureLeafSections:

    def test_section_width_positive_for_rect(self):
        """Ширина в каждом сечении > 0 для прямоугольника."""
        mask = make_rect_mask(h=40, w=200)
        m = measure_leaf(mask)
        for sec in m.widths.values():
            assert sec.width_px > 0

    def test_section_width_le_total_width(self):
        """Ширина в сечении не превышает PCA-ширину более чем на 1 пиксель
        (погрешность дискретизации сканирующей сетки)."""
        mask = make_rect_mask(h=40, w=200)
        m = measure_leaf(mask)
        for sec in m.widths.values():
            assert sec.width_px <= m.width_px + 1.0

    def test_section_p0_p1_not_none(self):
        """p0 и p1 не None для прямоугольника."""
        mask = make_rect_mask(h=60, w=200)
        m = measure_leaf(mask)
        for sec in m.widths.values():
            assert sec.p0 is not None
            assert sec.p1 is not None

    def test_section_mm_with_scale(self):
        """С mm_per_px = 0.05 ширины в сечениях тоже конвертируются."""
        mask = make_rect_mask(h=40, w=200)
        m = measure_leaf(mask, mm_per_px=0.05)
        for sec in m.widths.values():
            if sec.p0 is not None:
                assert sec.width_mm is not None
                assert abs(sec.width_mm - sec.width_px * 0.05) < 1e-5

    def test_section_mm_none_without_scale(self):
        """Без масштаба width_mm = None."""
        mask = make_rect_mask(h=40, w=200)
        m = measure_leaf(mask)
        for sec in m.widths.values():
            assert sec.width_mm is None

    def test_ellipse_midpoint_widest(self):
        """У эллипса ширина в середине (0.5) ≥ ширины у краёв."""
        mask = make_ellipse_mask(h=80, w=200)
        m = measure_leaf(mask)
        mid = m.widths[0.5].width_px
        lo = m.widths[0.125].width_px
        hi = m.widths[0.875].width_px
        assert mid >= lo - 1.0
        assert mid >= hi - 1.0


class TestMeasureLeafPca:

    def test_axes_orthogonal(self):
        """Оси u и v перпендикулярны: dot ≈ 0."""
        mask = make_rect_mask(h=30, w=150)
        m = measure_leaf(mask)
        dot = m.axis_u[0] * m.axis_v[0] + m.axis_u[1] * m.axis_v[1]
        assert abs(dot) < 1e-5

    def test_axes_unit_length(self):
        """Оси u и v — единичные векторы."""
        mask = make_rect_mask(h=30, w=150)
        m = measure_leaf(mask)
        norm_u = (m.axis_u[0] ** 2 + m.axis_u[1] ** 2) ** 0.5
        norm_v = (m.axis_v[0] ** 2 + m.axis_v[1] ** 2) ** 0.5
        assert abs(norm_u - 1.0) < 1e-5
        assert abs(norm_v - 1.0) < 1e-5

    def test_mean_xy_inside_mask(self):
        """Центр масс находится внутри маски."""
        h, w = 40, 120
        mask = make_rect_mask(h=h, w=w)
        m = measure_leaf(mask)
        cx, cy = m.mean_xy
        assert 0 <= cx < w
        assert 0 <= cy < h

    def test_length_endpoints_count(self):
        """length_endpoints содержит ровно две точки."""
        mask = make_rect_mask(h=20, w=100)
        m = measure_leaf(mask)
        assert len(m.length_endpoints) == 2
        for ep in m.length_endpoints:
            assert len(ep) == 2
            assert isinstance(ep[0], float)

    def test_width_endpoints_count(self):
        """width_endpoints содержит ровно две точки."""
        mask = make_rect_mask(h=20, w=100)
        m = measure_leaf(mask)
        assert len(m.width_endpoints) == 2


class TestMeasureLeafContour:

    def test_contour_nonempty(self):
        """Контур непустой."""
        mask = make_rect_mask(h=40, w=100)
        m = measure_leaf(mask)
        assert m.contour.shape[0] > 0

    def test_contour_shape_columns(self):
        """Контур имеет форму (N, 2)."""
        mask = make_rect_mask(h=40, w=100)
        m = measure_leaf(mask)
        assert m.contour.ndim == 2
        assert m.contour.shape[1] == 2

    def test_contour_within_image_bounds(self):
        """Точки контура находятся внутри маски."""
        h, w = 40, 100
        mask = make_rect_mask(h=h, w=w)
        m = measure_leaf(mask)
        assert np.all(m.contour[:, 0] >= 0) and np.all(m.contour[:, 0] < w)
        assert np.all(m.contour[:, 1] >= 0) and np.all(m.contour[:, 1] < h)


# ── _to_tuple ─────────────────────────────────────────────────────────────────


class TestToTuple:

    def test_basic_float_array(self):
        arr = np.array([1.5, 2.5])
        assert _to_tuple(arr) == (1.5, 2.5)

    def test_returns_python_floats(self):
        arr = np.array([3, 4], dtype=np.int32)
        result = _to_tuple(arr)
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_negative_values(self):
        arr = np.array([-3.14, -2.71])
        x, y = _to_tuple(arr)
        assert abs(x - (-3.14)) < 1e-5
        assert abs(y - (-2.71)) < 1e-5

    def test_zero_values(self):
        arr = np.array([0.0, 0.0])
        assert _to_tuple(arr) == (0.0, 0.0)


# ── SectionWidth и LeafMetrics ────────────────────────────────────────────────


class TestSectionWidth:

    def test_all_fields_set(self):
        sec = SectionWidth(
            width_px=10.0,
            width_mm=1.0,
            p0=(0.0, 0.0),
            p1=(10.0, 0.0),
            center=(5.0, 5.0),
        )
        assert sec.width_px == 10.0
        assert sec.width_mm == 1.0
        assert sec.p0 == (0.0, 0.0)
        assert sec.p1 == (10.0, 0.0)
        assert sec.center == (5.0, 5.0)

    def test_optional_fields_none(self):
        sec = SectionWidth(width_px=0.0, width_mm=None, p0=None, p1=None, center=(0.0, 0.0))
        assert sec.width_mm is None
        assert sec.p0 is None
        assert sec.p1 is None


# ── save_measurement_visualization ────────────────────────────────────────────


class TestSaveVisualization:

    def test_creates_png_file(self, tmp_path):
        """Функция создаёт файл по заданному пути."""
        mask = make_rect_mask(h=40, w=100)
        m = measure_leaf(mask)
        out = tmp_path / "vis.png"
        save_measurement_visualization(mask, m, str(out))
        assert out.exists()

    def test_output_file_nonempty(self, tmp_path):
        """Созданный файл не пустой."""
        mask = make_rect_mask(h=40, w=100)
        m = measure_leaf(mask)
        out = tmp_path / "vis.png"
        save_measurement_visualization(mask, m, str(out))
        assert out.stat().st_size > 0

    def test_with_mm_scale(self, tmp_path):
        """Визуализация с масштабом не вызывает исключений."""
        mask = make_rect_mask(h=40, w=100)
        m = measure_leaf(mask, mm_per_px=0.05)
        out = tmp_path / "vis_mm.png"
        save_measurement_visualization(mask, m, str(out))
        assert out.exists()

    def test_with_ellipse(self, tmp_path):
        """Визуализация эллиптической маски не вызывает исключений."""
        mask = make_ellipse_mask(h=80, w=200)
        m = measure_leaf(mask)
        out = tmp_path / "vis_ellipse.png"
        save_measurement_visualization(mask, m, str(out))
        assert out.exists()
