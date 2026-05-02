"""Тесты API core/leaf_measure.py.

Точность измерений на реальных листах --- в `test_biologist_metrics.py`.
Здесь только проверки API и краевые случаи: пустой кроп, формирование
fractions, имена колонок, сохранение визуализации.
"""
import numpy as np
import pytest

from core.leaf_measure import (
    LeafMetrics,
    SectionWidth,
    build_fractions,
    column_names_for,
    measure_leaf,
    save_measurement_visualization,
)


def _white_crop(h=80, w=40):
    """Кроп с «листом» по всему полю --- чёрного фона нет."""
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _empty_crop(h=80, w=40):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ── measure_leaf API ─────────────────────────────────────────────────────────


class TestMeasureLeafApi:
    def test_empty_crop_raises(self):
        with pytest.raises(ValueError, match="пуст"):
            measure_leaf(_empty_crop())

    def test_default_seven_fractions(self):
        assert len(measure_leaf(_white_crop()).widths) == 7

    def test_custom_fractions_keys(self):
        m = measure_leaf(_white_crop(), fractions=(0.25, 0.5, 0.75))
        assert set(m.widths.keys()) == {0.25, 0.5, 0.75}

    def test_default_unit_is_px(self):
        assert measure_leaf(_white_crop()).unit == "px"

    def test_custom_unit_passed_through(self):
        assert measure_leaf(_white_crop(), unit="мм", scale=0.1).unit == "мм"

    def test_scale_multiplies_length_and_width(self):
        m_px = measure_leaf(_white_crop())
        m_mm = measure_leaf(_white_crop(), unit="мм", scale=0.1)
        assert m_mm.length == pytest.approx(m_px.length * 0.1)
        assert m_mm.width == pytest.approx(m_px.width * 0.1)

    def test_section_widths_scaled(self):
        m_px = measure_leaf(_white_crop())
        m_mm = measure_leaf(_white_crop(), unit="мм", scale=0.1)
        for f, sec_px in m_px.widths.items():
            assert m_mm.widths[f].width == pytest.approx(sec_px.width * 0.1)

    def test_section_endpoints_set_for_solid_crop(self):
        for sec in measure_leaf(_white_crop()).widths.values():
            assert sec.p0 is not None
            assert sec.p1 is not None

    def test_section_returns_zero_for_empty_row(self):
        # Кроп с двумя горизонтальными «полосками» и пустотой ровно посередине.
        # bbox активных пикселей: y=0..99, fraction=0.5 → row=49 → попадает
        # в дыру → SectionWidth.p0/p1 = None, width = 0.
        h, w = 100, 40
        crop = np.zeros((h, w, 3), dtype=np.uint8)
        crop[:40, :] = 255
        crop[60:, :] = 255

        sec = measure_leaf(crop, fractions=(0.5,)).widths[0.5]
        assert sec.p0 is None
        assert sec.p1 is None
        assert sec.width == 0.0


# ── build_fractions ──────────────────────────────────────────────────────────


class TestBuildFractions:
    def test_n3(self):
        assert build_fractions(3) == (0.25, 0.5, 0.75)

    def test_n7_matches_default_measure_fractions(self):
        # build_fractions(7) должен совпадать с дефолтным fractions для measure_leaf
        assert build_fractions(7) == (1 / 8, 2 / 8, 3 / 8, 4 / 8, 5 / 8, 6 / 8, 7 / 8)

    @pytest.mark.parametrize("n", [1, 3, 5, 7, 10])
    def test_all_inside_open_unit_interval(self, n):
        assert all(0.0 < f < 1.0 for f in build_fractions(n))

    @pytest.mark.parametrize("n", [1, 5, 10])
    def test_count_matches_n(self, n):
        assert len(build_fractions(n)) == n


# ── column_names_for ─────────────────────────────────────────────────────────


class TestColumnNamesFor:
    def test_basic(self):
        assert column_names_for([0.5], "мм") == ["image", "length_мм", "width_мм", "width_0.500_мм"]

    def test_seven_fractions(self):
        cols = column_names_for(build_fractions(7), "px")
        assert cols[0] == "image"
        assert "length_px" in cols
        assert "width_0.125_px" in cols
        assert "width_0.875_px" in cols
        assert len(cols) == 1 + 2 + 7   # image + length + width + 7 sections


# ── Датаклассы ───────────────────────────────────────────────────────────────


class TestDataclasses:
    def test_section_width_optional_fields(self):
        sec = SectionWidth(width=0.0, p0=None, p1=None, center=(0.0, 0.0))
        assert sec.p0 is None and sec.p1 is None

    def test_leaf_metrics_basic(self):
        m = LeafMetrics(length=100.0, width=20.0, unit="px", widths={})
        assert m.length == 100.0 and m.unit == "px"


# ── save_measurement_visualization ───────────────────────────────────────────


class TestSaveVisualization:
    def test_creates_nonempty_png(self, tmp_path):
        crop = _white_crop()
        out = tmp_path / "vis.png"
        save_measurement_visualization(crop, measure_leaf(crop), str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_with_scale(self, tmp_path):
        crop = _white_crop()
        out = tmp_path / "vis_mm.png"
        m = measure_leaf(crop, unit="мм", scale=0.1)
        save_measurement_visualization(crop, m, str(out))
        assert out.exists()
