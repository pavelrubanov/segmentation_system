"""Тесты core/io.py: чтение картинок, русское склонение, форматирование."""
import numpy as np
import pytest
from PIL import Image

from core.io import format_count, pluralize_ru, read_rgb


class TestPluralizeRu:
    """Подбор русской словоформы по числу: (1, 2, 5)."""

    FORMS = ("маска", "маски", "масок")

    @pytest.mark.parametrize("n,expected", [
        (1, "маска"), (21, "маска"), (101, "маска"), (1001, "маска"),
        (2, "маски"), (3, "маски"), (4, "маски"),
        (22, "маски"), (23, "маски"), (104, "маски"),
        (5, "масок"), (10, "масок"), (20, "масок"), (25, "масок"), (100, "масок"),
        # Ключевое исключение: 11–14 → many, несмотря на mod10 ∈ {1..4}
        (11, "масок"), (12, "масок"), (13, "масок"), (14, "масок"),
        (111, "масок"), (112, "масок"), (114, "масок"),
        (0, "масок"),
    ])
    def test_form(self, n, expected):
        assert pluralize_ru(n, self.FORMS) == expected


class TestFormatCount:
    FORMS = ("маска", "маски", "масок")

    @pytest.mark.parametrize("n,expected", [
        (1, "1 маска"),
        (3, "3 маски"),
        (5, "5 масок"),
        (11, "11 масок"),
        (21, "21 маска"),
    ])
    def test_format(self, n, expected):
        assert format_count(n, self.FORMS) == expected


class TestReadRgb:
    def test_rgb_png(self, tmp_path):
        path = tmp_path / "rgb.png"
        Image.fromarray(np.full((40, 60, 3), 200, dtype=np.uint8)).save(path)
        img = read_rgb(path)
        assert img.shape == (40, 60, 3)
        assert img.dtype == np.uint8

    def test_jpeg(self, tmp_path):
        path = tmp_path / "x.jpg"
        Image.fromarray(np.full((40, 60, 3), 100, dtype=np.uint8)).save(path)
        img = read_rgb(path)
        assert img.shape == (40, 60, 3)

    def test_grayscale_converted_to_rgb(self, tmp_path):
        path = tmp_path / "gray.png"
        Image.fromarray(np.full((20, 30), 100, dtype=np.uint8), mode="L").save(path)
        img = read_rgb(path)
        assert img.shape == (20, 30, 3)
        np.testing.assert_array_equal(img[:, :, 0], img[:, :, 1])
        np.testing.assert_array_equal(img[:, :, 1], img[:, :, 2])

    def test_rgba_alpha_dropped(self, tmp_path):
        path = tmp_path / "rgba.png"
        rgba = np.zeros((30, 30, 4), dtype=np.uint8)
        rgba[..., :3] = 200
        rgba[..., 3] = 128
        Image.fromarray(rgba, mode="RGBA").save(path)
        img = read_rgb(path)
        assert img.shape == (30, 30, 3)

    def test_palette_converted(self, tmp_path):
        path = tmp_path / "p.png"
        Image.fromarray(np.full((20, 20, 3), 50, dtype=np.uint8)).convert("P").save(path)
        img = read_rgb(path)
        assert img.shape == (20, 20, 3)

    def test_accepts_str_and_path(self, tmp_path):
        path = tmp_path / "x.png"
        Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(path)
        np.testing.assert_array_equal(read_rgb(str(path)), read_rgb(path))
