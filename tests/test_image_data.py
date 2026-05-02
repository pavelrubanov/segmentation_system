"""Тесты core/image_data.py."""
import numpy as np
from PIL import Image

from core.image_data import ImageWithMasks


class TestImageWithMasks:
    def test_default_name_empty(self):
        assert ImageWithMasks(path="/tmp/x.jpg").name == ""

    def test_custom_name(self):
        assert ImageWithMasks(path="/tmp/x.jpg", name="leaf.jpg").name == "leaf.jpg"

    def test_default_mask_entries_empty(self):
        assert ImageWithMasks(path="/tmp/x.jpg").mask_entries == []

    def test_mask_entries_isolated_between_instances(self):
        # default_factory защищает от mutable-default
        a = ImageWithMasks(path="/tmp/a.jpg")
        b = ImageWithMasks(path="/tmp/b.jpg")
        a.mask_entries.append("x")
        assert b.mask_entries == []


class TestLoadImageFromDisk:
    def test_returns_rgb_uint8(self, tmp_path):
        p = tmp_path / "test.png"
        Image.fromarray(np.full((40, 60, 3), 123, dtype=np.uint8)).save(p)
        loaded = ImageWithMasks(path=str(p)).load_image_from_disk()
        assert loaded.shape == (40, 60, 3)
        assert loaded.dtype == np.uint8

    def test_grayscale_input_converted_to_rgb(self, tmp_path):
        p = tmp_path / "gray.png"
        Image.fromarray(np.full((20, 30), 50, dtype=np.uint8), mode="L").save(p)
        loaded = ImageWithMasks(path=str(p)).load_image_from_disk()
        assert loaded.shape == (20, 30, 3)

    def test_rgba_input_alpha_dropped(self, tmp_path):
        p = tmp_path / "rgba.png"
        rgba = np.zeros((10, 10, 4), dtype=np.uint8)
        rgba[..., 3] = 200
        Image.fromarray(rgba, mode="RGBA").save(p)
        loaded = ImageWithMasks(path=str(p)).load_image_from_disk()
        assert loaded.shape == (10, 10, 3)
