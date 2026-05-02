"""Тесты core/sources.py: disk_source и AutoSegmentSource (с моками)."""
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from core import sources as sources_mod
from core.file_naming import crop_filename, mask_filename
from core.sources import AutoSegmentSource, disk_source


def _save_png(path, h=40, w=30):
    Image.fromarray(np.full((h, w, 3), 100, dtype=np.uint8)).save(path)


class TestDiskSource:
    def test_yields_one_crop(self, tmp_path):
        path = tmp_path / "img.png"; _save_png(path)
        items = list(disk_source(str(path)))
        assert len(items) == 1

    def test_name_is_stem(self, tmp_path):
        path = tmp_path / "leaf_x.png"; _save_png(path)
        assert list(disk_source(str(path)))[0].name == "leaf_x"

    def test_crop_is_rgb_uint8(self, tmp_path):
        path = tmp_path / "x.png"; _save_png(path)
        crop = list(disk_source(str(path)))[0].crop
        assert crop.ndim == 3 and crop.shape[2] == 3
        assert crop.dtype == np.uint8


class TestAutoSegmentSourceInit:
    def test_creates_output_dirs(self, tmp_path):
        AutoSegmentSource(
            predictor=MagicMock(), classifier=MagicMock(),
            crops_dir=tmp_path / "crops", masks_dir=tmp_path / "masks",
        )
        assert (tmp_path / "crops").is_dir()
        assert (tmp_path / "masks").is_dir()


class TestAutoSegmentSourceCall:
    def _src(self, tmp_path, classifier):
        return AutoSegmentSource(
            predictor=MagicMock(), classifier=classifier,
            crops_dir=tmp_path / "crops", masks_dir=tmp_path / "masks",
        )

    def _patch_auto_segment(self, monkeypatch, pairs):
        monkeypatch.setattr(sources_mod, "auto_segment", lambda img, p: pairs)

    def test_filters_only_good_leaf(self, tmp_path, monkeypatch):
        h, w = 30, 40
        crop = np.full((h, w, 3), 100, dtype=np.uint8)
        mask = np.full((h, w), 255, dtype=np.uint8)
        self._patch_auto_segment(monkeypatch, [(mask, crop)] * 3)

        clf = MagicMock()
        clf.classify_batch = MagicMock(return_value=[
            ("good_leaf", 0.9), ("non_leaf", 0.8), ("good_leaf", 0.7),
        ])

        img_path = tmp_path / "src.png"; _save_png(img_path)
        items = list(self._src(tmp_path, clf)(str(img_path)))

        assert len(items) == 2
        assert len(list((tmp_path / "crops").glob("*.png"))) == 2
        assert len(list((tmp_path / "masks").glob("*.png"))) == 2

    def test_filenames_use_convention(self, tmp_path, monkeypatch):
        h, w = 30, 40
        self._patch_auto_segment(
            monkeypatch,
            [(np.full((h, w), 255, dtype=np.uint8), np.full((h, w, 3), 100, dtype=np.uint8))],
        )
        clf = MagicMock()
        clf.classify_batch = MagicMock(return_value=[("good_leaf", 0.9)])

        img_path = tmp_path / "leaf123.png"; _save_png(img_path)
        list(self._src(tmp_path, clf)(str(img_path)))

        assert (tmp_path / "crops" / crop_filename("leaf123", 1)).exists()
        assert (tmp_path / "masks" / mask_filename("leaf123", 1)).exists()

    def test_indices_only_increment_for_good(self, tmp_path, monkeypatch):
        # Должны быть имена _0001 и _0002 (только good), не _0001 и _0003.
        h, w = 30, 40
        self._patch_auto_segment(
            monkeypatch,
            [(np.full((h, w), 255, dtype=np.uint8), np.full((h, w, 3), 100, dtype=np.uint8))] * 3,
        )
        clf = MagicMock()
        clf.classify_batch = MagicMock(return_value=[
            ("non_leaf", 0.5), ("good_leaf", 0.9), ("good_leaf", 0.8),
        ])

        img_path = tmp_path / "x.png"; _save_png(img_path)
        items = list(self._src(tmp_path, clf)(str(img_path)))

        names = sorted(it.name for it in items)
        assert names == ["x_leaf_crop_0001", "x_leaf_crop_0002"]

    def test_empty_auto_segment_yields_nothing(self, tmp_path, monkeypatch):
        self._patch_auto_segment(monkeypatch, [])
        img_path = tmp_path / "x.png"; _save_png(img_path)
        # Classifier даже не должен вызываться
        clf = MagicMock()
        items = list(self._src(tmp_path, clf)(str(img_path)))
        assert items == []
        clf.classify_batch.assert_not_called()
