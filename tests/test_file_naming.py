"""core/file_naming.py — единая точка правды для имён артефактов."""
import pytest

from core.file_naming import (
    CROP_INFIX,
    SUPPORTED_INPUT_EXTENSIONS,
    crop_file_filter,
    crop_filename,
    image_file_filter,
    is_crop_filename,
    mask_filename,
    vis_filename,
)


class TestCropFilename:
    def test_basic(self):
        assert crop_filename("IMG_4321", 1) == "IMG_4321_leaf_crop_0001.png"

    @pytest.mark.parametrize("idx,expected_pad", [(7, "0007"), (42, "0042"), (1234, "1234")])
    def test_index_padded_to_4_digits(self, idx, expected_pad):
        assert crop_filename("foo", idx) == f"foo_leaf_crop_{expected_pad}.png"

    def test_extension_always_png(self):
        assert crop_filename("a.jpg", 1).endswith(".png")


class TestMaskFilename:
    def test_basic(self):
        assert mask_filename("IMG_4321", 7) == "IMG_4321_mask_0007.png"

    def test_no_collision_with_crop(self):
        assert crop_filename("foo", 5) != mask_filename("foo", 5)


class TestVisFilename:
    def test_basic(self):
        assert vis_filename("IMG_4321_leaf_crop_0001") == "IMG_4321_leaf_crop_0001_vis.png"


class TestIsCropFilename:
    @pytest.mark.parametrize("stem,idx", [("foo", 1), ("IMG_4321", 9999), ("a_b_c", 42)])
    def test_round_trip(self, stem, idx):
        assert is_crop_filename(crop_filename(stem, idx))

    def test_mask_not_recognized(self):
        assert not is_crop_filename(mask_filename("foo", 1))

    @pytest.mark.parametrize("name", ["random.jpg", "foo_crop.png", "leaf_crop.png", ""])
    def test_arbitrary_names_not_recognized(self, name):
        # Маркер ищется с подчёркиваниями с обеих сторон — без них не считается.
        assert not is_crop_filename(name)


class TestFilters:
    def test_crop_filter_lists_all_extensions(self):
        f = crop_file_filter()
        for ext in SUPPORTED_INPUT_EXTENSIONS:
            assert f"_{CROP_INFIX}_*.{ext}" in f

    def test_image_filter_lists_all_extensions(self):
        f = image_file_filter()
        for ext in SUPPORTED_INPUT_EXTENSIONS:
            assert f"*.{ext}" in f


class TestCrossPackageConsistency:
    # classifier/mobilenet.py хранит свою копию CROP_INFIX — если разъедутся,
    # сканер классификатора перестанет видеть кропы и тихо.
    def test_classifier_crop_infix_matches_core(self):
        from classifier.mobilenet import _CROP_INFIX
        assert _CROP_INFIX == CROP_INFIX
