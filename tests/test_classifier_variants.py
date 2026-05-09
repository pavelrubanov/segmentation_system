"""classifier/variants.py: 6 детерминированных аугментаций."""
import numpy as np
import pytest

from classifier.variants import IDENTITY_VARIANT, VARIANTS, Variant, apply_variant


@pytest.fixture
def img():
    # Градиент по R/G + плоский B — чтобы аугментации сдвигали средние.
    g = np.zeros((32, 32, 3), dtype=np.uint8)
    for x in range(32):
        g[:, x, 0] = x * 8
        g[:, x, 1] = (31 - x) * 8
    g[:, :, 2] = 100
    return g


class TestVariantsList:
    def test_six_variants(self):
        assert len(VARIANTS) == 6

    def test_identity_first(self):
        assert VARIANTS[0] == IDENTITY_VARIANT

    def test_three_orig_three_hflip(self):
        assert sum(v.hflip for v in VARIANTS) == 3


class TestApplyVariant:
    def test_identity_returns_input_object(self, img):
        # Identity-вариант должен отдать тот же объект, без копии.
        assert apply_variant(img, IDENTITY_VARIANT) is img

    def test_hflip_only(self, img):
        out = apply_variant(img, Variant(True, 1.0, 1.0, 1.0))
        np.testing.assert_array_equal(out, img[:, ::-1, :])

    def test_brightness_up_increases_mean(self, img):
        assert apply_variant(img, Variant(False, 1.5, 1.0, 1.0)).mean() > img.mean()

    def test_brightness_down_decreases_mean(self, img):
        assert apply_variant(img, Variant(False, 0.5, 1.0, 1.0)).mean() < img.mean()

    def test_shape_and_dtype_preserved_for_all(self, img):
        for v in VARIANTS:
            out = apply_variant(img, v)
            assert out.shape == img.shape
            assert out.dtype == np.uint8
