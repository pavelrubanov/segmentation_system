"""core/predictor.py с замоканными SAM-классами.

Подменяем атрибуты уже импортированного core.predictor через monkeypatch —
не sys.modules, иначе мок утечёт в интеграционные тесты, которым нужен SAM.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest

import core.predictor as cp


_mock_sam_instance = MagicMock()
_mock_predictor_cls = MagicMock()
_mock_amg_cls = MagicMock()
_mock_registry = {"vit_t": MagicMock(return_value=_mock_sam_instance)}


@pytest.fixture(autouse=True)
def _patch_sam(monkeypatch):
    monkeypatch.setattr(cp, "sam_model_registry", _mock_registry)
    monkeypatch.setattr(cp, "SamPredictor", _mock_predictor_cls)
    monkeypatch.setattr(cp, "SamAutomaticMaskGenerator", _mock_amg_cls)


@pytest.fixture()
def predictor():
    _mock_registry["vit_t"].reset_mock()
    _mock_predictor_cls.reset_mock()
    return cp.Predictor(checkpoint_path="fake_weights.pt")


def _fake_predict(h: int, w: int, best_idx: int = 1):
    masks = np.zeros((3, h, w), dtype=bool)
    masks[best_idx] = True
    scores = np.array([0.3, 0.95, 0.6])
    scores[best_idx] = max(scores) + 0.1
    return masks, scores, None


class TestPredictorInit:
    def test_registry_called_with_checkpoint(self, predictor):
        _mock_registry["vit_t"].assert_called_once_with(checkpoint="fake_weights.pt")

    def test_sam_predictor_instantiated(self, predictor):
        _mock_predictor_cls.assert_called_once()

    def test_sam_predictor_attribute_set(self, predictor):
        assert predictor.sam_predictor is not None


class TestPredictNoPrompts:
    def test_no_points_no_box_returns_none(self, predictor):
        assert predictor.predict([], [], None) is None

    def test_sam_predict_not_called_when_no_prompts(self, predictor):
        predictor.predict([], [], None)
        predictor.sam_predictor.predict.assert_not_called()


class TestPredictWithPrompts:
    def test_positive_point_returns_mask(self, predictor):
        predictor.sam_predictor.predict.return_value = _fake_predict(50, 60)
        result = predictor.predict(pos_points=[(25, 25)], neg_points=[], box=None)
        assert result is not None

    def test_output_dtype_uint8(self, predictor):
        predictor.sam_predictor.predict.return_value = _fake_predict(50, 60)
        result = predictor.predict(pos_points=[(25, 25)], neg_points=[], box=None)
        assert result.dtype == np.uint8

    def test_output_values_0_or_255(self, predictor):
        predictor.sam_predictor.predict.return_value = _fake_predict(50, 60)
        result = predictor.predict(pos_points=[(25, 25)], neg_points=[], box=None)
        assert set(np.unique(result)).issubset({0, 255})

    def test_output_shape_matches_input(self, predictor):
        h, w = 64, 96
        predictor.sam_predictor.predict.return_value = _fake_predict(h, w)
        result = predictor.predict(pos_points=[(30, 30)], neg_points=[], box=None)
        assert result.shape == (h, w)

    def test_best_mask_selected_by_score(self, predictor):
        h, w = 40, 40
        masks = np.zeros((3, h, w), dtype=bool)
        masks[0, :, :] = False
        masks[1, :10, :10] = True
        masks[2, :, :] = True
        scores = np.array([0.4, 0.6, 0.95])
        predictor.sam_predictor.predict.return_value = (masks, scores, None)

        result = predictor.predict(pos_points=[(20, 20)], neg_points=[], box=None)
        assert np.all(result == 255)

    def test_box_only_prompt(self, predictor):
        predictor.sam_predictor.predict.return_value = _fake_predict(80, 100)
        result = predictor.predict(pos_points=[], neg_points=[], box=[10, 10, 90, 70])
        assert result is not None

    def test_negative_points_label_zero(self, predictor):
        predictor.sam_predictor.predict.return_value = _fake_predict(50, 50)
        predictor.predict(
            pos_points=[(10, 10)],
            neg_points=[(40, 40)],
            box=None,
        )
        kwargs = predictor.sam_predictor.predict.call_args.kwargs
        labels = list(kwargs["point_labels"])
        assert labels == [1, 0]

    def test_point_coords_combined(self, predictor):
        predictor.sam_predictor.predict.return_value = _fake_predict(50, 50)
        predictor.predict(
            pos_points=[(10, 20)],
            neg_points=[(30, 40)],
            box=None,
        )
        kwargs = predictor.sam_predictor.predict.call_args.kwargs
        coords = kwargs["point_coords"]
        assert coords.shape == (2, 2)
        np.testing.assert_array_equal(coords[0], [10, 20])
        np.testing.assert_array_equal(coords[1], [30, 40])

    def test_no_points_box_not_none_in_call(self, predictor):
        predictor.sam_predictor.predict.return_value = _fake_predict(50, 50)
        box = [5, 5, 45, 45]
        predictor.predict(pos_points=[], neg_points=[], box=box)
        kwargs = predictor.sam_predictor.predict.call_args.kwargs
        np.testing.assert_array_equal(kwargs["box"], box)

    def test_multimask_output_true(self, predictor):
        predictor.sam_predictor.predict.return_value = _fake_predict(50, 50)
        predictor.predict(pos_points=[(25, 25)], neg_points=[], box=None)
        kwargs = predictor.sam_predictor.predict.call_args.kwargs
        assert kwargs["multimask_output"] is True


def _ann(area: int, h: int, w: int) -> dict:
    seg = np.zeros((h, w), dtype=bool)
    seg.reshape(-1)[:area] = True
    return {"segmentation": seg, "area": int(seg.sum())}


class TestPredictAll:
    def _patch_amg(self, predictor, anns):
        predictor._mask_gen.generate = MagicMock(return_value=anns)

    def test_returns_uint8_binary(self, predictor):
        h, w = 100, 100
        self._patch_amg(predictor, [_ann(2000, h, w)])
        out = predictor.predict_all(np.zeros((h, w, 3), dtype=np.uint8))
        assert len(out) == 1
        assert out[0].dtype == np.uint8
        assert set(np.unique(out[0])).issubset({0, 255})

    def test_filters_too_small(self, predictor):
        h, w = 100, 100
        # min_area_frac=0.001 → отсеиваем всё мельче 10 px.
        self._patch_amg(predictor, [_ann(5, h, w), _ann(2000, h, w)])
        assert len(predictor.predict_all(np.zeros((h, w, 3), dtype=np.uint8))) == 1

    def test_filters_too_big(self, predictor):
        h, w = 100, 100
        # max_area_frac=0.7 → отсеиваем всё крупнее 7000 px.
        self._patch_amg(predictor, [_ann(2000, h, w), _ann(8000, h, w)])
        assert len(predictor.predict_all(np.zeros((h, w, 3), dtype=np.uint8))) == 1

    def test_empty_annotations(self, predictor):
        h, w = 100, 100
        self._patch_amg(predictor, [])
        assert predictor.predict_all(np.zeros((h, w, 3), dtype=np.uint8)) == []

    def test_custom_thresholds(self, predictor):
        h, w = 100, 100
        self._patch_amg(predictor, [_ann(5, h, w)])
        out = predictor.predict_all(
            np.zeros((h, w, 3), dtype=np.uint8),
            min_area_frac=0.0001, max_area_frac=0.9,
        )
        assert len(out) == 1
