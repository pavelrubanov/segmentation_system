"""Тесты модуля core/predictor.py (с заглушками для MobileSAM и torch)."""
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

# ── Мок mobile_sam (до импорта predictor) ────────────────────────────────────
# Создаём фиктивный модуль, чтобы не требовалась установка MobileSAM.

_mock_sam_instance = MagicMock()
_mock_predictor_cls = MagicMock()
_mock_registry = {"vit_t": MagicMock(return_value=_mock_sam_instance)}

_mock_mobile_sam = types.ModuleType("mobile_sam")
_mock_mobile_sam.sam_model_registry = _mock_registry
_mock_mobile_sam.SamPredictor = _mock_predictor_cls
sys.modules.setdefault("mobile_sam", _mock_mobile_sam)

from core.predictor import Predictor  # noqa: E402  (импорт после патча)


# ── Фикстура ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def predictor():
    """Создаёт экземпляр Predictor с моком sam, сбрасывает счётчики между тестами."""
    _mock_registry["vit_t"].reset_mock()
    _mock_predictor_cls.reset_mock()
    return Predictor(checkpoint_path="fake_weights.pt")


def _fake_predict(h: int, w: int, best_idx: int = 1):
    """Возвращает (masks, scores, _) с нужным «лучшим» индексом."""
    masks = np.zeros((3, h, w), dtype=bool)
    masks[best_idx] = True
    scores = np.array([0.3, 0.95, 0.6])
    scores[best_idx] = max(scores) + 0.1
    return masks, scores, None


# ── Инициализация ─────────────────────────────────────────────────────────────


class TestPredictorInit:

    def test_registry_called_with_checkpoint(self, predictor):
        """При инициализации вызывается sam_model_registry["vit_t"] с нужным путём."""
        _mock_registry["vit_t"].assert_called_once_with(checkpoint="fake_weights.pt")

    def test_sam_predictor_instantiated(self, predictor):
        """SamPredictor создаётся ровно один раз."""
        _mock_predictor_cls.assert_called_once()

    def test_sam_predictor_attribute_set(self, predictor):
        """Атрибут sam_predictor установлен."""
        assert predictor.sam_predictor is not None


# ── predict: граничные случаи ─────────────────────────────────────────────────


class TestPredictNoPrompts:

    def test_no_points_no_box_returns_none(self, predictor):
        """Нет ни точек, ни bounding box → None."""
        assert predictor.predict([], [], None) is None

    def test_sam_predict_not_called_when_no_prompts(self, predictor):
        """SAM-предиктор не вызывается, если нет промптов."""
        predictor.predict([], [], None)
        predictor.sam_predictor.predict.assert_not_called()


# ── predict: промпты ──────────────────────────────────────────────────────────


class TestPredictWithPrompts:

    def test_positive_point_returns_mask(self, predictor):
        """Одна позитивная точка → результат не None."""
        predictor.sam_predictor.predict.return_value = _fake_predict(50, 60)
        result = predictor.predict(pos_points=[(25, 25)], neg_points=[], box=None)
        assert result is not None

    def test_output_dtype_uint8(self, predictor):
        """Выходная маска имеет тип uint8."""
        predictor.sam_predictor.predict.return_value = _fake_predict(50, 60)
        result = predictor.predict(pos_points=[(25, 25)], neg_points=[], box=None)
        assert result.dtype == np.uint8

    def test_output_values_0_or_255(self, predictor):
        """Пиксели маски принимают только значения 0 или 255."""
        predictor.sam_predictor.predict.return_value = _fake_predict(50, 60)
        result = predictor.predict(pos_points=[(25, 25)], neg_points=[], box=None)
        assert set(np.unique(result)).issubset({0, 255})

    def test_output_shape_matches_input(self, predictor):
        """Форма выходной маски совпадает с формой входных масок."""
        h, w = 64, 96
        predictor.sam_predictor.predict.return_value = _fake_predict(h, w)
        result = predictor.predict(pos_points=[(30, 30)], neg_points=[], box=None)
        assert result.shape == (h, w)

    def test_best_mask_selected_by_score(self, predictor):
        """Выбирается маска с наибольшим score."""
        h, w = 40, 40
        masks = np.zeros((3, h, w), dtype=bool)
        masks[0, :, :] = False   # лучший score → индекс 2
        masks[1, :10, :10] = True
        masks[2, :, :] = True    # этот должен быть выбран
        scores = np.array([0.4, 0.6, 0.95])
        predictor.sam_predictor.predict.return_value = (masks, scores, None)

        result = predictor.predict(pos_points=[(20, 20)], neg_points=[], box=None)
        assert np.all(result == 255)

    def test_box_only_prompt(self, predictor):
        """Только bounding box — predict работает без точек."""
        predictor.sam_predictor.predict.return_value = _fake_predict(80, 100)
        result = predictor.predict(pos_points=[], neg_points=[], box=[10, 10, 90, 70])
        assert result is not None

    def test_negative_points_label_zero(self, predictor):
        """Негативные точки передаются с label=0, позитивные — с label=1."""
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
        """Координаты точек объединяются: сначала pos, потом neg."""
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
        """При наличии bounding box параметр box передаётся в SAM."""
        predictor.sam_predictor.predict.return_value = _fake_predict(50, 50)
        box = [5, 5, 45, 45]
        predictor.predict(pos_points=[], neg_points=[], box=box)
        kwargs = predictor.sam_predictor.predict.call_args.kwargs
        np.testing.assert_array_equal(kwargs["box"], box)

    def test_multimask_output_true(self, predictor):
        """SAM вызывается с multimask_output=True."""
        predictor.sam_predictor.predict.return_value = _fake_predict(50, 50)
        predictor.predict(pos_points=[(25, 25)], neg_points=[], box=None)
        kwargs = predictor.sam_predictor.predict.call_args.kwargs
        assert kwargs["multimask_output"] is True
