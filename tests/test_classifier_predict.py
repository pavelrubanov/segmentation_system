"""Тесты classifier/predict.py: загрузка pkl, кэш, классификация (с моками)."""
import pickle
from unittest.mock import patch

import numpy as np
import pytest

from classifier.predict import (
    LeafClassifier,
    _classifier_cache,
    find_available_leaf_types,
    get_classifier,
)


class _DummyXgb:
    """Мини-замена XGBoost для тестов: pickle-able, predict_proba возвращает
    заранее заданные вероятности по каждой строке батча."""

    def __init__(self, probs: list[list[float]]) -> None:
        self._probs = np.asarray(probs, dtype=float)

    def predict_proba(self, X) -> np.ndarray:
        n = len(X)
        if self._probs.ndim == 1:
            return np.tile(self._probs, (n, 1))
        # Если задан per-sample матрица --- берём первые n строк
        return self._probs[:n]


def _save_pkl(path, model, classes, leaf_type="branch"):
    with open(path, "wb") as f:
        pickle.dump({"model": model, "classes": list(classes), "leaf_type": leaf_type}, f)


@pytest.fixture(autouse=True)
def clear_cache():
    _classifier_cache.clear()
    yield
    _classifier_cache.clear()


# ── find_available_leaf_types ────────────────────────────────────────────────


class TestFindAvailableLeafTypes:
    def test_finds_models_alphabetical(self, tmp_path):
        for name in ("model_z.pkl", "model_a.pkl", "model_m.pkl"):
            (tmp_path / name).touch()
        result = find_available_leaf_types(tmp_path)
        assert [t for t, _ in result] == ["a", "m", "z"]

    def test_ignores_non_model_files(self, tmp_path):
        (tmp_path / "model_branch.pkl").touch()
        (tmp_path / "garbage.pkl").touch()
        (tmp_path / "model.pkl").touch()
        (tmp_path / "not_a_model.txt").touch()
        result = find_available_leaf_types(tmp_path)
        assert [t for t, _ in result] == ["branch"]

    def test_paths_are_strings(self, tmp_path):
        (tmp_path / "model_x.pkl").touch()
        result = find_available_leaf_types(tmp_path)
        assert isinstance(result[0][1], str)

    def test_empty_dir(self, tmp_path):
        assert find_available_leaf_types(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path):
        assert find_available_leaf_types(tmp_path / "nope") == []


# ── LeafClassifier ───────────────────────────────────────────────────────────


class TestLeafClassifier:
    def test_loads_classes_and_leaf_type(self, tmp_path):
        path = tmp_path / "model_x.pkl"
        _save_pkl(path, _DummyXgb([0.5, 0.5]), ["good_leaf", "bad_leaf"], "branch")
        clf = LeafClassifier(path)
        assert clf.classes == ["good_leaf", "bad_leaf"]
        assert clf.leaf_type == "branch"

    def test_classify_batch_empty(self, tmp_path):
        path = tmp_path / "model_x.pkl"
        _save_pkl(path, _DummyXgb([0.5, 0.5]), ["a", "b"])
        assert LeafClassifier(path).classify_batch([]) == []

    @patch("classifier.predict.mobilenet.encode")
    def test_classify_batch_returns_argmax_per_row(self, mock_encode, tmp_path):
        path = tmp_path / "model_x.pkl"
        # Каждая строка --- разные вероятности
        _save_pkl(
            path,
            _DummyXgb([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]]),
            ["good_leaf", "bad_leaf", "non_leaf"],
        )
        mock_encode.return_value = np.zeros((2, 576))
        crops = [np.zeros((10, 10, 3), dtype=np.uint8)] * 2

        result = LeafClassifier(path).classify_batch(crops)
        assert result == [("bad_leaf", pytest.approx(0.7)), ("good_leaf", pytest.approx(0.6))]

    @patch("classifier.predict.mobilenet.encode")
    def test_classify_single_uses_batch(self, mock_encode, tmp_path):
        path = tmp_path / "model_x.pkl"
        _save_pkl(path, _DummyXgb([0.9, 0.05, 0.05]), ["good_leaf", "bad_leaf", "non_leaf"])
        mock_encode.return_value = np.zeros((1, 576))

        label, conf = LeafClassifier(path).classify(np.zeros((10, 10, 3), dtype=np.uint8))
        assert label == "good_leaf"
        assert conf == pytest.approx(0.9)


# ── get_classifier (кэш по пути) ─────────────────────────────────────────────


class TestGetClassifier:
    def test_caches_by_path(self, tmp_path):
        path = tmp_path / "model_x.pkl"
        _save_pkl(path, _DummyXgb([1.0]), ["x"])
        assert get_classifier(path) is get_classifier(path)

    def test_different_paths_different_instances(self, tmp_path):
        p1 = tmp_path / "model_a.pkl"; _save_pkl(p1, _DummyXgb([1.0]), ["x"])
        p2 = tmp_path / "model_b.pkl"; _save_pkl(p2, _DummyXgb([1.0]), ["x"])
        assert get_classifier(p1) is not get_classifier(p2)

    def test_relative_and_absolute_path_same_instance(self, tmp_path, monkeypatch):
        path = tmp_path / "model_x.pkl"
        _save_pkl(path, _DummyXgb([1.0]), ["x"])
        monkeypatch.chdir(tmp_path)
        assert get_classifier(path.resolve()) is get_classifier("model_x.pkl")
