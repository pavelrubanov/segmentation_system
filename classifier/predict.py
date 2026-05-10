"""Классификатор листа для приложения (inference API).

Использование:
    from classifier import LeafClassifier
    clf = LeafClassifier("models/model_branch.pkl")
    label, conf = clf.classify(crop)   # crop --- уже препроцессированный RGB bbox
    # label ∈ {"good_leaf", "bad_leaf", "non_leaf"}

На вход принимается bbox-кроп из `core.leaf_pipeline.transform_leaf()`
(апекс вверх, чёрный фон, переменный размер). Внутри: mobilenet.encode,
XGBoost predict_proba, порог p_good по типу листа
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from . import mobilenet

GOOD_THRESHOLDS: dict[str, float] = {
    "branch": 0.22,
    "capillifolium": 0.24,
    "girgensohnii": 0.12,
}

# Кэш классификаторов по абсолютному пути к pkl: загрузка XGBoost тяжёлая,
# а в одной сессии один и тот же weights_path запрашивается многократно
# (повторные запуски авто-обработки, кнопка авто-сегментации в окне).
_classifier_cache: dict[str, "LeafClassifier"] = {}


def get_classifier(weights_path: str | Path) -> "LeafClassifier":
    """Вернуть классификатор по пути к pkl. Загружается один раз за сессию."""
    key = str(Path(weights_path).resolve())
    if (cached := _classifier_cache.get(key)) is None:
        cached = LeafClassifier(key)
        _classifier_cache[key] = cached
    return cached


def find_available_leaf_types(weights_dir: str | Path) -> list[tuple[str, str]]:
    """Все файлы вида `model_<type>.pkl` в папке. Возвращает [(type, path), ...]."""
    d = Path(weights_dir)
    if not d.is_dir():
        return []
    return sorted(
        (p.stem.removeprefix("model_"), str(p))
        for p in d.glob("model_*.pkl")
    )


class LeafClassifier:
    """Загружается один раз. Классифицирует препроцессированные кропы."""

    def __init__(self, weights_path: str | Path) -> None:
        with open(weights_path, "rb") as f:
            blob = pickle.load(f)
        self.model = blob["model"]
        self.classes: list[str] = list(blob["classes"])
        self.leaf_type: str = blob["leaf_type"]
        self.good_idx = self.classes.index("good_leaf")
        self.p_good = GOOD_THRESHOLDS[self.leaf_type]

    def classify(self, crop: np.ndarray) -> tuple[str, float]:
        """Классифицировать один кроп. Возвращает (class_name, confidence)."""
        return self.classify_batch([crop])[0]

    def classify_batch(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Возвращает [(class_name, confidence), ...].

        good_leaf присваивается, если p(good_leaf) >= p_good (рабочая точка
        recall >= 0.95). Иначе --- argmax по остальным классам.
        """
        if not crops:
            return []
        probs = self.model.predict_proba(mobilenet.encode(crops))
        good_probs = probs[:, self.good_idx]
        rest = np.delete(probs, self.good_idx, axis=1)
        rest_classes = [c for i, c in enumerate(self.classes) if i != self.good_idx]
        rest_idxs = rest.argmax(axis=1)
        out: list[tuple[str, float]] = []
        for b in range(len(crops)):
            if good_probs[b] >= self.p_good:
                out.append(("good_leaf", float(good_probs[b])))
            else:
                j = int(rest_idxs[b])
                out.append((rest_classes[j], float(rest[b, j])))
        return out
