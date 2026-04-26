"""Классификатор листа для приложения (inference API).

Использование:
    from classifier import LeafClassifier
    clf = LeafClassifier("models/model_branch.pkl")
    label, conf = clf.classify(crop)   # crop --- уже препроцессированный RGB bbox
    # label ∈ {"good_leaf", "bad_leaf", "non_leaf"}

На вход принимается bbox-кроп из `core.leaf_pipeline.transform_leaf()`
(апекс вверх, чёрный фон, переменный размер). Внутри: mobilenet.encode,
XGBoost predict_proba, argmax по вероятностям.
"""
from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

import numpy as np

from . import mobilenet

_MODEL_RE = re.compile(r"^model_(?P<type>.+)\.pkl$")

# Кэш классификаторов по абсолютному пути к pkl: загрузка XGBoost тяжёлая,
# а в одной сессии один и тот же weights_path запрашивается многократно
# (повторные запуски авто-обработки, кнопка авто-сегментации в окне).
_classifier_cache: dict[str, "LeafClassifier"] = {}


def get_classifier(weights_path: str | Path) -> "LeafClassifier":
    """Вернуть классификатор по пути к pkl. Загружается один раз за сессию."""
    key = str(Path(weights_path).resolve())
    cached = _classifier_cache.get(key)
    if cached is None:
        cached = LeafClassifier(key)
        _classifier_cache[key] = cached
    return cached


def find_available_leaf_types(weights_dir: str | Path) -> list[tuple[str, str]]:
    """Найти все файлы вида `model_<type>.pkl` в папке. Возвращает [(type, path), ...]."""
    d = Path(weights_dir)
    if not d.is_dir():
        return []
    return [
        (m.group("type"), str(d / name))
        for name in sorted(os.listdir(d))
        if (m := _MODEL_RE.match(name))
    ]


class LeafClassifier:
    """Загружается один раз. Классифицирует препроцессированные кропы."""

    def __init__(self, weights_path: str | Path) -> None:
        with open(weights_path, "rb") as f:
            blob = pickle.load(f)
        self.model = blob["model"]
        self.classes: list[str] = list(blob["classes"])
        self.leaf_type: str = blob.get("leaf_type", "?")

    def classify(self, crop: np.ndarray) -> tuple[str, float]:
        """Классифицировать один кроп. Возвращает (class_name, confidence)."""
        return self.classify_batch([crop])[0]

    def classify_batch(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Классифицировать батч кропов. Возвращает [(class_name, confidence), ...]."""
        if not crops:
            return []
        feats = mobilenet.encode(crops)
        probs = self.model.predict_proba(feats)
        idxs = probs.argmax(axis=1)
        return [(self.classes[i], float(probs[b, i])) for b, i in enumerate(idxs)]
