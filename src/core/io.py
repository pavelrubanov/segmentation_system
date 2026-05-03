"""Мелкие хелперы, общие между UI и пакетной обработкой."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def read_rgb(path: str | Path) -> np.ndarray:
    """Прочитать изображение и вернуть RGB numpy-массив (uint8 H×W×3)."""
    return np.array(Image.open(path).convert("RGB"))


def make_run_dir(out_dir: Path, tag: str) -> Path:
    """Создать `out_dir/{tag}_{YYYY-MM-DD_HH-MM-SS}/` и вернуть путь.

    Метка времени гарантирует уникальность между запусками; тег режима
    (`segment`, `measure`, `auto`) даёт группировку по типу при сортировке.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run = out_dir / f"{tag}_{timestamp}"
    run.mkdir(parents=True, exist_ok=True)
    return run


def pluralize_ru(n: int, forms: Sequence[str]) -> str:
    """Выбрать русскую форму существительного по числу: (1, 2, 5).

    Например: pluralize_ru(3, ("маска", "маски", "масок")) → "маски".
    """
    one, few, many = forms
    mod10, mod100 = n % 10, n % 100
    if mod10 == 1 and mod100 != 11:
        return one
    if 2 <= mod10 <= 4 and not (12 <= mod100 <= 14):
        return few
    return many


def format_count(n: int, forms: Sequence[str]) -> str:
    """`pluralize_ru` + само число впереди ("3 маски")."""
    return f"{n} {pluralize_ru(n, forms)}"
