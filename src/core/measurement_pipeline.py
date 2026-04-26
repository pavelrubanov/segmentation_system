"""Универсальный конвейер измерений и экспорта.

Любой режим работы (готовые кропы с диска, кропы из автосегментации) сводится
к одному циклу: для каждого исходного файла источник выдаёт 0..N кропов;
каждый кроп измеряется, его визуализация сохраняется, строка попадает в экспорт.

Источник --- это просто `Callable[[str], Iterable[CropItem]]`. Конкретные
реализации в `core/sources.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from .export import save_csv, save_xlsx
from .file_naming import vis_filename
from .leaf_measure import (
    LeafMetrics,
    build_fractions,
    measure_leaf,
    save_measurement_visualization,
)


@dataclass
class CropItem:
    """Один кроп для обработки. `crop` --- RGB H×W×3 uint8.
    `name` --- базовое имя без расширения (попадёт в колонку `image`)."""
    name: str
    crop: np.ndarray


CropSource = Callable[[str], Iterable[CropItem]]


@dataclass
class PipelineSettings:
    """Базовые настройки обработки, общие для всех источников."""
    files: list[str]
    out_dir: Path
    n_sections: int
    unit_name: str
    px_per_unit: Optional[float]   # None --- без пересчёта, считаем в пикселях
    export_format: str             # "csv" или "xlsx"

    @property
    def unit(self) -> str:
        return self.unit_name if self.px_per_unit else "px"

    @property
    def scale(self) -> float:
        return 1.0 / self.px_per_unit if self.px_per_unit else 1.0


@dataclass
class MeasurementResult:
    name: str
    metrics: LeafMetrics
    vis_path: Path


@dataclass
class PipelineResult:
    rows: list[dict] = field(default_factory=list)
    thumb_paths: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def n_ok(self) -> int:
        return len(self.rows)


def measure_one(
    item: CropItem,
    fractions: Sequence[float],
    unit: str,
    scale: float,
    vis_dir: Path,
) -> MeasurementResult:
    """Измерить один кроп и сохранить визуализацию рядом."""
    metrics = measure_leaf(item.crop, fractions=fractions, unit=unit, scale=scale)
    vis_path = vis_dir / vis_filename(item.name)
    save_measurement_visualization(item.crop, metrics, str(vis_path))
    return MeasurementResult(name=item.name, metrics=metrics, vis_path=vis_path)


def metrics_to_row(
    res: MeasurementResult,
    fractions: Sequence[float],
    unit: str,
    with_crop_col: bool,
) -> dict:
    """Собрать строку для экспорта. with_crop_col добавляет колонку 'crop'
    с путём к vis-картинке (для встраивания в XLSX)."""
    row: dict = {}
    if with_crop_col:
        row["crop"] = str(res.vis_path)
    row["image"] = res.name
    row[f"length_{unit}"] = f"{res.metrics.length:.2f}"
    row[f"width_{unit}"] = f"{res.metrics.width:.2f}"
    for f in fractions:
        sec = res.metrics.widths.get(f)
        row[f"width_{f:.3f}_{unit}"] = f"{sec.width:.2f}" if sec else "0.00"
    return row


def write_export(
    rows: list[dict],
    out_path: Path,
    export_format: str,
    thumb_paths: list[str] | None = None,
) -> None:
    """Сохранить rows в CSV или XLSX."""
    if not rows:
        return
    if export_format == "xlsx":
        save_xlsx(rows, out_path, thumb_paths=thumb_paths)
    else:
        save_csv(rows, out_path)


def run_pipeline(
    files: Sequence[str],
    source: CropSource,
    settings: PipelineSettings,
    on_progress: Callable[[int, str], bool] | None = None,
) -> PipelineResult:
    """Прогнать пакет файлов через source. Для каждого выданного CropItem
    измерить, сохранить vis, добавить строку. В конце записать экспорт.

    on_progress(i, name) → False прерывает цикл (для отмены пользователем).
    Ошибка обработки одного файла попадает в `result.errors`, цикл продолжается.
    Ошибка экспорта тоже попадает в `result.errors` --- результаты измерений
    к этому моменту уже на диске (vis-картинки), теряется только сводный файл.
    """
    fractions = build_fractions(settings.n_sections)
    unit, scale = settings.unit, settings.scale
    use_xlsx = settings.export_format == "xlsx"

    vis_dir = settings.out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    result = PipelineResult()
    for i, fpath in enumerate(files):
        name = Path(fpath).name
        if on_progress is not None and not on_progress(i, name):
            break
        try:
            for item in source(fpath):
                res = measure_one(item, fractions, unit, scale, vis_dir)
                result.rows.append(metrics_to_row(
                    res, fractions, unit, with_crop_col=use_xlsx))
                result.thumb_paths.append(str(res.vis_path))
        except Exception as exc:
            result.errors.append(f"{name}: {exc}")

    if result.rows:
        out_name = "measurements.xlsx" if use_xlsx else "measurements.csv"
        try:
            write_export(
                result.rows, settings.out_dir / out_name, settings.export_format,
                thumb_paths=result.thumb_paths if use_xlsx else None,
            )
        except Exception as exc:
            result.errors.append(f"Ошибка экспорта: {exc}")
    return result
