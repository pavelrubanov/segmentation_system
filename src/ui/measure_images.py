"""Авто-сегментация и замеры: на вход исходные фото, на выход --- метрики
по найденным хорошим листьям + сохранённые crops/masks/vis + CSV/XLSX."""
from dataclasses import asdict
from pathlib import Path

from PyQt6 import QtWidgets

from classifier import find_available_leaf_types, get_classifier
from core.file_naming import image_file_filter
from core.paths import resource_path
from core.predictor import Predictor
from core.sources import AutoSegmentSource

from .runner import (
    FilesPicker,
    ImagesPipelineSettings,
    PipelineWorker,
    SettingsDialog,
    format_done_msg,
    run_with_progress,
)

_CLASSIFIER_WEIGHTS_DIR = resource_path("models")


class ImagesSettingsDialog(SettingsDialog):
    """Диалог настроек для авто-режима = база + выбор модели классификатора."""

    def __init__(self, leaf_types: list[tuple[str, str]], parent=None):
        super().__init__(
            title="Авто-сегментация и замеры",
            files_picker=FilesPicker(
                "Изображения",
                "Выберите изображения",
                image_file_filter(),
                ("изображение", "изображения", "изображений"),
            ),
            default_format="xlsx",
            parent=parent,
        )
        self._leaf_types = leaf_types

        grp = QtWidgets.QGroupBox("Тип листа")
        h = QtWidgets.QHBoxLayout(grp)
        self._combo = QtWidgets.QComboBox()
        for t, _ in leaf_types:
            self._combo.addItem(t.replace("_", " ").title(), t)
        if not leaf_types:
            self._combo.addItem("нет моделей", None)
            self._combo.setEnabled(False)
        h.addWidget(self._combo, 1)
        self._insert_extra(grp)
        self._add_validator(lambda: self._combo.currentData() is not None)

    def settings(self) -> ImagesPipelineSettings:
        leaf_type = self._combo.currentData()
        weights = next(p for t, p in self._leaf_types if t == leaf_type)
        return ImagesPipelineSettings(
            **asdict(self.pipeline_settings()),
            leaf_type=leaf_type,
            classifier_weights=weights,
        )


def measure_images(parent: QtWidgets.QWidget, predictor: Predictor) -> None:
    """Открыть диалог и запустить авто-сегментацию + замеры по выбранным фото."""
    leaf_types = find_available_leaf_types(str(_CLASSIFIER_WEIGHTS_DIR))
    if not leaf_types:
        QtWidgets.QMessageBox.warning(
            parent, "Недоступно",
            "Классификатор листьев не найден. Проверь модели в models/.")
        return

    dlg = ImagesSettingsDialog(leaf_types, parent=parent)
    if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return

    s = dlg.settings()
    try:
        classifier = get_classifier(s.classifier_weights)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(
            parent, "Ошибка", f"Не удалось загрузить классификатор: {exc}")
        return

    source = AutoSegmentSource(
        predictor=predictor,
        classifier=classifier,
        crops_dir=s.out_dir / "crops",
        masks_dir=s.out_dir / "masks",
    )
    worker = PipelineWorker(files=s.files, source=source, settings=s)

    def msg(result):
        return format_done_msg(
            result, total_files=len(s.files),
            files_label="Обработано изображений",
            rows_label="Найдено хороших листьев",
            out_dir=s.out_dir,
        )

    run_with_progress(
        parent, worker,
        title="Авто-сегментация и замеры",
        label="Обработка изображений…",
        finished_msg=msg,
    )
