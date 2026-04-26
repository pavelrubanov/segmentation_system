"""Общие виджеты, диалог настроек, фоновый воркер и runner для batch-команд.

Архитектура: оба batch-режима (готовые кропы и автосегментация исходных
изображений) делят один диалог настроек, один фоновый воркер и один runner
с прогресс-диалогом. Различие только в источнике кропов (см. core/sources.py)
и в том, добавляет ли диалог дополнительные поля (наследник SettingsDialog).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from PyQt6 import QtCore, QtWidgets

from core.io import format_count
from core.measurement_pipeline import (
    CropSource,
    PipelineResult,
    PipelineSettings,
    run_pipeline,
)


@dataclass
class ImagesPipelineSettings(PipelineSettings):
    """Настройки авто-режима = базовые + выбор модели классификатора."""
    leaf_type: str = ""
    classifier_weights: str = ""


# --- блоки виджетов для диалога ---


class SectionsBlock(QtWidgets.QGroupBox):
    """GroupBox со SpinBox для числа поперечных сечений."""

    def __init__(self, default: int = 7, parent=None):
        super().__init__("Поперечные сечения", parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Количество линий:"))
        self.spin = QtWidgets.QSpinBox()
        self.spin.setRange(1, 99)
        self.spin.setValue(default)
        layout.addWidget(self.spin)
        layout.addStretch()

    def value(self) -> int:
        return self.spin.value()


class ScaleBlock(QtWidgets.QGroupBox):
    """GroupBox с чекбоксом «Пересчитывать в единицы измерения» и полями."""

    def __init__(self, parent=None):
        super().__init__("Масштаб", parent)
        layout = QtWidgets.QGridLayout(self)
        self.chk = QtWidgets.QCheckBox("Пересчитывать в единицы измерения")
        self.chk.toggled.connect(self._on_toggled)
        layout.addWidget(self.chk, 0, 0, 1, 3)

        self.lbl_px = QtWidgets.QLabel("Пикселей на единицу:")
        self.spin_px = QtWidgets.QDoubleSpinBox()
        self.spin_px.setRange(0.01, 100_000.0)
        self.spin_px.setDecimals(2)
        self.spin_px.setValue(100.0)
        layout.addWidget(self.lbl_px, 1, 0)
        layout.addWidget(self.spin_px, 1, 1)

        self.lbl_unit = QtWidgets.QLabel("Единица:")
        self.edit_unit = QtWidgets.QLineEdit("мм")
        self.edit_unit.setMaximumWidth(80)
        layout.addWidget(self.lbl_unit, 2, 0)
        layout.addWidget(self.edit_unit, 2, 1)

        self._on_toggled(False)

    def _on_toggled(self, on: bool) -> None:
        for w in (self.lbl_px, self.spin_px, self.lbl_unit, self.edit_unit):
            w.setEnabled(on)

    def px_per_unit(self) -> Optional[float]:
        return self.spin_px.value() if self.chk.isChecked() else None

    def unit_name(self) -> str:
        return self.edit_unit.text().strip() or "мм"


class FormatBlock(QtWidgets.QGroupBox):
    """GroupBox с выбором CSV/XLSX."""

    def __init__(self, default: str = "csv", parent=None):
        super().__init__("Формат экспорта", parent)
        layout = QtWidgets.QHBoxLayout(self)
        self.rb_csv = QtWidgets.QRadioButton("CSV")
        self.rb_xlsx = QtWidgets.QRadioButton("Excel (.xlsx)")
        (self.rb_xlsx if default == "xlsx" else self.rb_csv).setChecked(True)
        layout.addWidget(self.rb_csv)
        layout.addWidget(self.rb_xlsx)
        layout.addStretch()

    def value(self) -> str:
        return "xlsx" if self.rb_xlsx.isChecked() else "csv"


class FilesPicker(QtWidgets.QGroupBox):
    """GroupBox с кнопкой «Выбрать…» и подписью с количеством файлов."""

    def __init__(self, title: str, dialog_caption: str, file_filter: str,
                 forms: tuple[str, str, str], parent=None):
        super().__init__(title, parent)
        self._caption = dialog_caption
        self._filter = file_filter
        self._forms = forms
        self._files: list[str] = []
        self._on_change: Callable[[], None] = lambda: None

        layout = QtWidgets.QHBoxLayout(self)
        self.lbl = QtWidgets.QLabel("Файлы не выбраны")
        self.lbl.setObjectName("secondary")
        btn = QtWidgets.QPushButton("Выбрать…")
        btn.clicked.connect(self._pick)
        layout.addWidget(self.lbl, 1)
        layout.addWidget(btn)

    def files(self) -> list[str]:
        return list(self._files)

    def on_change(self, cb: Callable[[], None]) -> None:
        self._on_change = cb

    def _pick(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, self._caption, "", self._filter)
        if files:
            self._files = files
            self.lbl.setText(f"Выбрано: {format_count(len(files), self._forms)}")
            self._on_change()


class OutDirPicker(QtWidgets.QGroupBox):
    """GroupBox с кнопкой «Выбрать…» для выходной папки."""

    def __init__(self, parent=None):
        super().__init__("Выходная директория", parent)
        self._on_change: Callable[[], None] = lambda: None
        layout = QtWidgets.QHBoxLayout(self)
        self.lbl = QtWidgets.QLabel("Не выбрана")
        self.lbl.setObjectName("secondary")
        btn = QtWidgets.QPushButton("Выбрать…")
        btn.clicked.connect(self._pick)
        layout.addWidget(self.lbl, 1)
        layout.addWidget(btn)

    def path(self) -> Optional[Path]:
        if self.lbl.text() == "Не выбрана":
            return None
        return Path(self.lbl.text())

    def on_change(self, cb: Callable[[], None]) -> None:
        self._on_change = cb

    def _pick(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите выходную директорию")
        if d:
            self.lbl.setText(d)
            self._on_change()


# --- диалог настроек ---


class SettingsDialog(QtWidgets.QDialog):
    """Каркас диалога: файлы + out_dir + (вставка наследника) + sections + scale + format."""

    def __init__(
        self,
        *,
        title: str,
        files_picker: FilesPicker,
        default_format: str = "csv",
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(520)

        self._files = files_picker
        self._out = OutDirPicker()
        self._sections = SectionsBlock()
        self._scale = ScaleBlock()
        self._format = FormatBlock(default=default_format)
        self._extra_validators: list[Callable[[], bool]] = []

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setSpacing(12)
        self._layout.addWidget(self._files)
        self._layout.addWidget(self._out)
        # сюда `_insert_extra` будет вставлять блоки наследника
        self._extras_anchor = self._layout.count()
        self._layout.addWidget(self._sections)
        self._layout.addWidget(self._scale)
        self._layout.addWidget(self._format)

        # OK / Отмена
        btns = QtWidgets.QHBoxLayout()
        btns.addStretch()
        btn_cancel = QtWidgets.QPushButton("Отмена")
        btn_cancel.clicked.connect(self.reject)
        self._btn_ok = QtWidgets.QPushButton("Обработать")
        self._btn_ok.setObjectName("primary")
        self._btn_ok.setEnabled(False)
        self._btn_ok.clicked.connect(self.accept)
        btns.addWidget(btn_cancel)
        btns.addWidget(self._btn_ok)
        self._layout.addLayout(btns)

        self._files.on_change(self._update_ok)
        self._out.on_change(self._update_ok)

    def _insert_extra(self, widget: QtWidgets.QWidget) -> None:
        """Наследник вставляет свои поля между out_dir и sections."""
        self._layout.insertWidget(self._extras_anchor, widget)
        self._extras_anchor += 1

    def _add_validator(self, fn: Callable[[], bool]) -> None:
        """Доп. условие, при котором кнопка OK становится активной."""
        self._extra_validators.append(fn)
        self._update_ok()

    def _update_ok(self) -> None:
        self._btn_ok.setEnabled(
            bool(self._files.files())
            and self._out.path() is not None
            and all(v() for v in self._extra_validators)
        )

    def pipeline_settings(self) -> PipelineSettings:
        return PipelineSettings(
            files=self._files.files(),
            out_dir=self._out.path(),
            n_sections=self._sections.value(),
            unit_name=self._scale.unit_name(),
            px_per_unit=self._scale.px_per_unit(),
            export_format=self._format.value(),
        )


# --- фоновый воркер и runner с прогресс-диалогом ---


class PipelineWorker(QtCore.QThread):
    """Универсальный воркер: гоняет run_pipeline в фоне и шлёт сигналы."""
    progress = QtCore.pyqtSignal(int, str)
    finished_ok = QtCore.pyqtSignal(object)   # PipelineResult
    failed = QtCore.pyqtSignal(str)

    def __init__(self, files: Sequence[str], source: CropSource,
                 settings: PipelineSettings):
        super().__init__()
        self._files = list(files)
        self._source = source
        self._settings = settings
        self._cancel = False

    @property
    def file_count(self) -> int:
        return len(self._files)

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            def cb(i: int, name: str) -> bool:
                self.progress.emit(i, name)
                return not self._cancel
            result = run_pipeline(
                files=self._files,
                source=self._source,
                settings=self._settings,
                on_progress=cb,
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


# Worker'ы держим тут, чтобы они не уехали в GC до завершения --- runner сам
# их забывает по сигналу finished. Нет привязки к UI-объектам => нет
# monkey-patch'ей на NavigationWindow.
_active_workers: set[PipelineWorker] = set()


def run_with_progress(
    parent: QtWidgets.QWidget,
    worker: PipelineWorker,
    *,
    title: str,
    label: str,
    finished_msg: Callable[[PipelineResult], str],
) -> None:
    """Создать ProgressDialog, подключить сигналы worker'а, удерживать ссылку
    на воркер до завершения. По окончании --- MessageBox с finished_msg(result).
    """
    total = worker.file_count
    progress = QtWidgets.QProgressDialog(label, "Отмена", 0, total, parent)
    progress.setWindowTitle(title)
    progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(0)
    progress.setAutoClose(False)
    progress.setAutoReset(False)

    _active_workers.add(worker)

    def on_progress(i: int, name: str) -> None:
        progress.setValue(i)
        progress.setLabelText(f"[{i + 1}/{total}] {name}")

    def on_finished(result: PipelineResult) -> None:
        progress.setValue(total)
        progress.close()
        _active_workers.discard(worker)
        QtWidgets.QMessageBox.information(parent, title, finished_msg(result))

    def on_failed(err: str) -> None:
        progress.close()
        _active_workers.discard(worker)
        QtWidgets.QMessageBox.critical(parent, "Ошибка", err)

    worker.progress.connect(on_progress)
    worker.finished_ok.connect(on_finished)
    worker.failed.connect(on_failed)
    progress.canceled.connect(worker.cancel)
    worker.finished.connect(worker.deleteLater)
    worker.start()


def format_done_msg(
    result: PipelineResult,
    *,
    total_files: int,
    files_label: str,
    rows_label: str,
    out_dir: Path,
) -> str:
    """Стандартный текст итогового MessageBox: `total_files` файлов,
    `n_ok` строк в результате, перечисление первых ошибок."""
    msg = (f"{files_label}: {total_files}\n"
           f"{rows_label}: {result.n_ok}\n"
           f"Результаты: {out_dir}")
    if result.errors:
        msg += f"\n\nОшибки ({len(result.errors)}):\n" + "\n".join(result.errors[:5])
    return msg
