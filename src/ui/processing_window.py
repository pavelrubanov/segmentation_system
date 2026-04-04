import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PyQt6 import QtWidgets, QtCore

from core.leaf_measure import measure_leaf, save_measurement_visualization


# ── Настройки обработки ─────────────────────────────────────────────────────


@dataclass
class ProcessingSettings:
    files: list[str]
    out_dir: Path
    n_sections: int
    unit_name: str
    px_per_unit: Optional[float]  # None → без пересчёта


class ProcessingSettingsDialog(QtWidgets.QDialog):
    """Диалог параметров пакетной обработки."""

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setWindowTitle("Параметры обработки")
        self.setMinimumWidth(480)
        self._files: list[str] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        # ── Файлы масок ─────────────────────────────────────────────
        grp_files = QtWidgets.QGroupBox("Маски")
        fl = QtWidgets.QHBoxLayout(grp_files)
        self._lbl_files = QtWidgets.QLabel("Файлы не выбраны")
        self._lbl_files.setObjectName("secondary")
        btn_files = QtWidgets.QPushButton("Выбрать…")
        btn_files.clicked.connect(self._pick_files)
        fl.addWidget(self._lbl_files, 1)
        fl.addWidget(btn_files)
        layout.addWidget(grp_files)

        # ── Выходная директория ─────────────────────────────────────
        grp_out = QtWidgets.QGroupBox("Выходная директория")
        ol = QtWidgets.QHBoxLayout(grp_out)
        self._lbl_out = QtWidgets.QLabel("Не выбрана")
        self._lbl_out.setObjectName("secondary")
        btn_out = QtWidgets.QPushButton("Выбрать…")
        btn_out.clicked.connect(self._pick_out_dir)
        ol.addWidget(self._lbl_out, 1)
        ol.addWidget(btn_out)
        layout.addWidget(grp_out)

        # ── Количество поперечных сечений ───────────────────────────
        grp_sec = QtWidgets.QGroupBox("Поперечные сечения")
        sl = QtWidgets.QHBoxLayout(grp_sec)
        sl.addWidget(QtWidgets.QLabel("Количество линий:"))
        self._spin_sections = QtWidgets.QSpinBox()
        self._spin_sections.setRange(1, 99)
        self._spin_sections.setValue(7)
        sl.addWidget(self._spin_sections)
        sl.addStretch()
        layout.addWidget(grp_sec)

        # ── Масштаб ─────────────────────────────────────────────────
        grp_scale = QtWidgets.QGroupBox("Масштаб")
        scl = QtWidgets.QGridLayout(grp_scale)

        self._chk_scale = QtWidgets.QCheckBox("Пересчитывать в единицы измерения")
        self._chk_scale.toggled.connect(self._on_scale_toggled)
        scl.addWidget(self._chk_scale, 0, 0, 1, 3)

        self._lbl_px = QtWidgets.QLabel("Пикселей на единицу:")
        self._spin_px = QtWidgets.QDoubleSpinBox()
        self._spin_px.setRange(0.01, 100_000.0)
        self._spin_px.setDecimals(2)
        self._spin_px.setValue(100.0)
        scl.addWidget(self._lbl_px, 1, 0)
        scl.addWidget(self._spin_px, 1, 1)

        self._lbl_unit = QtWidgets.QLabel("Единица:")
        self._edit_unit = QtWidgets.QLineEdit("мм")
        self._edit_unit.setMaximumWidth(80)
        scl.addWidget(self._lbl_unit, 2, 0)
        scl.addWidget(self._edit_unit, 2, 1)

        layout.addWidget(grp_scale)

        self._on_scale_toggled(False)

        # ── Кнопки ──────────────────────────────────────────────────
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
        layout.addLayout(btns)

    # ── Слоты ───────────────────────────────────────────────────────

    def _pick_files(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Выберите маски", "",
            "Mask images (*_mask_*.png *_mask_*.jpg *_mask_*.bmp *_mask_*.tif)")
        if files:
            self._files = files
            n = len(files)
            self._lbl_files.setText(self._pluralize_masks(n))
            self._update_ok()

    def _pick_out_dir(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите выходную директорию")
        if d:
            self._lbl_out.setText(d)
            self._update_ok()

    def _on_scale_toggled(self, on: bool) -> None:
        for w in (self._lbl_px, self._spin_px, self._lbl_unit, self._edit_unit):
            w.setEnabled(on)

    def _update_ok(self) -> None:
        self._btn_ok.setEnabled(bool(self._files) and self._lbl_out.text() != "Не выбрана")

    # ── Результат ───────────────────────────────────────────────────

    def settings(self) -> ProcessingSettings:
        use_scale = self._chk_scale.isChecked()
        return ProcessingSettings(
            files=self._files,
            out_dir=Path(self._lbl_out.text()),
            n_sections=self._spin_sections.value(),
            unit_name=self._edit_unit.text().strip() or "мм",
            px_per_unit=self._spin_px.value() if use_scale else None,
        )

    @staticmethod
    def _pluralize_masks(n: int) -> str:
        mod = n % 10
        mod100 = n % 100
        if mod == 1 and mod100 != 11:
            word = "маска"
        elif 2 <= mod <= 4 and not (12 <= mod100 <= 14):
            word = "маски"
        else:
            word = "масок"
        return f"Выбрано: {n} {word}"


# ── Пакетная обработка ──────────────────────────────────────────────────────


def _build_fractions(n: int) -> tuple[float, ...]:
    """Равномерные фракции внутри (0, 1) для n поперечных сечений."""
    return tuple(i / (n + 1) for i in range(1, n + 1))


def run_processing(parent: QtWidgets.QWidget):
    """Пакетная обработка масок: измерение + визуализация + CSV."""

    dlg = ProcessingSettingsDialog(parent)
    if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return

    s = dlg.settings()
    out_path = s.out_dir
    fractions = _build_fractions(s.n_sections)
    unit = s.unit_name if s.px_per_unit else "px"
    scale = 1.0 / s.px_per_unit if s.px_per_unit else 1.0

    # Прогресс-бар
    progress = QtWidgets.QProgressDialog("Обработка масок…", "Отмена", 0, len(s.files), parent)
    progress.setWindowTitle("Обработка")
    progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(0)

    rows: list[dict[str, str]] = []

    for i, fpath in enumerate(s.files):
        if progress.wasCanceled():
            break
        progress.setValue(i)
        QtWidgets.QApplication.processEvents()

        name = Path(fpath).stem
        try:
            mask = np.array(Image.open(fpath).convert("L"))
            metrics = measure_leaf(mask, fractions=fractions, unit=unit, scale=scale)

            save_measurement_visualization(
                mask=mask,
                metrics=metrics,
                out_path=str(out_path / f"{name}_vis.png"),
            )

            row: dict[str, str] = {
                "image": name,
                f"length_{unit}": f"{metrics.length:.2f}",
                f"width_{unit}": f"{metrics.width:.2f}",
            }
            for f in fractions:
                sec = metrics.widths.get(f)
                row[f"width_{f:.3f}_{unit}"] = f"{sec.width:.2f}" if sec else "0.00"
            rows.append(row)

        except Exception as exc:
            QtWidgets.QMessageBox.warning(parent, "Ошибка", f"{name}: {exc}")

    progress.setValue(len(s.files))

    # Сохранение CSV
    if rows:
        csv_path = out_path / "measurements.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    QtWidgets.QMessageBox.information(
        parent, "Готово",
        f"Обработано: {len(rows)} из {len(s.files)}\nРезультаты: {out_path}")

