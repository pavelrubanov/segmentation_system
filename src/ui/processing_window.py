import csv
from pathlib import Path

import numpy as np
from PIL import Image
from PyQt6 import QtWidgets, QtCore

from core.leaf_measure import measure_leaf, save_measurement_visualization


def run_processing(parent: QtWidgets.QWidget):
    """Пакетная обработка масок: измерение + визуализация + CSV."""

    # 1) Выбор файлов масок
    files, _ = QtWidgets.QFileDialog.getOpenFileNames(
        parent, "Выберите маски", "",
        "Mask images (*_mask_*.png *_mask_*.jpg *_mask_*.bmp *_mask_*.tif)")
    if not files:
        return

    # 2) Выбор выходной директории
    out_dir = QtWidgets.QFileDialog.getExistingDirectory(parent, "Выберите выходную директорию")
    if not out_dir:
        return
    out_path = Path(out_dir)

    # 3) Прогресс-бар
    progress = QtWidgets.QProgressDialog("Обработка масок…", "Отмена", 0, len(files), parent)
    progress.setWindowTitle("Обработка")
    progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(0)

    fractions = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
    rows = []

    for i, fpath in enumerate(files):
        if progress.wasCanceled():
            break
        progress.setValue(i)
        QtWidgets.QApplication.processEvents()

        name = Path(fpath).stem
        try:
            mask = np.array(Image.open(fpath).convert("L"))
            metrics = measure_leaf(mask, fractions=fractions)

            # Сохранение визуализации
            save_measurement_visualization(
                mask=mask,
                metrics=metrics,
                out_path=str(out_path / f"{name}_vis.png"),
            )

            # Собираем строку для CSV
            row = {
                "image": name,
                "length_px": f"{metrics.length_px:.1f}",
                "width_px": f"{metrics.width_px:.1f}",
            }
            for f in fractions:
                sec = metrics.widths.get(f)
                row[f"width_{f:.3f}"] = f"{sec.width_px:.1f}" if sec else "0.0"
            rows.append(row)

        except Exception as exc:
            QtWidgets.QMessageBox.warning(parent, "Ошибка", f"{name}: {exc}")

    progress.setValue(len(files))

    # 4) Сохранение CSV
    if rows:
        csv_path = out_path / "measurements.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    # 5) Сообщение об успехе
    QtWidgets.QMessageBox.information(
        parent, "Готово",
        f"Обработано: {len(rows)} из {len(files)}\nРезультаты: {out_path}")

