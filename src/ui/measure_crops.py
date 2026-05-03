"""Замеры по готовым кропам: батч-обработка PNG-кропов, экспорт CSV/XLSX."""
from PyQt6 import QtWidgets

from core.file_naming import crop_file_filter
from core.io import make_run_dir
from core.sources import disk_source

from .runner import (
    FilesPicker,
    PipelineWorker,
    SettingsDialog,
    format_done_msg,
    run_with_progress,
)


def measure_crops(parent: QtWidgets.QWidget) -> None:
    """Открыть диалог настроек и запустить батч-замеры по выбранным кропам."""
    dlg = SettingsDialog(
        title="Замеры по кропам",
        files_picker=FilesPicker(
            "Кропы листа",
            "Выберите препроцессированные кропы листьев",
            crop_file_filter(),
            ("кроп", "кропа", "кропов"),
        ),
        default_format="csv",
        parent=parent,
    )
    if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return

    s = dlg.pipeline_settings()
    s.out_dir = make_run_dir(s.out_dir, "measure")
    worker = PipelineWorker(files=s.files, source=disk_source, settings=s)

    def msg(result):
        return format_done_msg(
            result, total_files=len(s.files),
            files_label="Обработано", rows_label="Получено строк",
            out_dir=s.out_dir,
        )

    run_with_progress(
        parent, worker,
        title="Замеры по кропам",
        label="Обработка кропов…",
        finished_msg=msg,
    )
