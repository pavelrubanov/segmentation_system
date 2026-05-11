"""Точка входа приложения."""
import argparse
import io
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.paths import resource_path

# В windowed-сборке sys.stdout/stderr == None --- любой print/tqdm/warning
# падает на None.write(). Шлём в файлы рядом с exe (заодно есть логи
# для разбора инцидентов).
if getattr(sys, "frozen", False):
    log_dir = Path(sys.executable).parent
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        sys.stdout = open(log_dir / "stdout.log", "a", buffering=1,
                          encoding="utf-8", errors="replace")
        sys.stderr = open(log_dir / "stderr.log", "a", buffering=1,
                          encoding="utf-8", errors="replace")
        sys.stdout.write(f"\n=== {stamp} ===\n")
        sys.stderr.write(f"\n=== {stamp} ===\n")
    except OSError:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

# на Windows torch обязан загрузить свои DLL до PyQt6,
# иначе c10.dll падает с WinError 1114
import torch  # noqa: F401
from PyQt6 import QtCore, QtWidgets
from ui.navigation_window import NavigationWindow
from ui.style import apply_theme


class _PredictorLoader(QtCore.QThread):
    """Фоновая загрузка MobileSAM, чтобы окно показалось мгновенно."""
    ready = QtCore.pyqtSignal(object)

    def __init__(self, checkpoint: str):
        super().__init__()
        self._checkpoint = checkpoint

    def run(self) -> None:
        from core.predictor import Predictor
        self.ready.emit(Predictor(self._checkpoint))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(resource_path("models/mobile_sam.pt")),
        help="Путь к весам MobileSAM",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"[ERR] веса не найдены: {args.checkpoint}")

    app = QtWidgets.QApplication(sys.argv)
    apply_theme(app)
    win = NavigationWindow()
    win.show()

    loader = _PredictorLoader(args.checkpoint)
    loader.ready.connect(win.set_predictor)
    loader.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
