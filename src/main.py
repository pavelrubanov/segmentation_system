"""Точка входа приложения."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# на Windows torch обязан загрузить свои DLL до PyQt6,
# иначе c10.dll падает с WinError 1114
import torch  # noqa: F401

from core.paths import resource_path
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
