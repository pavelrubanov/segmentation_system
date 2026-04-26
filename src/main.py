"""Точка входа приложения."""
import argparse
import os
import sys
from pathlib import Path

# чтобы можно было импортировать пакет classifier (он соседом к src/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.predictor import Predictor
from PyQt6 import QtWidgets
from ui.navigation_window import NavigationWindow
from ui.style import apply_theme


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="../models/mobile_sam.pt",
        help="Путь к весам MobileSAM",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"[ERR] веса не найдены: {args.checkpoint}")

    predictor = Predictor(args.checkpoint)

    app = QtWidgets.QApplication(sys.argv)
    apply_theme(app)
    win = NavigationWindow(predictor)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
