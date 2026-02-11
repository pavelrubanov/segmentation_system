"""
Minimal interactive SAM UI (MobileSAM version) + points + boxes одновременно

Управление:
  - ЛКМ клик      = ПОЛОЖИТЕЛЬНАЯ точка (зелёная)
  - ПКМ клик      = ОТРИЦАТЕЛЬНАЯ точка (красная)
  - ЛКМ drag      = BOX (жёлтый прямоугольник)
  - Load Image    = загрузить изображение
  - Clean         = очистить точки/box/маску
  - Save Mask     = сохранить текущую маску в PNG (0/255)

Запуск:
  python sam_ui.py --checkpoint weights/mobile_sam.pt
"""

import sys
import os
import argparse
from core.predictor import Predictor
from PyQt6 import QtWidgets
from ui.navigation_window import NavigationWindow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Путь к весам MobileSAM (mobile_sam.pt / .pth)",
        default="../weights/mobile_sam.pt",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[ERR] checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    predictor = Predictor(args.checkpoint)

    app = QtWidgets.QApplication(sys.argv)
    win = NavigationWindow(predictor)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
