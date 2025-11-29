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
import torch
from PyQt5 import QtWidgets
import matplotlib
matplotlib.use("Qt5Agg")
from mobile_sam import sam_model_registry, SamPredictor
from src.main_window import MainWindow


def load_predictor(checkpoint_path: str):
    model_type = "vit_t"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mobile_sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    mobile_sam.to(device=device)
    mobile_sam.eval()

    predictor = SamPredictor(mobile_sam)
    return predictor, device


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

    predictor, device = load_predictor(args.checkpoint)

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(predictor, device)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
