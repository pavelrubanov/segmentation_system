#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal interactive SAM UI (MobileSAM version)

Фичи:
  - правая кнопка мыши = ПОЛОЖИТЕЛЬНАЯ точка (зелёная)
  - левая кнопка мыши  = ОТРИЦАТЕЛЬНАЯ точка (красная)
  - Load Image: загрузить изображение
  - Clean Points: очистить точки и маску
  - Save Mask: сохранить текущую маску в PNG (0/255)

Зависимости:
  pip install torch torchvision
  pip install git+https://github.com/ChaoningZhang/MobileSAM.git
  pip install PyQt5 matplotlib pillow numpy

Запуск:
  python sam_ui.py --checkpoint weights/mobile_sam.pt
"""

import sys
import os
import argparse

import torch
import numpy as np
from PIL import Image

from PyQt5 import QtWidgets, QtCore

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# MobileSAM imports (интерфейс тот же, что у оригинального Segment Anything)
# см. README в MobileSAM: sam_model_registry, SamPredictor, model_type="vit_t",
# checkpoint="./weights/mobile_sam.pt" :contentReference[oaicite:9]{index=9}
from mobile_sam import sam_model_registry, SamPredictor


class ImageCanvas(FigureCanvas):
    """
    Matplotlib canvas внутри PyQt5.
    Показывает картинку, точки и текущую маску.
    Ловит клики мыши и кидает наверх (в MainWindow.add_point).
    """

    def __init__(self, controller):
        self.controller = controller

        fig = Figure()
        self.ax = fig.add_subplot(111)
        self.ax.set_axis_off()

        super().__init__(fig)
        # запретить дефолтное контекстное меню матплотлиба на правый клик
        self.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)

        # подписываемся на клики
        self.mpl_connect("button_press_event", self.on_click)

        # данные для отображения
        self.image = None       # np.uint8 [H,W,3]
        self.mask = None        # np.uint8 [H,W] 0/1
        self.pos_points = []    # list of (x,y)
        self.neg_points = []    # list of (x,y)

    def on_click(self, event):
        """
        Mouse click handler.
        event.button: 1=левый, 3=правый (на большинстве систем)
        Мы интерпретируем:
          правый -> положительная точка (label=1)
          левый  -> отрицательная точка (label=0)
        """
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = float(event.xdata)
        y = float(event.ydata)

        # right click -> positive (green), left -> negative (red)
        if event.button == 3:
            # положительная
            self.controller.add_point((x, y), positive=True)
        elif event.button == 1:
            # отрицательная
            self.controller.add_point((x, y), positive=False)

    def set_image(self, img_np):
        """
        Установить новое изображение и сбросить все аннотации.
        img_np: np.uint8 [H,W,3] RGB
        """
        self.image = img_np
        self.mask = None
        self.pos_points = []
        self.neg_points = []
        self.redraw()

    def update_overlay(self, mask, pos_points, neg_points):
        """
        Обновить отображаемую маску и точки.
        mask: np.uint8 [H,W] (0/1) или None
        pos_points/neg_points: списки (x,y)
        """
        self.mask = mask
        self.pos_points = list(pos_points)
        self.neg_points = list(neg_points)
        self.redraw()

    def clear_points_and_mask(self):
        """
        Удалить точки и маску, но картинку не трогать.
        """
        self.mask = None
        self.pos_points = []
        self.neg_points = []
        self.redraw()

    def redraw(self):
        """
        Перерисовать ось.
        """
        self.ax.clear()
        self.ax.set_axis_off()

        if self.image is not None:
            self.ax.imshow(self.image)

        if self.mask is not None:
            # наложим маску полупрозрачно
            # маска у нас бинарная (0/1), превратим в 0..1 alpha.
            self.ax.imshow(
                np.ma.masked_where(self.mask == 0, self.mask),
                alpha=0.5,
                cmap="jet",
                interpolation="nearest",
            )

        # положительные точки (зелёные кружки)
        if len(self.pos_points) > 0:
            xs = [p[0] for p in self.pos_points]
            ys = [p[1] for p in self.pos_points]
            self.ax.scatter(
                xs,
                ys,
                c="lime",
                marker="o",
                s=60,
                edgecolors="black",
                linewidths=1.0,
            )

        # отрицательные точки (красные крестики)
        if len(self.neg_points) > 0:
            xs = [p[0] for p in self.neg_points]
            ys = [p[1] for p in self.neg_points]
            self.ax.scatter(
                xs,
                ys,
                c="red",
                marker="x",
                s=60,
                linewidths=2.0,
            )

        self.draw()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, predictor, device):
        super().__init__()

        self.setWindowTitle("MobileSAM UI")

        self.predictor = predictor
        self.device = device

        # состояние
        self.image_np = None  # np.uint8 [H,W,3]
        self.pos_points = []  # [(x,y), ...] (right clicks)
        self.neg_points = []  # [(x,y), ...] (left clicks)
        self.current_mask = None  # np.uint8 [H,W] 0/1
        self.image_is_set = False  # был ли predictor.set_image уже вызван

        # === UI элементы ===
        self.canvas = ImageCanvas(controller=self)

        self.load_btn = QtWidgets.QPushButton("Load Image")
        self.clean_btn = QtWidgets.QPushButton("Clean Points")
        self.save_btn = QtWidgets.QPushButton("Save Mask")

        self.info_label = QtWidgets.QLabel(
            "Правый клик = положительная точка (зелёная)\n"
            "Левый клик = отрицательная точка (красная)"
        )
        self.info_label.setWordWrap(True)

        # привязываем сигналы
        self.load_btn.clicked.connect(self.on_load_image)
        self.clean_btn.clicked.connect(self.on_clean)
        self.save_btn.clicked.connect(self.on_save_mask)

        # layout справа (кнопки)
        side_layout = QtWidgets.QVBoxLayout()
        side_layout.addWidget(self.load_btn)
        side_layout.addWidget(self.clean_btn)
        side_layout.addWidget(self.save_btn)
        side_layout.addSpacing(20)
        side_layout.addWidget(self.info_label)
        side_layout.addStretch(1)

        side_widget = QtWidgets.QWidget()
        side_widget.setLayout(side_layout)

        # общий layout
        main_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(main_widget)
        main_layout.addWidget(self.canvas, stretch=1)
        main_layout.addWidget(side_widget, stretch=0)

        self.setCentralWidget(main_widget)
        self.resize(1200, 800)

    # ---------- Логика ----------

    def on_load_image(self):
        """
        Диалог выбора изображения. Загружаем, конвертим в RGB np.uint8,
        скармливаем predictor.set_image(...)
        """
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Выбери изображение", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if fname is None or fname == "":
            return

        pil_img = Image.open(fname).convert("RGB")
        self.image_np = np.array(pil_img)  # H,W,3 uint8

        # передаём в predictor
        self.predictor.set_image(self.image_np)
        self.image_is_set = True

        # сбрасываем состояние точек/маски
        self.pos_points = []
        self.neg_points = []
        self.current_mask = None

        # перерисовать канвас
        self.canvas.set_image(self.image_np)

    def on_clean(self):
        """
        Удалить все точки и текущую маску, но оставить изображение.
        """
        self.pos_points = []
        self.neg_points = []
        self.current_mask = None

        self.canvas.clear_points_and_mask()

    def on_save_mask(self):
        """
        Сохраняем текущую маску как PNG (0/255).
        """
        if self.current_mask is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Нет маски",
                "Сначала кликни по изображению, чтобы получить маску.",
            )
            return

        save_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Сохранить маску как...",
            "mask.png",
            "PNG (*.png);;All Files (*)",
        )
        if save_name is None or save_name == "":
            return

        mask_img = (self.current_mask * 255).astype(np.uint8)
        Image.fromarray(mask_img).save(save_name)

    def add_point(self, xy, positive):
        """
        Вызывается canvas-ом при клике.
        xy = (x, y) в координатах изображения.
        positive: True (правый клик) -> label=1
                  False (левый клик) -> label=0
        """
        if self.image_np is None:
            return  # нет картинки - нечего делать

        if positive:
            self.pos_points.append(xy)
        else:
            self.neg_points.append(xy)

        # после каждого нового поинта пересчитать сегментацию
        self.run_segmentation()
        # обновить отрисовку
        self.canvas.update_overlay(
            self.current_mask, self.pos_points, self.neg_points
        )

    def run_segmentation(self):
        """
        Прогоняем MobileSAM по текущим точкам и получаем маску.
        Используем тот же API, что у оригинального SAM:
        predictor.predict(point_coords=..., point_labels=..., multimask_output=True)
        Возвращает несколько масок и скор; берём лучшую по score. :contentReference[oaicite:10]{index=10}
        """
        if not self.image_is_set:
            # защита: изображение не было подано в predictor.set_image
            if self.image_np is not None:
                self.predictor.set_image(self.image_np)
                self.image_is_set = True
            else:
                return

        # Если нет ни одной точки - просто убираем маску.
        pts_all = self.pos_points + self.neg_points
        if len(pts_all) == 0:
            self.current_mask = None
            return

        # Готовим входы
        # SAM ожидает np.array([[x,y], ...], dtype=np.float32),
        # labels: 1 для положительных, 0 для отрицательных.
        point_coords = np.array(
            pts_all,
            dtype=np.float32,
        )
        point_labels = np.array(
            [1] * len(self.pos_points) + [0] * len(self.neg_points),
            dtype=np.int32,
        )

        with torch.no_grad():
            # multimask_output=True -> SAM/MobileSAM вернёт до 3 вариантов масок
            # + их score; мы выберем маску с максимальным score
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

        # masks: [N, H, W] bool/uint8
        # scores: [N] float
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]

        # гарантируем uint8 {0,1}
        if best_mask.dtype != np.uint8:
            best_mask = best_mask.astype(np.uint8)

        self.current_mask = best_mask


def load_predictor(checkpoint_path: str):
    """
    Загружаем MobileSAM в память, как описано в README:
        model_type = "vit_t"
        sam_checkpoint = "./weights/mobile_sam.pt"
        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        mobile_sam.to(device)
        mobile_sam.eval()
        predictor = SamPredictor(mobile_sam)
    :contentReference[oaicite:11]{index=11}
    """
    model_type = "vit_t"  # MobileSAM = tiny ViT encoder ("vit_t") в офиц. репо. :contentReference[oaicite:12]{index=12}
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
        required=True,
        help="Путь к весам MobileSAM (mobile_sam.pt)",
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
