#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal interactive SAM UI (MobileSAM version) + points + boxes одновременно

Управление:
  - ЛКМ клик      = ПОЛОЖИТЕЛЬНАЯ точка (зелёная)
  - ПКМ клик      = ОТРИЦАТЕЛЬНАЯ точка (красная)
  - ЛКМ drag      = BOX (жёлтый прямоугольник)
  - Load Image    = загрузить изображение
  - Clean         = очистить точки/box/маску
  - Save Mask     = сохранить текущую маску в PNG (0/255)

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
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as mpatches

from mobile_sam import sam_model_registry, SamPredictor


class ImageCanvas(FigureCanvas):
    """
    Matplotlib canvas внутри PyQt5.
    Показывает картинку, точки, box и текущую маску.
    - ЛКМ drag: RectangleSelector -> box
    - ЛКМ/ПКМ click: точки (через press+release, чтобы не ставить точку при drag)
    """

    def __init__(self, controller):
        self.controller = controller

        fig = Figure()
        self.ax = fig.add_subplot(111)
        self.ax.set_axis_off()
        super().__init__(fig)

        self.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)

        # состояние для click-vs-drag
        self._press = None  # (x_px, y_px, button, xdata, ydata)

        # события мыши
        self.mpl_connect("button_press_event", self.on_press)
        self.mpl_connect("button_release_event", self.on_release)

        # данные
        self.image = None       # np.uint8 [H,W,3]
        self.mask = None        # np.uint8 [H,W] 0/1
        self.pos_points = []    # list[(x,y)]
        self.neg_points = []    # list[(x,y)]
        self.box = None         # (x0,y0,x1,y1) or None

        # artists (чтобы не делать ax.clear() каждый раз)
        self.img_artist = None
        self.mask_artist = None
        self.pos_scatter = None
        self.neg_scatter = None
        self.box_patch = None

        # RectangleSelector всегда активен, только ЛКМ-drag
        # minspanx/minspany в координатах данных (тут ~ пиксели изображения)
        self.rect_selector = RectangleSelector(
            self.ax,
            self.on_box_select,
            useblit=True,
            button=[1],
            interactive=False,
            spancoords="data",
            minspanx=5,
            minspany=5,
        )
        self.rect_selector.set_active(True)

    # ---------- Input handling ----------

    def on_press(self, event):
        if event.inaxes != self.ax:
            self._press = None
            return
        if event.xdata is None or event.ydata is None:
            self._press = None
            return
        # event.x/event.y — пиксели экрана; удобно мерять смещение
        self._press = (float(event.x), float(event.y), int(event.button), float(event.xdata), float(event.ydata))

    def on_release(self, event):
        """
        Ставим точки только если это клик (малое смещение).
        Если это drag ЛКМ, RectangleSelector сам вызовет on_box_select(...)
        """
        if self._press is None:
            return
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x0_px, y0_px, button, _, _ = self._press
        dx = abs(float(event.x) - x0_px)
        dy = abs(float(event.y) - y0_px)

        CLICK_THR_PX = 3.0
        is_click = (dx <= CLICK_THR_PX and dy <= CLICK_THR_PX)
        if not is_click:
            return  # drag

        x = float(event.xdata)
        y = float(event.ydata)

        # 1=ЛКМ -> positive, 3=ПКМ -> negative
        if button == 1:
            self.controller.add_point((x, y), positive=True)
        elif button == 3:
            self.controller.add_point((x, y), positive=False)

    def on_box_select(self, eclick, erelease):
        if eclick.xdata is None or eclick.ydata is None:
            return
        if erelease.xdata is None or erelease.ydata is None:
            return

        x0, y0 = float(eclick.xdata), float(eclick.ydata)
        x1, y1 = float(erelease.xdata), float(erelease.ydata)

        xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
        ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)

        # safety: хотя minspan уже стоит, фильтр оставим
        if abs(xmax - xmin) < 2 or abs(ymax - ymin) < 2:
            return

        self.controller.set_box((xmin, ymin, xmax, ymax))

    # ---------- Rendering ----------

    def _ensure_artists(self):
        if self.image is None:
            return

        if self.img_artist is None:
            self.img_artist = self.ax.imshow(self.image)
            self.ax.set_axis_off()

        if self.mask_artist is None:
            dummy = np.ma.masked_where(np.zeros(self.image.shape[:2], dtype=np.uint8) == 0,
                                       np.zeros(self.image.shape[:2], dtype=np.uint8))
            self.mask_artist = self.ax.imshow(
                dummy, alpha=0.5, cmap="jet", interpolation="nearest", visible=False
            )

        if self.pos_scatter is None:
            self.pos_scatter = self.ax.scatter(
                [], [], c="lime", marker="o", s=60, edgecolors="black", linewidths=1.0
            )

        if self.neg_scatter is None:
            self.neg_scatter = self.ax.scatter(
                [], [], c="red", marker="x", s=60, linewidths=2.0
            )

        if self.box_patch is None:
            self.box_patch = mpatches.Rectangle(
                (0, 0), 1, 1, fill=False, linewidth=2.0, edgecolor="yellow", visible=False
            )
            self.ax.add_patch(self.box_patch)

    def set_image(self, img_np):
        self.image = img_np
        self.mask = None
        self.pos_points = []
        self.neg_points = []
        self.box = None

        # сброс artists (проще и надежнее при смене размера картинки)
        self.ax.clear()
        self.ax.set_axis_off()
        self.img_artist = None
        self.mask_artist = None
        self.pos_scatter = None
        self.neg_scatter = None
        self.box_patch = None

        # RectangleSelector продолжает быть привязан к self.ax (ось та же),
        # но после clear() его визуальный прямоугольник исчезает — это нормально.

        self._ensure_artists()
        self._render()
        self.draw()

    def update_overlay(self, mask, pos_points, neg_points, box):
        self.mask = mask
        self.pos_points = list(pos_points)
        self.neg_points = list(neg_points)
        self.box = box
        self._ensure_artists()
        self._render()
        self.draw_idle()

    def clear_points_and_mask(self):
        self.mask = None
        self.pos_points = []
        self.neg_points = []
        self.box = None
        self._ensure_artists()
        self._render()
        self.draw_idle()

    def _render(self):
        if self.image is None:
            return

        # image
        self.img_artist.set_data(self.image)

        # mask
        if self.mask is None:
            self.mask_artist.set_visible(False)
        else:
            m = np.ma.masked_where(self.mask == 0, self.mask)
            self.mask_artist.set_data(m)
            self.mask_artist.set_visible(True)

        # points
        if self.pos_points:
            xs = [p[0] for p in self.pos_points]
            ys = [p[1] for p in self.pos_points]
            self.pos_scatter.set_offsets(np.column_stack([xs, ys]))
        else:
            self.pos_scatter.set_offsets(np.zeros((0, 2)))

        if self.neg_points:
            xs = [p[0] for p in self.neg_points]
            ys = [p[1] for p in self.neg_points]
            self.neg_scatter.set_offsets(np.column_stack([xs, ys]))
        else:
            self.neg_scatter.set_offsets(np.zeros((0, 2)))

        # box
        if self.box is None:
            self.box_patch.set_visible(False)
        else:
            x0, y0, x1, y1 = self.box
            self.box_patch.set_bounds(x0, y0, x1 - x0, y1 - y0)
            self.box_patch.set_visible(True)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, predictor, device):
        super().__init__()
        self.setWindowTitle("MobileSAM UI")

        self.predictor = predictor
        self.device = device

        # состояние
        self.image_np = None
        self.pos_points = []
        self.neg_points = []
        self.box = None                 # (x0,y0,x1,y1) или None
        self.current_mask = None        # np.uint8 [H,W] 0/1
        self.image_is_set = False

        # UI
        self.canvas = ImageCanvas(controller=self)

        self.load_btn = QtWidgets.QPushButton("Load Image")
        self.clean_btn = QtWidgets.QPushButton("Clean")
        self.save_btn = QtWidgets.QPushButton("Save Mask")

        self.info_label = QtWidgets.QLabel(
            "ЛКМ клик = положительная точка (зелёная)\n"
            "ПКМ клик = отрицательная точка (красная)\n"
            "ЛКМ drag = box (жёлтый прямоугольник)"
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
        if not fname:
            return

        pil_img = Image.open(fname).convert("RGB")
        self.image_np = np.array(pil_img)

        self.predictor.set_image(self.image_np)
        self.image_is_set = True

        self.pos_points = []
        self.neg_points = []
        self.box = None
        self.current_mask = None

        self.canvas.set_image(self.image_np)

    def on_clean(self):
        self.pos_points = []
        self.neg_points = []
        self.box = None
        self.current_mask = None
        self.canvas.clear_points_and_mask()

    def on_save_mask(self):
        if self.current_mask is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Нет маски",
                "Сначала добавь точки и/или box, чтобы получить маску.",
            )
            return

        save_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Сохранить маску как...",
            "mask.png",
            "PNG (*.png);;All Files (*)",
        )
        if not save_name:
            return

        mask_img = (self.current_mask * 255).astype(np.uint8)
        Image.fromarray(mask_img).save(save_name)

    # ---------- Prompts ----------

    def add_point(self, xy, positive: bool):
        if self.image_np is None:
            return

        if positive:
            self.pos_points.append(xy)
        else:
            self.neg_points.append(xy)

        self.run_segmentation()
        self.canvas.update_overlay(self.current_mask, self.pos_points, self.neg_points, self.box)

    def set_box(self, box_xyxy):
        if self.image_np is None:
            return

        self.box = box_xyxy
        self.run_segmentation()
        self.canvas.update_overlay(self.current_mask, self.pos_points, self.neg_points, self.box)

    # ---------- Segmentation ----------

    def run_segmentation(self):
        if not self.image_is_set:
            if self.image_np is not None:
                self.predictor.set_image(self.image_np)
                self.image_is_set = True
            else:
                return

        has_points = (len(self.pos_points) + len(self.neg_points)) > 0
        has_box = self.box is not None

        if not has_points and not has_box:
            self.current_mask = None
            return

        point_coords = None
        point_labels = None
        if has_points:
            pts_all = self.pos_points + self.neg_points
            point_coords = np.array(pts_all, dtype=np.float32)
            point_labels = np.array(
                [1] * len(self.pos_points) + [0] * len(self.neg_points),
                dtype=np.int32,
            )

        box_np = None
        if has_box:
            box_np = np.array(self.box, dtype=np.float32)  # XYXY

        with torch.no_grad():
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_np,
                multimask_output=True,
            )

        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx]
        if best_mask.dtype != np.uint8:
            best_mask = best_mask.astype(np.uint8)

        self.current_mask = best_mask


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
        required=True,
        help="Путь к весам MobileSAM (mobile_sam.pt / .pth)",
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
