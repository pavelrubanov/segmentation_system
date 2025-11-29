import torch
import numpy as np
from PIL import Image

from PyQt5 import QtWidgets
import matplotlib
matplotlib.use("Qt5Agg")

from src.image_canvas import ImageCanvas


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
