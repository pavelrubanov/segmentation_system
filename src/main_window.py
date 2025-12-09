import numpy as np
from PIL import Image

from PyQt5 import QtWidgets
import matplotlib
matplotlib.use("Qt5Agg")

from image_canvas import ImageCanvas

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, predictor):
        super().__init__()
        self.setWindowTitle("Segmentation System")

        # UI
        self.canvas = ImageCanvas(predictor)

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
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Выбери изображение", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not fname:
            return

        pil_img = Image.open(fname).convert("RGB")
        self.canvas.set_image(np.array(pil_img))

    def on_clean(self):
        self.canvas.clean()
        self.canvas.redraw()

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