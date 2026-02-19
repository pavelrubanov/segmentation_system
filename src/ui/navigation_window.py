from PyQt6 import QtWidgets
from .segmentation.window import Window as SegmentationWindow
from .processing_window import run_processing


class NavigationWindow(QtWidgets.QMainWindow):
    """Основное навигационное окно приложения"""

    def __init__(self, predictor):
        super().__init__()
        self.setWindowTitle("Segmentation System - Main")
        self.predictor = predictor
        self.segmentation_window = None

        # Кнопки
        self.segment_images_btn = QtWidgets.QPushButton("Segment images")
        self.process_images_btn = QtWidgets.QPushButton("Process masks")
        self.leave_btn = QtWidgets.QPushButton("Leave")

        # Привязка сигналов
        self.segment_images_btn.clicked.connect(self.on_segment_images)
        self.process_images_btn.clicked.connect(self.on_process_images)
        self.leave_btn.clicked.connect(self.close)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.segment_images_btn)
        layout.addWidget(self.process_images_btn)
        layout.addStretch()
        layout.addWidget(self.leave_btn)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.resize(300, 200)

    def on_segment_images(self):
        """Открывает окно сегментации изображений"""
        if self.segmentation_window is None or not self.segmentation_window.isVisible():
            self.segmentation_window = SegmentationWindow(self.predictor, self)
            self.segmentation_window.show()

    def on_process_images(self):
        """Запускает обработку масок"""
        run_processing(self)




