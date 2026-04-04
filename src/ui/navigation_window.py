from PyQt6 import QtWidgets, QtCore, QtGui
from .segmentation.window import Window as SegmentationWindow
from .processing_window import run_processing


class NavigationWindow(QtWidgets.QMainWindow):
    """Навигационное окно — экран приветствия."""

    def __init__(self, predictor):
        super().__init__()
        self.setWindowTitle("Морфометрия листьев")
        self.predictor = predictor
        self.segmentation_window = None

        # ── Заголовок ─────────────────────────────────────────────────────
        title = QtWidgets.QLabel("Морфометрия листьев")
        title.setObjectName("heading")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        subtitle = QtWidgets.QLabel("Сегментация и параметрический анализ")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # ── Кнопки-карточки ───────────────────────────────────────────────
        sp = self.style().standardIcon

        self.segment_btn = QtWidgets.QPushButton("  Сегментация изображений")
        self.segment_btn.setObjectName("card")
        self.segment_btn.setIcon(sp(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton))
        self.segment_btn.setIconSize(QtCore.QSize(28, 28))
        self.segment_btn.setMinimumHeight(64)
        self.segment_btn.setToolTip("Интерактивная сегментация с MobileSAM")
        self.segment_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        self.process_btn = QtWidgets.QPushButton("  Обработка масок")
        self.process_btn.setObjectName("card")
        self.process_btn.setIcon(sp(QtWidgets.QStyle.StandardPixmap.SP_FileDialogDetailedView))
        self.process_btn.setIconSize(QtCore.QSize(28, 28))
        self.process_btn.setMinimumHeight(64)
        self.process_btn.setToolTip("Пакетные измерения и экспорт в CSV")
        self.process_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        self.quit_btn = QtWidgets.QPushButton("Выход")
        self.quit_btn.setObjectName("flat")
        self.quit_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        # ── Сигналы ───────────────────────────────────────────────────────
        self.segment_btn.clicked.connect(self.on_segment_images)
        self.process_btn.clicked.connect(self.on_process_images)
        self.quit_btn.clicked.connect(self.close)

        # ── Layout ────────────────────────────────────────────────────────
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(32, 28, 32, 20)
        layout.setSpacing(8)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(24)
        layout.addWidget(self.segment_btn)
        layout.addWidget(self.process_btn)
        layout.addStretch()
        layout.addWidget(self.quit_btn, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.resize(480, 360)

    def on_segment_images(self):
        if self.segmentation_window is None or not self.segmentation_window.isVisible():
            self.segmentation_window = SegmentationWindow(self.predictor, self)
            self.segmentation_window.show()

    def on_process_images(self):
        run_processing(self)
