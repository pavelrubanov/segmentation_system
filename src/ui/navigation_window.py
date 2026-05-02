from PyQt6 import QtWidgets, QtCore, QtGui
from .segmentation.window import Window as SegmentationWindow
from .measure_crops import measure_crops
from .measure_images import measure_images


class NavigationWindow(QtWidgets.QMainWindow):
    """Навигационное окно — экран приветствия."""

    def __init__(self, predictor):
        super().__init__()
        self.setWindowTitle("Морфометрия листьев")
        self.predictor = predictor
        self.segmentation_window = None

        # заголовок
        title = QtWidgets.QLabel("Морфометрия листьев")
        title.setObjectName("heading")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        subtitle = QtWidgets.QLabel("Сегментация и параметрический анализ")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # кнопки-карточки
        sp = self.style().standardIcon

        self.segment_btn = QtWidgets.QPushButton("  Интерактивная сегментация")
        self.segment_btn.setObjectName("card")
        self.segment_btn.setIcon(sp(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton))
        self.segment_btn.setIconSize(QtCore.QSize(28, 28))
        self.segment_btn.setMinimumHeight(64)
        self.segment_btn.setToolTip("Интерактивная сегментация с MobileSAM")
        self.segment_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        self.process_btn = QtWidgets.QPushButton("  Параметрические измерения листьев")
        self.process_btn.setObjectName("card")
        self.process_btn.setIcon(sp(QtWidgets.QStyle.StandardPixmap.SP_FileDialogDetailedView))
        self.process_btn.setIconSize(QtCore.QSize(28, 28))
        self.process_btn.setMinimumHeight(64)
        self.process_btn.setToolTip("Пакетные измерения по готовым кропам и экспорт в CSV/XLSX")
        self.process_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        self.auto_btn = QtWidgets.QPushButton("  Авто-сегментация и измерения")
        self.auto_btn.setObjectName("card")
        self.auto_btn.setIcon(sp(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay))
        self.auto_btn.setIconSize(QtCore.QSize(28, 28))
        self.auto_btn.setMinimumHeight(64)
        self.auto_btn.setToolTip("Сегментация исходных фото, отбор классификатором, замеры, выгрузка в Excel")
        self.auto_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        self.quit_btn = QtWidgets.QPushButton("Выход")
        self.quit_btn.setObjectName("flat")
        self.quit_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        # сигналы
        self.segment_btn.clicked.connect(self.on_segment_images)
        self.process_btn.clicked.connect(self.on_process_images)
        self.auto_btn.clicked.connect(self.on_auto_process)
        self.quit_btn.clicked.connect(self.close)

        # лейаут
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(32, 28, 32, 20)
        layout.setSpacing(8)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(24)
        layout.addWidget(self.segment_btn)
        layout.addWidget(self.process_btn)
        layout.addWidget(self.auto_btn)
        layout.addStretch()
        layout.addWidget(self.quit_btn, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.resize(480, 440)

    def on_segment_images(self):
        if self.segmentation_window is None or not self.segmentation_window.isVisible():
            self.segmentation_window = SegmentationWindow(self.predictor, self)
            self.segmentation_window.show()

    def on_process_images(self):
        measure_crops(self)

    def on_auto_process(self):
        measure_images(self, self.predictor)
