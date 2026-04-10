import shutil
import numpy as np
from pathlib import Path

from PyQt6 import QtWidgets, QtCore, QtGui
from core.image_data import ImageWithMasks
from .masks_buffer import MasksBuffer
from .canvas import SegmentationCanvas
from ui.style import icon_crosshair, icon_brush, icon_eraser


class Window(QtWidgets.QMainWindow):
    def __init__(self, predictor, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Система сегментации")
        self.predictor = predictor

        self.images_data: list[ImageWithMasks] = []
        self.current_image_idx = 0

        sp = self.style().standardIcon

        # основные виджеты
        self.canvas = SegmentationCanvas(predictor)
        self.masks_buffer = MasksBuffer()

        # меню
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("Файл")
        file_menu.addAction(
            sp(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton),
            "Загрузить изображения",
            QtGui.QKeySequence("Ctrl+O"),
            self.on_load_images)
        file_menu.addSeparator()
        file_menu.addAction(
            sp(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton),
            "Сохранить маски и выйти",
            QtGui.QKeySequence("Ctrl+S"),
            self.on_save_and_leave)

        edit_menu = menu_bar.addMenu("Правка")
        edit_menu.addAction(
            sp(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload),
            "Очистить",
            QtGui.QKeySequence("Ctrl+Delete"),
            self.on_clean)

        # тулбар
        toolbar = QtWidgets.QToolBar("Инструменты")
        toolbar.setMovable(False)
        toolbar.setIconSize(QtCore.QSize(22, 22))
        self.addToolBar(toolbar)

        self._tool_group = QtWidgets.QButtonGroup(self)
        self._tool_group.setExclusive(True)

        self.pointer_btn = QtWidgets.QToolButton()
        self.pointer_btn.setIcon(icon_crosshair())
        self.pointer_btn.setToolTip("Режим промптов")
        self.pointer_btn.setCheckable(True)
        self.pointer_btn.setChecked(True)

        self.brush_btn = QtWidgets.QToolButton()
        self.brush_btn.setIcon(icon_brush())
        self.brush_btn.setToolTip("Кисть (дорисовать)")
        self.brush_btn.setCheckable(True)
        self.brush_btn.setEnabled(False)

        self.eraser_btn = QtWidgets.QToolButton()
        self.eraser_btn.setIcon(icon_eraser())
        self.eraser_btn.setToolTip("Ластик (стереть)")
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.setEnabled(False)

        self._tool_group.addButton(self.pointer_btn, 0)
        self._tool_group.addButton(self.brush_btn, 1)
        self._tool_group.addButton(self.eraser_btn, 2)

        self.brush_size_spin = QtWidgets.QSpinBox()
        self.brush_size_spin.setRange(1, 200)
        self.brush_size_spin.setValue(30)
        self.brush_size_spin.setPrefix("Размер: ")
        self.brush_size_spin.setFixedHeight(35)
        self.brush_size_spin.setFixedWidth(130)
        self.brush_size_spin.setEnabled(False)

        toolbar.addWidget(self.pointer_btn)
        toolbar.addWidget(self.brush_btn)
        toolbar.addWidget(self.eraser_btn)
        toolbar.addSeparator()
        toolbar.addWidget(self.brush_size_spin)

        # Spacer → навигация справа в тулбаре
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                             QtWidgets.QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self.prev_btn = QtWidgets.QToolButton()
        self.prev_btn.setIcon(sp(QtWidgets.QStyle.StandardPixmap.SP_ArrowLeft))
        self.prev_btn.setToolTip("Предыдущее изображение")
        self.prev_btn.setEnabled(False)

        self.nav_label = QtWidgets.QLabel("0/0")
        self.nav_label.setObjectName("secondary")
        self.nav_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.nav_label.setMinimumWidth(50)

        self.next_btn = QtWidgets.QToolButton()
        self.next_btn.setIcon(sp(QtWidgets.QStyle.StandardPixmap.SP_ArrowRight))
        self.next_btn.setToolTip("Следующее изображение")
        self.next_btn.setEnabled(False)

        toolbar.addWidget(self.prev_btn)
        toolbar.addWidget(self.nav_label)
        toolbar.addWidget(self.next_btn)

        # статус-бар
        self._status_image = QtWidgets.QLabel("")
        self._status_mode = QtWidgets.QLabel("Промпты")
        self.statusBar().addWidget(self._status_image, 1)
        self.statusBar().addPermanentWidget(self._status_mode)

        # боковая панель
        self.load_images_btn = QtWidgets.QPushButton(
            sp(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton),
            " Загрузить изображения")
        self.load_images_btn.setObjectName("primary")

        self.clean_btn = QtWidgets.QPushButton(
            sp(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload),
            " Очистить")

        self.save_to_buffer_btn = QtWidgets.QPushButton("Фиксировать маску")
        self.save_to_buffer_btn.setObjectName("primary")

        self.info_label = QtWidgets.QLabel()
        self.info_label.setObjectName("secondary")
        self.info_label.setWordWrap(True)

        self.save_and_leave_btn = QtWidgets.QPushButton(
            sp(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton),
            " Сохранить и выйти")
        self.save_and_leave_btn.setObjectName("success")

        side = QtWidgets.QVBoxLayout()
        side.setContentsMargins(12, 12, 12, 12)
        side.setSpacing(8)
        side.addWidget(self.load_images_btn)
        side.addWidget(self.clean_btn)
        side.addWidget(self.save_to_buffer_btn)
        side.addSpacing(12)
        side.addWidget(self.info_label)
        side.addStretch(1)
        side.addWidget(self.save_and_leave_btn)

        side_widget = QtWidgets.QWidget()
        side_widget.setObjectName("sidePanel")
        side_widget.setLayout(side)

        # три колонки: буфер масок | канвас | сайдбар
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self.masks_buffer)
        splitter.addWidget(self.canvas)
        splitter.addWidget(side_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([220, 1400, 260])

        self.setCentralWidget(splitter)
        self.resize(1920, 1080)

        # сигналы
        self.load_images_btn.clicked.connect(self.on_load_images)
        self.clean_btn.clicked.connect(self.on_clean)
        self.save_to_buffer_btn.clicked.connect(self.on_save_to_buffer)
        self.prev_btn.clicked.connect(self.on_previous_image)
        self.next_btn.clicked.connect(self.on_next_image)
        self.save_and_leave_btn.clicked.connect(self.on_save_and_leave)
        self._tool_group.idClicked.connect(self._on_tool_changed)
        self.brush_size_spin.valueChanged.connect(
            lambda v: self.canvas.set_brush_radius(v))
        self.masks_buffer.mask_edit_requested.connect(self._on_mask_edit_requested)

        self._update_edit_ui()

    # обработчики действий

    def on_load_images(self):
        fnames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Выбери изображения", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif)")
        if not fnames:
            return
        self.images_data = [
            ImageWithMasks(path=f, name=Path(f).name)
            for f in fnames
        ]
        self.current_image_idx = 0
        self._load_current_image()

    def on_clean(self):
        """Очищает canvas, но НЕ сбрасывает editing_index — позволяет пересегментировать."""
        self.canvas.clean()
        self.canvas.redraw()
        self._update_edit_ui()

    def on_save_to_buffer(self):
        """Двухшаговый сценарий: фиксация маски → редактирование → сохранение."""
        # Шаг 2: сохраняем отредактированную маску
        if self.canvas.edit_mode:
            mask = self.canvas.finish_edit()
            if mask is None:
                self._warn("Нет маски", "Не найдена отредактированная маска.")
                return
            self.masks_buffer.save_mask(mask)
            self.canvas.clean()
            self.canvas.redraw()
            self._update_edit_ui()
            return

        # Шаг 1: фиксируем маску предиктора → включаем edit mode
        if self.canvas.current_mask is None:
            self._warn("Нет маски",
                       "Сначала поставь точки и/или box, "
                       "чтобы получить маску предиктора.")
            return

        if not self.canvas.start_edit():
            self._warn("Ошибка",
                       "Не удалось зафиксировать маску для редактирования.")
            return
        self._update_edit_ui()

    def on_previous_image(self):
        if self.images_data and self.current_image_idx > 0:
            self.current_image_idx -= 1
            self._load_current_image()

    def on_next_image(self):
        if self.images_data and self.current_image_idx < len(self.images_data) - 1:
            self.current_image_idx += 1
            self._load_current_image()

    def on_save_and_leave(self):
        mask_paths = self.masks_buffer.all_mask_paths()
        if not mask_paths:
            self._warn("Нет масок", "Нет сохранённых масок для экспорта.")
            return

        save_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Выберите папку для сохранения масок")
        if not save_dir:
            return

        try:
            dest = Path(save_dir)
            for p in mask_paths:
                shutil.copy2(p, dest / p.name)
            QtWidgets.QMessageBox.information(
                self, "Успех",
                f"Сохранено масок: {len(mask_paths)}\nПапка: {dest}")
            self.close()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Ошибка", f"Ошибка при сохранении: {exc}")

    # приватные методы

    def _load_current_image(self):
        if not self.images_data:
            return
        data = self.images_data[self.current_image_idx]
        self.canvas.set_image(data.image)
        self.masks_buffer.set_image_name(data.name)
        self._update_nav()
        self._update_edit_ui()

    def _update_nav(self):
        total = len(self.images_data)
        cur = self.current_image_idx + 1 if self.images_data else 0
        self.nav_label.setText(f"{cur}/{total}")
        self.prev_btn.setEnabled(self.current_image_idx > 0)
        self.next_btn.setEnabled(self.current_image_idx < total - 1)
        name = self.images_data[self.current_image_idx].name if self.images_data else ""
        self._status_image.setText(f"  {name} — {cur}/{total}" if name else "")

    def _update_edit_ui(self):
        editing = self.canvas.edit_mode
        is_existing = self.masks_buffer.editing_index is not None

        self.pointer_btn.setEnabled(not editing)
        self.brush_btn.setEnabled(editing)
        self.eraser_btn.setEnabled(editing)
        self.brush_size_spin.setEnabled(editing)

        if editing:
            if not self.brush_btn.isChecked() and not self.eraser_btn.isChecked():
                self.brush_btn.setChecked(True)
                self.canvas.set_tool("draw")
            btn_text = "Сохранить изменения" if is_existing else "Сохранить в буфер"
            self.save_to_buffer_btn.setText(btn_text)
            self._status_mode.setText("Редактирование")
            self.info_label.setText(
                "Режим ручного редактирования маски:\n"
                "- ЛКМ: рисовать выбранным инструментом\n"
                "- Ctrl + ЛКМ drag: перемещение\n"
                f"Нажми '{btn_text}'.\n"
                "Или 'Очистить' для пересегментации.")
        elif is_existing:
            self.save_to_buffer_btn.setText("Фиксировать маску")
            self.pointer_btn.setChecked(True)
            self._status_mode.setText("Промпты")
            self.info_label.setText(
                f"Пересегментация маски #{self.masks_buffer.editing_index + 1}.\n"
                "ЛКМ клик = положительная точка (зелёная)\n"
                "ПКМ клик = отрицательная точка (красная)\n"
                "ЛКМ drag = box (жёлтый прямоугольник)\n"
                "Далее нажми 'Фиксировать маску'.")
        else:
            self.save_to_buffer_btn.setText("Фиксировать маску")
            self.pointer_btn.setChecked(True)
            self._status_mode.setText("Промпты")
            self.info_label.setText(
                "ЛКМ клик = положительная точка (зелёная)\n"
                "ПКМ клик = отрицательная точка (красная)\n"
                "ЛКМ drag = box (жёлтый прямоугольник)\n"
                "Ctrl + ЛКМ drag = перемещение\n"
                "Далее нажми 'Фиксировать маску'.")

    def _on_mask_edit_requested(self, mask: np.ndarray):
        """Двойной клик по маске в буфере → загрузить для редактирования."""
        if self.canvas.image_np is None:
            return
        self.canvas.clean()
        self.canvas.load_mask_for_edit(mask)
        self._update_edit_ui()

    def _on_tool_changed(self, btn_id: int):
        match btn_id:
            case 1: self.canvas.set_tool("draw")
            case 2: self.canvas.set_tool("erase")

    def _warn(self, title, text):
        QtWidgets.QMessageBox.warning(self, title, text)
