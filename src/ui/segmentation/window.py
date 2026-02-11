import numpy as np
from PIL import Image
from pathlib import Path

from PyQt6 import QtWidgets

from core.image_data import ImageWithMasks
from .masks_buffer import MasksBuffer
from .canvas import SegmentationCanvas


class Window(QtWidgets.QMainWindow):
    def __init__(self, predictor, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Segmentation System")
        self.predictor = predictor

        self.images_data: list[ImageWithMasks] = []
        self.current_image_idx = 0

        # ── Виджеты ───────────────────────────────────────────────────────────
        self.canvas = SegmentationCanvas(predictor)
        self.masks_buffer = MasksBuffer()

        self.image_name_label = QtWidgets.QLabel("Имя изображения:")
        self.image_name = QtWidgets.QLabel()

        self.load_images_btn   = QtWidgets.QPushButton("Загрузить изображения")
        self.clean_btn         = QtWidgets.QPushButton("Очистить")
        self.save_to_buffer_btn = QtWidgets.QPushButton("Фиксировать маску")
        self.prev_image_btn    = QtWidgets.QPushButton("Предыдущее")
        self.next_image_btn    = QtWidgets.QPushButton("Следующее (0/0)")
        self.save_and_leave_btn = QtWidgets.QPushButton("Сохранить и выйти")

        # Инструменты кисти
        self.edit_group = QtWidgets.QGroupBox("Редактирование маски")
        self.draw_radio = QtWidgets.QRadioButton("Кисть (дорисовать)")
        self.erase_radio = QtWidgets.QRadioButton("Ластик (стереть)")
        self.draw_radio.setChecked(True)
        self.brush_size_label = QtWidgets.QLabel("Размер кисти:")
        self.brush_size = QtWidgets.QSpinBox()
        self.brush_size.setRange(1, 200)
        self.brush_size.setValue(15)

        edit_layout = QtWidgets.QVBoxLayout()
        edit_layout.addWidget(self.draw_radio)
        edit_layout.addWidget(self.erase_radio)
        edit_layout.addSpacing(8)
        edit_layout.addWidget(self.brush_size_label)
        edit_layout.addWidget(self.brush_size)
        self.edit_group.setLayout(edit_layout)
        self.edit_group.setEnabled(False)

        self.info_label = QtWidgets.QLabel()
        self.info_label.setWordWrap(True)

        # ── Сигналы ───────────────────────────────────────────────────────────
        self.load_images_btn.clicked.connect(self.on_load_images)
        self.clean_btn.clicked.connect(self.on_clean)
        self.save_to_buffer_btn.clicked.connect(self.on_save_to_buffer)
        self.prev_image_btn.clicked.connect(self.on_previous_image)
        self.next_image_btn.clicked.connect(self.on_next_image)
        self.save_and_leave_btn.clicked.connect(self.on_save_and_leave)
        self.draw_radio.toggled.connect(self._on_tool_changed)
        self.erase_radio.toggled.connect(self._on_tool_changed)
        self.brush_size.valueChanged.connect(
            lambda v: self.canvas.set_brush_radius(v))

        # ── Layout ────────────────────────────────────────────────────────────
        side = QtWidgets.QVBoxLayout()
        side.addWidget(self.image_name_label)
        side.addWidget(self.image_name)
        side.addSpacing(10)
        for btn in (self.load_images_btn, self.clean_btn,
                     self.save_to_buffer_btn, self.prev_image_btn,
                     self.next_image_btn):
            side.addWidget(btn)
        side.addSpacing(20)
        side.addWidget(self.edit_group)
        side.addSpacing(10)
        side.addWidget(self.info_label)
        side.addStretch(1)
        side.addWidget(self.save_and_leave_btn)

        side_widget = QtWidgets.QWidget()
        side_widget.setLayout(side)

        main_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(main_widget)
        main_layout.addWidget(self.masks_buffer, stretch=0)
        main_layout.addWidget(self.canvas, stretch=1)
        main_layout.addWidget(side_widget, stretch=0)

        self.setCentralWidget(main_widget)
        self.resize(1920, 1080)
        self._update_edit_ui()

    # ── Действия ──────────────────────────────────────────────────────────────

    def on_load_images(self):
        fnames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Выбери изображения", "",
            "Images (*.png *.jpg *.jpeg *.bmp)")
        if not fnames:
            return
        self.images_data = [
            ImageWithMasks(image=np.array(Image.open(f).convert("RGB")),
                           name=Path(f).name)
            for f in fnames
        ]
        self.current_image_idx = 0
        self._load_current_image()

    def on_clean(self):
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
            self.masks_buffer.add(mask)
            self.on_clean()
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
        if not self.images_data:
            self._warn("Нет данных", "Нет изображений для сохранения.")
            return
        try:
            out = Path("output")
            out.mkdir(parents=True, exist_ok=True)
            for idx, d in enumerate(self.images_data):
                name = Path(d.name).stem if d.name else f"image_{idx:04d}"
                Image.fromarray(d.image).save(out / f"{name}.png")
            QtWidgets.QMessageBox.information(
                self, "Успех",
                f"Все изображения и маски сохранены в {out}/")
            self.close()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Ошибка", f"Ошибка при сохранении: {exc}")

    # ── Внутренние ────────────────────────────────────────────────────────────

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
        self.next_image_btn.setText(f"Следующее ({cur}/{total})")
        self.prev_image_btn.setEnabled(self.current_image_idx > 0)
        self.next_image_btn.setEnabled(self.current_image_idx < total - 1)

    def _update_edit_ui(self):
        editing = self.canvas.edit_mode
        self.edit_group.setEnabled(editing)
        if editing:
            self.save_to_buffer_btn.setText("Фиксировать и сохранить в буфер")
            self.info_label.setText(
                "Режим ручного редактирования маски:\n"
                "- ЛКМ: рисовать выбранным инструментом\n"
                "- Ctrl + ЛКМ drag: перемещение\n"
                "Нажми 'Фиксировать и сохранить в буфер'.")
        else:
            self.save_to_buffer_btn.setText("Фиксировать маску")
            self.info_label.setText(
                "ЛКМ клик = положительная точка (зелёная)\n"
                "ПКМ клик = отрицательная точка (красная)\n"
                "ЛКМ drag = box (жёлтый прямоугольник)\n"
                "Ctrl + ЛКМ drag = перемещение\n"
                "Далее нажми 'Фиксировать маску'.")

    def _on_tool_changed(self):
        if self.draw_radio.isChecked():
            self.canvas.set_tool("draw")
        elif self.erase_radio.isChecked():
            self.canvas.set_tool("erase")

    def _warn(self, title, text):
        QtWidgets.QMessageBox.warning(self, title, text)
