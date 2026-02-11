import numpy as np
from PIL import Image
from pathlib import Path

from PyQt6 import QtWidgets

from core.image_data import ImageWithMasks
from .masks_buffer import MasksBuffer
from .qt_canvas import QtSegmentationCanvas


class Window(QtWidgets.QMainWindow):
    def __init__(self, predictor, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Segmentation System")
        self.predictor = predictor

        # Данные: массив изображений с масками
        self.images_data: list[ImageWithMasks] = []
        self.current_image_idx = 0

        # UI
        self.canvas = QtSegmentationCanvas(predictor)
        self.masks_buffer = MasksBuffer()

        # Поле для имени изображения
        self.image_name_label = QtWidgets.QLabel("Имя изображения:")
        self.image_name = QtWidgets.QLabel()

        # Кнопки
        self.load_images_btn = QtWidgets.QPushButton("Загрузить изображения")
        self.clean_btn = QtWidgets.QPushButton("Очистить")
        # Кнопка работает в 2 шага:
        # 1) фиксирует маску предиктора и включает ручное редактирование
        # 2) фиксирует отредактированную маску и сохраняет в буфер
        self.save_to_buffer_btn = QtWidgets.QPushButton("Фиксировать маску")
        self.prev_image_btn = QtWidgets.QPushButton("Предыдущее")
        self.next_image_btn = QtWidgets.QPushButton("Следующее (0/0)")
        self.save_and_leave_btn = QtWidgets.QPushButton("Сохранить и выйти")

        # Инструменты редактирования маски
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

        self.info_label = QtWidgets.QLabel(
            "ЛКМ клик = положительная точка (зелёная)\n"
            "ПКМ клик = отрицательная точка (красная)\n"
            "ЛКМ drag = box (жёлтый прямоугольник)\n"
            "Ctrl + ЛКМ drag = перемещение\n"
            "Далее нажми 'Фиксировать маску' чтобы зафиксировать маску и включить ручное редактирование"
        )
        self.info_label.setWordWrap(True)

        # Привязка сигналов
        self.load_images_btn.clicked.connect(self.on_load_images)
        self.clean_btn.clicked.connect(self.on_clean)
        self.save_to_buffer_btn.clicked.connect(self.on_save_to_buffer)
        self.prev_image_btn.clicked.connect(self.on_previous_image)
        self.next_image_btn.clicked.connect(self.on_next_image)
        self.save_and_leave_btn.clicked.connect(self.on_save_and_leave)

        self.draw_radio.toggled.connect(self._on_tool_changed)
        self.erase_radio.toggled.connect(self._on_tool_changed)
        self.brush_size.valueChanged.connect(self._on_brush_size_changed)

        # Layout справа (кнопки)
        side_layout = QtWidgets.QVBoxLayout()
        side_layout.addWidget(self.image_name_label)
        side_layout.addWidget(self.image_name)
        side_layout.addSpacing(10)
        side_layout.addWidget(self.load_images_btn)
        side_layout.addWidget(self.clean_btn)
        side_layout.addWidget(self.save_to_buffer_btn)
        side_layout.addWidget(self.prev_image_btn)
        side_layout.addWidget(self.next_image_btn)
        side_layout.addSpacing(20)
        side_layout.addWidget(self.edit_group)
        side_layout.addSpacing(10)
        side_layout.addWidget(self.info_label)
        side_layout.addStretch(1)
        side_layout.addWidget(self.save_and_leave_btn)

        side_widget = QtWidgets.QWidget()
        side_widget.setLayout(side_layout)

        # Общий layout
        main_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(main_widget)
        main_layout.addWidget(self.masks_buffer, stretch=0)
        # QGraphicsView уже имеет скроллбары и быстрый зум
        main_layout.addWidget(self.canvas, stretch=1)
        main_layout.addWidget(side_widget, stretch=0)

        self.setCentralWidget(main_widget)
        self.resize(1920, 1080)
        self._update_edit_ui()

    def on_load_images(self):
        """Загружает несколько изображений"""
        fnames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Выбери изображения", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not fnames:
            return

        self.images_data = []
        for fname in fnames:
            pil_img = Image.open(fname).convert("RGB")
            img_array = np.array(pil_img)
            name = Path(fname).name
            self.images_data.append(ImageWithMasks(image=img_array, name=name))

        if self.images_data:
            self.current_image_idx = 0
            self._load_current_image()

    def on_clean(self):
        """Очищает canvas"""
        self.canvas.clean()
        self.canvas.redraw()
        self._update_edit_ui()

    def on_save_to_buffer(self):
        """
        Двухшаговый сценарий:
        - шаг 1: фиксируем маску предиктора -> ручное редактирование
        - шаг 2: фиксируем отредактированную маску -> сохраняем в буфер
        """
        # В edit-mode сохраняем отредактированную маску независимо от current_mask
        if self.canvas.edit_mode:
            final_mask = self.canvas.finish_edit()
            if final_mask is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Нет маски",
                    "Не найдена отредактированная маска.",
                )
                return
            self.masks_buffer.add(final_mask)
            self.on_clean()
            return

        if self.canvas.current_mask is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Нет маски",
                "Сначала поставь точки и/или box, чтобы получить маску предиктора.",
            )
            return

        if not self.canvas.edit_mode:
            ok = self.canvas.start_edit_from_current_mask()
            if not ok:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Не удалось",
                    "Не удалось зафиксировать маску для редактирования.",
                )
                return
            self._update_edit_ui()
            return

    def on_previous_image(self):
        """Переходит к предыдущему изображению"""
        if not self.images_data or self.current_image_idx == 0:
            return

        self.current_image_idx -= 1
        self._load_current_image()

    def on_next_image(self):
        """Переходит к следующему изображению"""
        if not self.images_data:
            return

        total = len(self.images_data)
        if self.current_image_idx >= total - 1:
            return

        self.current_image_idx += 1
        self._load_current_image()


    def on_save_and_leave(self):
        """Сохраняет все изображения (маски уже на диске), затем закрывает окно"""
        if not self.images_data:
            QtWidgets.QMessageBox.warning(
                self,
                "Нет данных",
                "Нет изображений для сохранения.",
            )
            return

        OUTPUT_DIR = "output"  # TODO: сделать выбор директории в настройках

        try:
            output_path = Path(OUTPUT_DIR)
            output_path.mkdir(parents=True, exist_ok=True)

            for idx, img_data in enumerate(self.images_data):
                # Генерируем имя файла
                if img_data.name:
                    base_name = Path(img_data.name).stem
                else:
                    base_name = f"image_{idx:04d}"

                # Сохраняем изображение (без изменения масштаба)
                img_path = output_path / f"{base_name}.png"
                img_pil = Image.fromarray(img_data.image)
                img_pil.save(img_path)

            # Маски уже сохранены на диск в output/masks/ при работе с ними
            QtWidgets.QMessageBox.information(
                self,
                "Успех",
                f"Все изображения и маски сохранены в {output_path}/\n"
            )
            self.close()
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Ошибка",
                f"Ошибка при сохранении: {str(e)}",
            )

    def _load_current_image(self):
        """Загружает текущее изображение на canvas"""
        if not self.images_data:
            return

        img_data = self.images_data[self.current_image_idx]
        self.canvas.set_image(img_data.image)

        # Загружаем маски с диска для этого изображения
        self.masks_buffer.set_image_name(img_data.name)

        self._update_navigation_info()
        self._update_edit_ui()

    def _update_navigation_info(self):
        """Обновляет информацию о навигации и состояние кнопок"""
        total = len(self.images_data)
        current = self.current_image_idx + 1 if self.images_data else 0
        self.next_image_btn.setText(f"Следующее ({current}/{total})")

        # Управление доступностью кнопок
        if not self.images_data:
            self.prev_image_btn.setEnabled(False)
            self.next_image_btn.setEnabled(False)
        else:
            # Кнопка "Назад" неактивна на первом изображении
            self.prev_image_btn.setEnabled(self.current_image_idx > 0)
            # Кнопка "Далее" неактивна на последнем изображении
            self.next_image_btn.setEnabled(self.current_image_idx < total - 1)

    def _update_edit_ui(self):
        """Обновляет подписи/доступность UI в зависимости от режима редактирования."""
        if self.canvas.edit_mode:
            self.save_to_buffer_btn.setText("Фиксировать и сохранить в буфер")
            self.edit_group.setEnabled(True)
            self.info_label.setText(
                "Режим ручного редактирования маски:\n"
                "- ЛКМ: рисовать выбранным инструментом\n"
                "- Ctrl + ЛКМ drag: перемещение\n"
                "Нажми 'Фиксировать и сохранить в буфер' чтобы зафиксировать и сохранить маску в буфер."
            )
        else:
            self.save_to_buffer_btn.setText("Фиксировать маску")
            self.edit_group.setEnabled(False)
            self.info_label.setText(
                "ЛКМ клик = положительная точка (зелёная)\n"
                "ПКМ клик = отрицательная точка (красная)\n"
                "ЛКМ drag = box (жёлтый прямоугольник)\n"
                "Ctrl + ЛКМ drag = перемещение\n"
                "Далее нажми 'Фиксировать маску' чтобы зафиксировать маску и включить ручное редактирование"
            )


    def _on_tool_changed(self):
        if self.draw_radio.isChecked():
            self.canvas.set_tool("draw")
        elif self.erase_radio.isChecked():
            self.canvas.set_tool("erase")

    def _on_brush_size_changed(self, v: int):
        self.canvas.set_brush_radius(v)