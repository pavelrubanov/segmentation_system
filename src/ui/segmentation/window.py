import numpy as np
from PIL import Image
from pathlib import Path

from PyQt6 import QtWidgets
import matplotlib
matplotlib.use("QtAgg")

from core.image_data import ImageWithMasks
from .masks_buffer import MasksBuffer
from .canvas import SegmentationCanvas


class Window(QtWidgets.QMainWindow):
    def __init__(self, predictor, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Segmentation System")
        self.predictor = predictor

        # Данные: массив изображений с масками
        self.images_data: list[ImageWithMasks] = []
        self.current_image_idx = 0

        # UI
        self.canvas = SegmentationCanvas(predictor)
        self.masks_buffer = MasksBuffer()

        # Поле для имени изображения
        self.image_name_label = QtWidgets.QLabel("Image name:")
        self.image_name_input = QtWidgets.QLineEdit()
        self.image_name_input.setPlaceholderText("Введите имя изображения")

        # Кнопки
        self.load_images_btn = QtWidgets.QPushButton("Load Images")
        self.clean_btn = QtWidgets.QPushButton("Clean")
        self.save_to_buffer_btn = QtWidgets.QPushButton("Save to buffer")
        self.prev_image_btn = QtWidgets.QPushButton("Prev image")
        self.next_image_btn = QtWidgets.QPushButton("Next image (0/0)")
        self.save_and_leave_btn = QtWidgets.QPushButton("Save and leave")

        self.info_label = QtWidgets.QLabel(
            "ЛКМ клик = положительная точка (зелёная)\n"
            "ПКМ клик = отрицательная точка (красная)\n"
            "ЛКМ drag = box (жёлтый прямоугольник)"
        )
        self.info_label.setWordWrap(True)

        # Привязка сигналов
        self.load_images_btn.clicked.connect(self.on_load_images)
        self.clean_btn.clicked.connect(self.on_clean)
        self.save_to_buffer_btn.clicked.connect(self.on_save_to_buffer)
        self.prev_image_btn.clicked.connect(self.on_previous_image)
        self.next_image_btn.clicked.connect(self.on_next_image)
        self.save_and_leave_btn.clicked.connect(self.on_save_and_leave)

        # Layout справа (кнопки)
        side_layout = QtWidgets.QVBoxLayout()
        side_layout.addWidget(self.image_name_label)
        side_layout.addWidget(self.image_name_input)
        side_layout.addSpacing(10)
        side_layout.addWidget(self.load_images_btn)
        side_layout.addWidget(self.clean_btn)
        side_layout.addWidget(self.save_to_buffer_btn)
        side_layout.addWidget(self.prev_image_btn)
        side_layout.addWidget(self.next_image_btn)
        side_layout.addSpacing(20)
        side_layout.addWidget(self.info_label)
        side_layout.addStretch(1)
        side_layout.addWidget(self.save_and_leave_btn)

        side_widget = QtWidgets.QWidget()
        side_widget.setLayout(side_layout)

        # Общий layout
        main_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(main_widget)
        main_layout.addWidget(self.masks_buffer, stretch=0)
        main_layout.addWidget(self.canvas, stretch=1)
        main_layout.addWidget(side_widget, stretch=0)

        self.setCentralWidget(main_widget)
        self.resize(1920, 1080)

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

    def _load_current_image(self):
        """Загружает текущее изображение на canvas"""
        if not self.images_data:
            return

        img_data = self.images_data[self.current_image_idx]
        self.canvas.set_image(img_data.image)
        self.image_name_input.setText(img_data.name)

        # Загружаем маски в буфер
        self.masks_buffer.masks = [mask.copy() for mask in img_data.masks]
        self.masks_buffer.list.clear()
        for mask in img_data.masks:
            item = QtWidgets.QListWidgetItem()
            item.setIcon(self.masks_buffer._mask_to_icon(mask))
            self.masks_buffer.list.addItem(item)

        self._update_navigation_info()

    def _update_navigation_info(self):
        """Обновляет информацию о навигации и состояние кнопок"""
        total = len(self.images_data)
        current = self.current_image_idx + 1 if self.images_data else 0
        self.next_image_btn.setText(f"Next image ({current}/{total})")
        
        # Управление доступностью кнопок
        if not self.images_data:
            self.prev_image_btn.setEnabled(False)
            self.next_image_btn.setEnabled(False)
        else:
            # Кнопка "Назад" неактивна на первом изображении
            self.prev_image_btn.setEnabled(self.current_image_idx > 0)
            # Кнопка "Далее" неактивна на последнем изображении
            self.next_image_btn.setEnabled(self.current_image_idx < total - 1)

    def on_clean(self):
        """Очищает canvas"""
        self.canvas.clean()
        self.canvas.redraw()

    def on_save_to_buffer(self):
        """Сохраняет текущую маску в буфер"""
        if self.canvas.current_mask is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Нет маски",
                "Сначала добавь точки и/или box, чтобы получить маску.",
            )
            return

        # Копируем маску, чтобы сохранить исходный размер
        mask_copy = np.array(self.canvas.current_mask, copy=True)
        self.masks_buffer.add(mask_copy)
        self.on_clean()

    def on_previous_image(self):
        """Переходит к предыдущему изображению"""
        if not self.images_data or self.current_image_idx == 0:
            return

        self._save_current_masks()

        self.current_image_idx -= 1
        self._load_current_image()

    def on_next_image(self):
        """Переходит к следующему изображению"""
        if not self.images_data:
            return

        total = len(self.images_data)
        if self.current_image_idx >= total - 1:
            return

        self._save_current_masks()

        self.current_image_idx += 1
        self._load_current_image()

    def _save_current_masks(self):
        """Сохраняет текущие маски из буфера в данные изображения"""
        if not self.images_data:
            return

        img_data = self.images_data[self.current_image_idx]
        img_data.masks = [mask.copy() for mask in self.masks_buffer.masks]
        img_data.name = self.image_name_input.text() or img_data.name

    def on_save_and_leave(self):
        """Сохраняет все изображения и маски, затем закрывает окно"""
        if not self.images_data:
            QtWidgets.QMessageBox.warning(
                self,
                "Нет данных",
                "Нет изображений для сохранения.",
            )
            return

        self._save_current_masks()

        OUTPUT_DIR = "output"  # TODO: сделать выбор директории в найстройках

        try:
            output_path = Path(OUTPUT_DIR)
            output_path.mkdir(parents=True, exist_ok=True)

            images_dir = output_path / "images"
            masks_dir = output_path / "masks"
            images_dir.mkdir(exist_ok=True)
            masks_dir.mkdir(exist_ok=True)

            for idx, img_data in enumerate(self.images_data):
                # Генерируем имя файла
                if img_data.name:
                    base_name = Path(img_data.name).stem
                else:
                    base_name = f"image_{idx:04d}"

                # Сохраняем изображение (без изменения масштаба)
                img_path = images_dir / f"{base_name}.png"
                img_pil = Image.fromarray(img_data.image)
                img_pil.save(img_path)

                # Сохраняем маски (без изменения масштаба)
                for mask_idx, mask in enumerate(img_data.masks):
                    mask_path = masks_dir / f"{base_name}_mask_{mask_idx:04d}.png"
                    # Конвертируем маску в uint8 [0, 255]
                    if mask.dtype != np.uint8:
                        mask_uint8 = (mask.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
                    else:
                        mask_uint8 = mask.copy()
                    mask_pil = Image.fromarray(mask_uint8, mode="L")
                    mask_pil.save(mask_path)

            QtWidgets.QMessageBox.information(
                self,
                "Успех",
                f"Все изображения и маски сохранены в директорию output/",
            )
            self.close()
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Ошибка",
                f"Ошибка при сохранении: {str(e)}",
            )
