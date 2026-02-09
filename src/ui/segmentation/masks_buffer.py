import numpy as np
from pathlib import Path
from PIL import Image
from PyQt6 import QtWidgets, QtCore, QtGui


class MasksBuffer(QtWidgets.QWidget):
    def __init__(self, parent=None, output_dir: str = "output"):
        super().__init__(parent)
        
        self.output_dir = Path(output_dir)

        # Храним пути к файлам масок, а не сами массивы
        self.mask_paths: list[Path] = []
        self.current_image_name = ""  # имя текущего изображения для поиска масок

        title = QtWidgets.QLabel("Masks\nbuffer")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.list = QtWidgets.QListWidget()
        self.list.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.list.setIconSize(QtCore.QSize(256, 256))
        self.list.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.list.setMovement(QtWidgets.QListView.Movement.Static)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)

        self.list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)

        delete_action = QtGui.QAction(self)
        delete_action.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Delete))
        delete_action.triggered.connect(self.delete_selected)
        self.list.addAction(delete_action)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self.list)

        self.setMinimumWidth(160)

    def set_image_name(self, image_name: str):
        """Устанавливает имя изображения и загружает соответствующие маски с диска"""
        self.current_image_name = image_name
        self._load_masks_from_disk()


    def add(self, mask: np.ndarray):
        """Добавляет маску: сохраняет на диск и добавляет в список"""
        if not self.current_image_name:
            return

        base_name = Path(self.current_image_name).stem
        
        # Находим следующий индекс маски (проверяем и на диске, и в текущих путях)
        next_idx = 0
        
        # Проверяем существующие файлы на диске
        pattern = f"{base_name}_mask_*.png"
        for mask_path in self.output_dir.glob(pattern):
            idx_str = mask_path.stem.split("_mask_")[-1]
            if idx_str.isnumeric():
                next_idx = max(int(idx_str), next_idx)

        next_idx += 1

        # Сохраняем маску на диск
        mask_path = self.output_dir / f"{base_name}_mask_{next_idx:04d}.png"
        mask_pil = Image.fromarray(mask, mode="L")
        mask_pil.save(mask_path)

        # Добавляем путь в список
        self.mask_paths.append(mask_path)
        
        # Создаем иконку (маска уже на диске, можно использовать её)
        item = QtWidgets.QListWidgetItem()
        item.setIcon(self._mask_to_icon(mask))
        self.list.addItem(item)

    def delete_selected(self):
        """Удаляет выбранные маски из списка и с диска"""
        rows = sorted([self.list.row(i) for i in self.list.selectedItems()], reverse=True)
        for r in rows:
            # Удаляем файл с диска
            if r < len(self.mask_paths):
                mask_path = self.mask_paths[r]
                if mask_path.exists():
                    mask_path.unlink()
            
            # Удаляем из списков
            self.list.takeItem(r)
            if r < len(self.mask_paths):
                self.mask_paths.pop(r)

    def clear(self, delete_files: bool = True):
        """Очищает буфер масок. Если delete_files=True, удаляет файлы с диска"""
        if delete_files:
            for mask_path in self.mask_paths:
                if mask_path.exists():
                    mask_path.unlink()
        
        self.mask_paths.clear()
        self.list.clear()

    def _load_masks_from_disk(self):
        """Загружает маски с диска для текущего изображения"""
        self.clear(delete_files=False)  # Очищаем текущий список, но не удаляем файлы

        if not self.current_image_name:
            return

        base_name = Path(self.current_image_name).stem

        # Ищем все маски для этого изображения
        pattern = f"{base_name}_mask_*.png"
        mask_files = sorted(self.output_dir.glob(pattern))

        for mask_path in mask_files:
            self.mask_paths.append(mask_path)
            # Загружаем маску только для создания иконки, затем освобождаем память
            mask_pil = Image.open(mask_path).convert("L")
            mask = np.array(mask_pil, dtype=np.uint8)

            item = QtWidgets.QListWidgetItem()
            item.setIcon(self._mask_to_icon(mask))
            self.list.addItem(item)
            del mask  # Освобождаем память сразу после создания иконки

    def _menu(self, pos):
        if self.list.itemAt(pos) is None:
            return
        menu = QtWidgets.QMenu(self)
        act = menu.addAction("Delete")
        if menu.exec(self.list.mapToGlobal(pos)) == act:
            self.delete_selected()

    @staticmethod
    def _mask_to_icon(mask: np.ndarray, size=256):
        """Создает иконку из маски (загружает в память временно)"""
        h, w = mask.shape

        fmt = QtGui.QImage.Format.Format_Grayscale8
        qimg = QtGui.QImage(mask.tobytes(), w, h, w, fmt)

        pm = QtGui.QPixmap.fromImage(qimg).scaled(
            size,
            size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        return QtGui.QIcon(pm)


