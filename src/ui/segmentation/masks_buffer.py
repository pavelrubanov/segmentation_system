import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
from PyQt6 import QtWidgets, QtCore, QtGui


class MasksBuffer(QtWidgets.QWidget):
    mask_edit_requested = QtCore.pyqtSignal(np.ndarray)  # маска для редактирования

    def __init__(self, parent=None):
        super().__init__(parent)

        self._temp_dir_obj = tempfile.TemporaryDirectory(prefix="segmasks_")
        self.output_dir = Path(self._temp_dir_obj.name)

        self.mask_paths: list[Path] = []
        self.current_image_name = ""
        self.editing_index: int | None = None  # какую маску редактируем (None = новая)

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
        self.list.itemDoubleClicked.connect(self._on_double_click)

        delete_action = QtGui.QAction(self)
        delete_action.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Delete))
        delete_action.triggered.connect(self.delete_selected)
        self.list.addAction(delete_action)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self.list)

        self.setMinimumWidth(160)

    def all_mask_paths(self) -> list[Path]:
        """Возвращает пути ко всем маскам всех изображений во временной папке."""
        return sorted(self.output_dir.glob("*_mask_*.png"))

    def set_image_name(self, image_name: str):
        """Устанавливает имя изображения и загружает соответствующие маски с диска."""
        self.current_image_name = image_name
        self.editing_index = None
        self._load_masks_from_disk()

    def save_mask(self, mask: np.ndarray):
        """Сохраняет маску: заменяет если editing_index задан, иначе добавляет новую."""
        if self.editing_index is not None:
            self._replace(self.editing_index, mask)
        else:
            self._add(mask)
        self.editing_index = None

    def delete_selected(self):
        """Удаляет выбранные маски из списка и с диска (чтобы glob не нашёл их при экспорте)."""
        for r in sorted([self.list.row(i) for i in self.list.selectedItems()], reverse=True):
            self.mask_paths.pop(r).unlink(missing_ok=True)
            self.list.takeItem(r)
        if self.editing_index is not None and self.editing_index >= len(self.mask_paths):
            self.editing_index = None

    def clear(self):
        """Сбрасывает UI-состояние буфера. Файлы во временной папке не трогает."""
        self.mask_paths.clear()
        self.list.clear()
        self.editing_index = None

    # ── Приватные ──────────────────────────────────────────────────────────────

    def _add(self, mask: np.ndarray):
        if not self.current_image_name:
            return
        base_name = Path(self.current_image_name).stem

        next_idx = 0
        for mask_path in self.output_dir.glob(f"{base_name}_mask_*.png"):
            idx_str = mask_path.stem.split("_mask_")[-1]
            if idx_str.isnumeric():
                next_idx = max(int(idx_str), next_idx)
        next_idx += 1

        mask_path = self.output_dir / f"{base_name}_mask_{next_idx:04d}.png"
        Image.fromarray(mask, mode="L").save(mask_path)
        self.mask_paths.append(mask_path)

        item = QtWidgets.QListWidgetItem()
        item.setIcon(self._mask_to_icon(mask))
        self.list.addItem(item)

    def _replace(self, index: int, mask: np.ndarray):
        if index < 0 or index >= len(self.mask_paths):
            return
        Image.fromarray(mask, mode="L").save(self.mask_paths[index])
        self.list.item(index).setIcon(self._mask_to_icon(mask))

    def _load_masks_from_disk(self):
        self.clear()
        if not self.current_image_name:
            return

        base_name = Path(self.current_image_name).stem
        for mask_path in sorted(self.output_dir.glob(f"{base_name}_mask_*.png")):
            self.mask_paths.append(mask_path)
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            item = QtWidgets.QListWidgetItem()
            item.setIcon(self._mask_to_icon(mask))
            self.list.addItem(item)
            del mask

    def _on_double_click(self, item: QtWidgets.QListWidgetItem):
        row = self.list.row(item)
        if row < 0 or row >= len(self.mask_paths):
            return
        self.editing_index = row
        mask = np.array(Image.open(self.mask_paths[row]).convert("L"), dtype=np.uint8)
        self.mask_edit_requested.emit(mask)

    def _menu(self, pos):
        if self.list.itemAt(pos) is None:
            return
        menu = QtWidgets.QMenu(self)
        act = menu.addAction("Delete")
        if menu.exec(self.list.mapToGlobal(pos)) == act:
            self.delete_selected()

    @staticmethod
    def _mask_to_icon(mask: np.ndarray, size=256):
        h, w = mask.shape
        fmt = QtGui.QImage.Format.Format_Grayscale8
        qimg = QtGui.QImage(mask.tobytes(), w, h, w, fmt)
        pm = QtGui.QPixmap.fromImage(qimg).scaled(
            size, size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation)
        return QtGui.QIcon(pm)
