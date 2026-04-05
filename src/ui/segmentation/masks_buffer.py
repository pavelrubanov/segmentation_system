import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
from PyQt6 import QtWidgets, QtCore, QtGui


_ICON_SIZE = 128
_MASK_COLOR = QtGui.QColor(0, 255, 255, 160)  # cyan для иконок
_CHECK_LIGHT = QtGui.QColor(220, 220, 220)
_CHECK_DARK = QtGui.QColor(200, 200, 200)
_CHECK_CELL = 8


class MasksBuffer(QtWidgets.QWidget):
    mask_edit_requested = QtCore.pyqtSignal(np.ndarray)  # маска для редактирования

    def __init__(self, parent=None):
        super().__init__(parent)

        self._temp_dir_obj = tempfile.TemporaryDirectory(prefix="segmasks_")
        self.output_dir = Path(self._temp_dir_obj.name)

        self.mask_paths: list[Path] = []
        self.current_image_name = ""
        self.editing_index: int | None = None

        title = QtWidgets.QLabel("Маски")
        title.setObjectName("subheading")

        self.count_label = QtWidgets.QLabel("0 масок")
        self.count_label.setObjectName("secondary")

        self.list = QtWidgets.QListWidget()
        self.list.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.list.setIconSize(QtCore.QSize(_ICON_SIZE, _ICON_SIZE))
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
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        layout.addWidget(title)
        layout.addWidget(self.count_label)
        layout.addWidget(self.list)

        self.setMinimumWidth(200)

    def all_mask_paths(self) -> list[Path]:
        return sorted(self.output_dir.glob("*_mask_*.png"))

    def set_image_name(self, image_name: str):
        self.current_image_name = image_name
        self.editing_index = None
        self._load_masks_from_disk()

    def save_mask(self, mask: np.ndarray):
        if self.editing_index is not None:
            self._replace(self.editing_index, mask)
        else:
            self._add(mask)
        self.editing_index = None

    def delete_selected(self):
        for r in sorted([self.list.row(i) for i in self.list.selectedItems()], reverse=True):
            self.mask_paths.pop(r).unlink(missing_ok=True)
            self.list.takeItem(r)
        if self.editing_index is not None and self.editing_index >= len(self.mask_paths):
            self.editing_index = None
        self._update_count()

    def clear(self):
        self.mask_paths.clear()
        self.list.clear()
        self.editing_index = None
        self._update_count()

    # приватные методы

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
        self._update_count()

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
        self._update_count()

    def _update_count(self):
        n = len(self.mask_paths)
        if n % 10 == 1 and n % 100 != 11:
            word = "маска"
        elif 2 <= n % 10 <= 4 and not (12 <= n % 100 <= 14):
            word = "маски"
        else:
            word = "масок"
        self.count_label.setText(f"{n} {word}")

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
        act = menu.addAction("Удалить")
        if menu.exec(self.list.mapToGlobal(pos)) == act:
            self.delete_selected()

    @staticmethod
    def _mask_to_icon(mask: np.ndarray, size: int = _ICON_SIZE) -> QtGui.QIcon:
        """Маска cyan на клетчатом фоне (checkerboard)."""
        h, w = mask.shape

        # маска → grayscale QImage → QPixmap нужного размера
        fmt = QtGui.QImage.Format.Format_Grayscale8
        qimg = QtGui.QImage(mask.tobytes(), w, h, w, fmt)
        mask_pm = QtGui.QPixmap.fromImage(qimg).scaled(
            size, size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation)

        # рисуем шахматный фон, поверх кладём маску цветом
        result = QtGui.QPixmap(mask_pm.size())
        p = QtGui.QPainter(result)
        # шахматный фон
        sw, sh = mask_pm.width(), mask_pm.height()
        for row in range(0, sh, _CHECK_CELL):
            for col in range(0, sw, _CHECK_CELL):
                color = _CHECK_LIGHT if (row // _CHECK_CELL + col // _CHECK_CELL) % 2 == 0 else _CHECK_DARK
                p.fillRect(col, row, _CHECK_CELL, _CHECK_CELL, color)
        # cyan overlay по маске
        colored = QtGui.QPixmap(mask_pm.size())
        colored.fill(_MASK_COLOR)
        colored.setMask(mask_pm.createMaskFromColor(
            QtGui.QColor(0, 0, 0), QtCore.Qt.MaskMode.MaskInColor))
        p.drawPixmap(0, 0, colored)
        p.end()

        return QtGui.QIcon(result)
