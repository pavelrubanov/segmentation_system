import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
import cv2
import numpy as np
from pathlib import Path
from typing import NamedTuple
from PIL import Image
from PyQt6 import QtWidgets, QtCore, QtGui
from core.file_naming import crop_filename, mask_filename
from core.io import format_count, read_rgb
from core.leaf_pipeline import transform_leaf


_ICON_SIZE = 128
_THUMB_LONG_SIDE = 384  # ~3× размера иконки --- источник для масок-иконок


class MaskEntry(NamedTuple):
    mask: Path   # полноразмерная RGB-картинка (фон чёрный за пределами листа), для редактирования
    crop: Path   # препроцессированный RGB-кроп (для классификатора и измерений)


class MasksBuffer(QtWidgets.QWidget):
    mask_edit_requested = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._temp_dir_obj = tempfile.TemporaryDirectory(prefix="segmasks_")
        self.output_dir = Path(self._temp_dir_obj.name)

        # Фоновая запись PNG: иконка ставится сразу из numpy, диск пишется в
        # отдельных потоках. `_pending` хранит in-flight future по path ---
        # при double-click / delete / save_and_leave ждём завершения.
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._pending: dict[Path, Future] = {}

        self.entries: list[MaskEntry] = []
        self.current_image_name = ""
        self.image_np: np.ndarray | None = None
        self.image_thumb: np.ndarray | None = None  # downscaled image для иконок
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

    def set_image(self, image_name: str, image_np: np.ndarray | None = None):
        self.current_image_name = image_name
        self.image_np = image_np
        # Кэш для иконок: масштабируем longest side к _THUMB_LONG_SIDE,
        # сохраняя пропорции. На 4K-кадре 16:9 это ~250 KB и размер 384×216.
        if image_np is not None:
            h, w = image_np.shape[:2]
            scale = _THUMB_LONG_SIDE / max(h, w)
            self.image_thumb = cv2.resize(
                image_np, (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            self.image_thumb = None
        self.editing_index = None

    def restore_entries(self, entries: list[MaskEntry]):
        """Восстановить буфер из сохранённого списка (читаем превью с диска)."""
        self.list.clear()
        self.entries = list(entries)
        self.editing_index = None
        for e in self.entries:
            item = QtWidgets.QListWidgetItem()
            item.setIcon(self._icon_from_path(e.mask))
            self.list.addItem(item)
        self._update_count()

    def wait_for_pending(self) -> None:
        """Дождаться завершения всех фоновых записей PNG."""
        for f in list(self._pending.values()):
            if not f.done():
                f.result()
        self._pending.clear()

    def save_mask(self, mask: np.ndarray, crop: np.ndarray | None = None):
        """Сохранить маску. Если crop передан --- использовать его, иначе пересчитать."""
        if self.editing_index is not None:
            self._replace(self.editing_index, mask, crop)
        else:
            self._add(mask, crop)
        self.editing_index = None

    def delete_selected(self):
        for r in sorted([self.list.row(i) for i in self.list.selectedItems()], reverse=True):
            e = self.entries.pop(r)
            for p in e:
                self._wait_for(p)
                p.unlink(missing_ok=True)
            self.list.takeItem(r)
        if self.editing_index is not None and self.editing_index >= len(self.entries):
            self.editing_index = None
        self._update_count()

    def clear(self):
        self.entries.clear()
        self.list.clear()
        self.editing_index = None
        self._update_count()

    # приватные методы

    def _make_entry(self, base_name: str, idx: int) -> MaskEntry:
        d = self.output_dir
        return MaskEntry(
            mask=d / mask_filename(base_name, idx),
            crop=d / crop_filename(base_name, idx),
        )

    def _write_files(self, e: MaskEntry, mask: np.ndarray,
                     crop: np.ndarray | None = None) -> np.ndarray:
        """Готовим full-size masked RGB и crop, пишем PNG в фон.

        Возвращает icon-source --- masked thumb-кадр (~384×216), на котором
        строится иконка без касания 4K-данных.
        """
        masked = cv2.bitwise_and(self.image_np, self.image_np, mask=mask)
        if crop is None:
            crop = transform_leaf(masked)
        self._submit(e.mask, masked)
        self._submit(e.crop, crop)

        # Иконка: маска downscaled к thumb, накладывается на image_thumb.
        h, w = self.image_thumb.shape[:2]
        mask_thumb = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return cv2.bitwise_and(self.image_thumb, self.image_thumb, mask=mask_thumb)

    def _submit(self, path: Path, arr: np.ndarray) -> None:
        """Кладём задачу записи PNG в фон. Если уже идёт запись в этот path
        (replace одной и той же маски) --- ждём предыдущую перед новой."""
        old = self._pending.pop(path, None)
        if old is not None and not old.done():
            old.result()
        self._pending[path] = self._executor.submit(_write_png, path, arr)

    def _wait_for(self, path: Path) -> None:
        """Ждём конкретную фоновую запись (для double-click, delete)."""
        f = self._pending.pop(path, None)
        if f is not None and not f.done():
            f.result()

    def _add(self, mask: np.ndarray, crop: np.ndarray | None = None):
        if not self.current_image_name:
            return
        base_name = Path(self.current_image_name).stem
        next_idx = (max((int(e.mask.stem.rsplit("_", 1)[1]) for e in self.entries), default=0) + 1)
        e = self._make_entry(base_name, next_idx)
        icon_src = self._write_files(e, mask, crop)
        self.entries.append(e)

        item = QtWidgets.QListWidgetItem()
        item.setIcon(self._icon_from_array(icon_src))
        self.list.addItem(item)
        self._update_count()

    def _replace(self, index: int, mask: np.ndarray, crop: np.ndarray | None = None):
        e = self.entries[index]
        icon_src = self._write_files(e, mask, crop)
        self.list.item(index).setIcon(self._icon_from_array(icon_src))

    def _update_count(self):
        self.count_label.setText(format_count(
            len(self.entries), ("маска", "маски", "масок")))

    def _on_double_click(self, item: QtWidgets.QListWidgetItem):
        row = self.list.row(item)
        if row < 0 or row >= len(self.entries):
            return
        self.editing_index = row
        path = self.entries[row].mask
        # Если запись ещё в полёте --- ждём.
        self._wait_for(path)
        rgb = read_rgb(path)
        mask = (rgb.sum(axis=2) > 0).astype(np.uint8) * 255
        self.mask_edit_requested.emit(mask)

    def _menu(self, pos):
        if self.list.itemAt(pos) is None:
            return
        menu = QtWidgets.QMenu(self)
        act = menu.addAction("Удалить")
        if menu.exec(self.list.mapToGlobal(pos)) == act:
            self.delete_selected()

    @staticmethod
    def _icon_from_path(path: Path, size: int = _ICON_SIZE) -> QtGui.QIcon:
        """Иконка = downscale RGB-картинки с диска (фон уже чёрный)."""
        pm = QtGui.QPixmap(str(path))
        if pm.isNull():
            return QtGui.QIcon()
        return QtGui.QIcon(pm.scaled(
            size, size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        ))

    @staticmethod
    def _icon_from_array(arr: np.ndarray, size: int = _ICON_SIZE) -> QtGui.QIcon:
        """Иконка прямо из numpy RGB --- без чтения с диска."""
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        h, w = arr.shape[:2]
        qimg = QtGui.QImage(
            arr.data, w, h, arr.strides[0],
            QtGui.QImage.Format.Format_RGB888,
        ).copy()  # detach от numpy, иначе пиксели уплывут вместе с массивом
        pm = QtGui.QPixmap.fromImage(qimg)
        return QtGui.QIcon(pm.scaled(
            size, size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        ))


def _write_png(path: Path, arr: np.ndarray) -> None:
    """Воркер: PNG-запись. Top-level чтобы не таскать `self` по тредам."""
    Image.fromarray(arr).save(path)
