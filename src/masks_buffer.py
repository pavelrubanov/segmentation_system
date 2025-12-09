import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

class MasksBuffer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.masks = []  # здесь храним маски целиком (np.ndarray)

        title = QtWidgets.QLabel("Masks\nbuffer")
        title.setAlignment(QtCore.Qt.AlignCenter)

        self.list = QtWidgets.QListWidget()
        self.list.setViewMode(QtWidgets.QListView.IconMode)
        self.list.setIconSize(QtCore.QSize(256, 256))
        self.list.setResizeMode(QtWidgets.QListView.Adjust)
        self.list.setMovement(QtWidgets.QListView.Static)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)

        delete_action = QtWidgets.QAction(self)
        delete_action.setShortcut(QtGui.QKeySequence.Delete)
        delete_action.triggered.connect(self.delete_selected)
        self.list.addAction(delete_action)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self.list)

        self.setMinimumWidth(160)

    def add(self, mask):
        mask = np.array(mask, copy=True)
        self.masks.append(mask)

        item = QtWidgets.QListWidgetItem()
        item.setIcon(self._mask_to_icon(mask))
        self.list.addItem(item)

    def delete_selected(self):
        rows = sorted([self.list.row(i) for i in self.list.selectedItems()], reverse=True)
        for r in rows:
            self.list.takeItem(r)
            self.masks.pop(r)

    def _menu(self, pos):
        if self.list.itemAt(pos) is None:
            return
        menu = QtWidgets.QMenu(self)
        act = menu.addAction("Delete")
        if menu.exec_(self.list.mapToGlobal(pos)) == act:
            self.delete_selected()

    @staticmethod
    def _mask_to_icon(mask, size = 256):
        m = mask
        if m.dtype != np.uint8:
            m = (m.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)

        h, w = m.shape
        m = np.ascontiguousarray(m)
        qimg = QtGui.QImage(m.data, w, h, w, QtGui.QImage.Format_Grayscale8).copy()
        pm = QtGui.QPixmap.fromImage(qimg).scaled(
            size, size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        return QtGui.QIcon(pm)

