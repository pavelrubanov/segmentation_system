"""
Канвас сегментации на QGraphicsView.

Режим prompts:  ЛКМ клик — positive точка, ПКМ клик — negative точка,
                ЛКМ drag — box.
Режим edit:     ЛКМ — кисть/ластик.
Ctrl + ЛКМ drag — перемещение (встроенный Qt ScrollHandDrag, в любом режиме).
Зум колёсиком (не меньше fit-to-view).
"""

import numpy as np
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from core.predictor import Predictor

_MASK_COLOR_BGRA = (255, 255, 0, 128)
_NO_BRUSH = QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush)
_LMB = QtCore.Qt.MouseButton.LeftButton
_RMB = QtCore.Qt.MouseButton.RightButton
_CTRL = QtCore.Qt.KeyboardModifier.ControlModifier
_SCROLL_HAND = QtWidgets.QGraphicsView.DragMode.ScrollHandDrag
_NO_DRAG = QtWidgets.QGraphicsView.DragMode.NoDrag


# ─── Overlay маски ────────────────────────────────────────────────────────────

class _MaskOverlay(QtWidgets.QGraphicsItem):
    """Рисует ARGB-изображение маски поверх фото."""

    def __init__(self, w: int, h: int):
        super().__init__()
        self._w, self._h = w, h
        self.image = self._make_image(w, h)

    def boundingRect(self):
        return QtCore.QRectF(0, 0, self._w, self._h)

    def paint(self, painter, _option, _widget=None):
        painter.drawImage(0, 0, self.image)

    def resize(self, w: int, h: int):
        self.prepareGeometryChange()
        self._w, self._h = w, h
        self.image = self._make_image(w, h)
        self.update()

    @staticmethod
    def _make_image(w, h):
        img = QtGui.QImage(w, h, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(QtCore.Qt.GlobalColor.transparent)
        return img

    def numpy_view(self):
        ptr = self.image.bits()
        ptr.setsize(self.image.sizeInBytes())
        return np.frombuffer(ptr, np.uint8).reshape(self._h, self._w, 4)


# ─── Основной канвас ─────────────────────────────────────────────────────────

class QtSegmentationCanvas(QtWidgets.QGraphicsView):

    def __init__(self, predictor: Predictor, parent=None):
        super().__init__(parent)
        self.predictor = predictor

        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)

        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing |
                            QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(_NO_DRAG)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        # Данные
        self.image_np: Optional[np.ndarray] = None
        self.pos_points: list = []
        self.neg_points: list = []
        self.box: Optional[tuple] = None
        self.current_mask: Optional[np.ndarray] = None
        self.edit_mode = False

        # Edit-mode
        self._tool = "draw"
        self._brush_radius = 15
        self._painting = False
        self._last_xy: Optional[tuple] = None
        self._mask_data: Optional[np.ndarray] = None
        self._disk_cache: dict[int, np.ndarray] = {}

        # Графические элементы
        self._img_item: Optional[QtWidgets.QGraphicsPixmapItem] = None
        self._overlay: Optional[_MaskOverlay] = None
        self._box_item: Optional[QtWidgets.QGraphicsRectItem] = None
        self._drag_rect: Optional[QtWidgets.QGraphicsRectItem] = None
        self._brush_circle: Optional[QtWidgets.QGraphicsEllipseItem] = None
        self._point_items: list = []

        # Состояние мыши
        self._press_pos: Optional[QtCore.QPoint] = None
        self._press_scene: Optional[QtCore.QPointF] = None
        self._dragging = False   # ЛКМ drag → box
        self._panning = False    # Qt ScrollHandDrag активен

    # ── Public API ────────────────────────────────────────────────────────────

    def set_tool(self, tool: str):
        if tool in ("draw", "erase"):
            self._tool = tool

    def set_brush_radius(self, r: int):
        self._brush_radius = max(1, min(200, int(r)))
        if self._brush_circle and not self._brush_circle.rect().isNull():
            c = self._brush_circle.rect().center()
            br = self._brush_radius
            self._brush_circle.setRect(c.x() - br, c.y() - br, br * 2, br * 2)

    def set_image(self, image_np: np.ndarray):
        self.clean()
        self.image_np = image_np
        self.predictor.set_image(image_np)
        h, w = image_np.shape[:2]

        qimg = QtGui.QImage(image_np.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()
        pm = QtGui.QPixmap.fromImage(qimg)
        if self._img_item is None:
            self._img_item = self._scene.addPixmap(pm)
            self._img_item.setZValue(0)
        else:
            self._img_item.setPixmap(pm)

        if self._overlay is None:
            self._overlay = _MaskOverlay(w, h)
            self._overlay.setZValue(10)
            self._scene.addItem(self._overlay)
        else:
            self._overlay.resize(w, h)

        if self._brush_circle is None:
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 200))
            pen.setWidthF(1.5)
            self._brush_circle = QtWidgets.QGraphicsEllipseItem()
            self._brush_circle.setPen(pen)
            self._brush_circle.setBrush(_NO_BRUSH)
            self._brush_circle.setZValue(50)
            self._brush_circle.setVisible(False)
            self._scene.addItem(self._brush_circle)

        self._scene.setSceneRect(0, 0, w, h)
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def clean(self):
        self.pos_points.clear()
        self.neg_points.clear()
        self.box = None
        self.current_mask = None
        self.edit_mode = False
        self._painting = False
        self._panning = False
        self._dragging = False
        self._last_xy = None
        self._mask_data = None

        for item in self._point_items:
            if item.scene() is self._scene:
                self._scene.removeItem(item)
        self._point_items.clear()

        for item in (self._box_item, self._drag_rect):
            if item is not None and item.scene() is self._scene:
                self._scene.removeItem(item)
        self._box_item = self._drag_rect = None

        if self._overlay:
            self._overlay.image.fill(QtCore.Qt.GlobalColor.transparent)
            self._overlay.update()
        if self._brush_circle:
            self._brush_circle.setVisible(False)

    def redraw(self):
        if not self.edit_mode:
            self._run_predictor()

    def start_edit_from_current_mask(self) -> bool:
        if self.current_mask is None or self.image_np is None:
            return False
        self.edit_mode = True
        self._mask_data = (self.current_mask > 0).astype(np.uint8) * 255
        self._sync_overlay()
        if self._brush_circle:
            self._brush_circle.setVisible(True)
        return True

    def finish_edit(self) -> Optional[np.ndarray]:
        if not self.edit_mode or self._mask_data is None:
            return None
        return (self._mask_data > 0).astype(np.uint8) * 255

    # ── Фокус и клавиатура ───────────────────────────────────────────────────

    def enterEvent(self, e):
        self.setFocus()
        super().enterEvent(e)

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key.Key_Control and not self._painting:
            self.setDragMode(_SCROLL_HAND)
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == QtCore.Qt.Key.Key_Control and not self._panning:
            self.setDragMode(_NO_DRAG)
        super().keyReleaseEvent(e)

    # ── События мыши ──────────────────────────────────────────────────────────

    def wheelEvent(self, e):
        if self.image_np is None:
            return
        dy = e.angleDelta().y()
        if dy == 0:
            return
        if dy < 0 and self._fits_fully():
            return
        factor = 1.25 if dy > 0 else 0.8
        self.scale(factor, factor)

    def mousePressEvent(self, e):
        if self.image_np is None:
            return

        # Ctrl + ЛКМ → pan (Qt ScrollHandDrag делает всё сам)
        if self.dragMode() == _SCROLL_HAND and e.button() == _LMB:
            self._panning = True
            super().mousePressEvent(e)
            return

        self._press_pos = e.position().toPoint()
        self._press_scene = self.mapToScene(self._press_pos)

        if self.edit_mode:
            if e.button() == _LMB:
                self._painting = True
                self._last_xy = (self._press_scene.x(), self._press_scene.y())
                self._paint_at(self._press_scene.x(), self._press_scene.y())
            return

        # Prompts mode: ЛКМ → потенциальный box
        if e.button() == _LMB:
            self._dragging = True
            self._ensure_drag_rect(self._press_scene)

    def mouseMoveEvent(self, e):
        if self.image_np is None:
            return

        if not self._panning:
            pt = self.mapToScene(e.position().toPoint())
            if self.edit_mode:
                self._move_brush(pt)
                if self._painting:
                    self._paint_at(pt.x(), pt.y())
            elif self._dragging and self._drag_rect and self._press_scene:
                self._drag_rect.setRect(QtCore.QRectF(self._press_scene, pt).normalized())

        # Всегда вызываем super: обрабатывает pan + обновляет позицию мыши для AnchorUnderMouse
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self.image_np is None:
            return

        if self._panning:
            self._panning = False
            super().mouseReleaseEvent(e)
            if not (e.modifiers() & _CTRL):
                self.setDragMode(_NO_DRAG)
            return

        pos = e.position().toPoint()
        click = self._press_pos is not None and \
                (pos - self._press_pos).manhattanLength() < 5

        if self.edit_mode:
            self._painting = False
            self._last_xy = None
            return

        if e.button() == _LMB:
            if click and self._press_scene:
                self.pos_points.append((self._press_scene.x(), self._press_scene.y()))
                self._draw_point(self._press_scene, positive=True)
                self._run_predictor()
            elif self._dragging and self._press_scene:
                self._commit_box(self._press_scene, self.mapToScene(pos))
                self._run_predictor()
            self._dragging = False
            self._remove_drag_rect()

        elif e.button() == _RMB and click and self._press_scene and not (e.modifiers() & _CTRL):
            self.neg_points.append((self._press_scene.x(), self._press_scene.y()))
            self._draw_point(self._press_scene, positive=False)
            self._run_predictor()

    # ── Точки и бокс ──────────────────────────────────────────────────────────

    def _draw_point(self, p: QtCore.QPointF, positive: bool):
        r = 4
        if positive:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black)
            brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        else:
            pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
            brush = _NO_BRUSH
        pen.setWidthF(1.5)
        dot = self._scene.addEllipse(p.x() - r, p.y() - r, 2 * r, 2 * r, pen, brush)
        dot.setZValue(30)
        self._point_items.append(dot)

        if not positive:
            pen2 = QtGui.QPen(QtGui.QColor(255, 0, 0))
            pen2.setWidthF(2)
            for dx, dy in ((1, 1), (1, -1)):
                line = self._scene.addLine(p.x() - 5 * dx, p.y() - 5 * dy,
                                           p.x() + 5 * dx, p.y() + 5 * dy, pen2)
                line.setZValue(31)
                self._point_items.append(line)

    def _ensure_drag_rect(self, p: QtCore.QPointF):
        pen = QtGui.QPen(QtGui.QColor(255, 255, 0))
        pen.setWidthF(2)
        if self._drag_rect is None:
            self._drag_rect = self._scene.addRect(QtCore.QRectF(p, p), pen, _NO_BRUSH)
            self._drag_rect.setZValue(25)
        else:
            self._drag_rect.setRect(QtCore.QRectF(p, p))

    def _commit_box(self, p0: QtCore.QPointF, p1: QtCore.QPointF):
        rect = QtCore.QRectF(p0, p1).normalized()
        if rect.width() < 3 or rect.height() < 3:
            return
        self.box = (rect.left(), rect.top(), rect.right(), rect.bottom())
        pen = QtGui.QPen(QtGui.QColor(255, 255, 0))
        pen.setWidthF(2)
        if self._box_item is None:
            self._box_item = self._scene.addRect(rect, pen, _NO_BRUSH)
            self._box_item.setZValue(24)
        else:
            self._box_item.setRect(rect)

    def _remove_drag_rect(self):
        if self._drag_rect and self._drag_rect.scene() is self._scene:
            self._scene.removeItem(self._drag_rect)
        self._drag_rect = None

    # ── Предиктор и overlay ───────────────────────────────────────────────────

    def _run_predictor(self):
        if self.image_np is None or self.edit_mode:
            return
        mask = self.predictor.predict(self.pos_points, self.neg_points, self.box)
        self.current_mask = mask
        if mask is None or self._overlay is None:
            if self._overlay:
                self._overlay.image.fill(QtCore.Qt.GlobalColor.transparent)
                self._overlay.update()
            return
        self._mask_data = (mask > 0).astype(np.uint8) * 255
        self._sync_overlay()

    def _sync_overlay(self, dirty: Optional[QtCore.QRect] = None):
        if self._overlay is None or self._mask_data is None:
            return
        arr = self._overlay.numpy_view()
        b, g, r, a = _MASK_COLOR_BGRA
        if dirty is None:
            arr[..., 0] = b
            arr[..., 1] = g
            arr[..., 2] = r
            arr[..., 3] = np.where(self._mask_data > 0, a, 0).astype(np.uint8)
            self._overlay.update()
        else:
            y0, y1 = max(0, dirty.top()), min(arr.shape[0], dirty.bottom() + 1)
            x0, x1 = max(0, dirty.left()), min(arr.shape[1], dirty.right() + 1)
            sub = arr[y0:y1, x0:x1]
            sub[..., 0] = b
            sub[..., 1] = g
            sub[..., 2] = r
            sub[..., 3] = np.where(self._mask_data[y0:y1, x0:x1] > 0, a, 0).astype(np.uint8)
            self._overlay.update(QtCore.QRectF(dirty))

    # ── Кисть / ластик ────────────────────────────────────────────────────────

    def _paint_at(self, x: float, y: float):
        if self._mask_data is None:
            return
        lx, ly = self._last_xy or (x, y)
        dist = ((x - lx) ** 2 + (y - ly) ** 2) ** 0.5
        n = max(1, int(dist / max(1, self._brush_radius / 2)))
        xs, ys = np.linspace(lx, x, n + 1), np.linspace(ly, y, n + 1)
        dirty = None
        for xi, yi in zip(xs, ys):
            d = self._stamp(int(round(xi)), int(round(yi)))
            if d:
                dirty = d if dirty is None else dirty.united(d)
        self._last_xy = (x, y)
        self.current_mask = self._mask_data
        if dirty:
            self._sync_overlay(dirty)

    def _stamp(self, cx: int, cy: int) -> Optional[QtCore.QRect]:
        h, w = self._mask_data.shape
        r = self._brush_radius
        x0, x1 = max(cx - r, 0), min(cx + r + 1, w)
        y0, y1 = max(cy - r, 0), min(cy + r + 1, h)
        if x1 <= x0 or y1 <= y0:
            return None
        disk = self._get_disk(r)
        kx0, ky0 = x0 - (cx - r), y0 - (cy - r)
        k = disk[ky0:ky0 + (y1 - y0), kx0:kx0 + (x1 - x0)]
        if not k.any():
            return None
        self._mask_data[y0:y1, x0:x1][k] = 255 if self._tool == "draw" else 0
        return QtCore.QRect(x0, y0, x1 - x0, y1 - y0)

    def _get_disk(self, r: int) -> np.ndarray:
        if r not in self._disk_cache:
            yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
            self._disk_cache[r] = (xx * xx + yy * yy) <= r * r
        return self._disk_cache[r]

    def _move_brush(self, p: QtCore.QPointF):
        if not self._brush_circle:
            return
        r = self._brush_radius
        self._brush_circle.setVisible(True)
        self._brush_circle.setRect(p.x() - r, p.y() - r, r * 2, r * 2)

    # ── Утилиты ───────────────────────────────────────────────────────────────

    def _fits_fully(self) -> bool:
        if self.image_np is None:
            return True
        view_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        h, w = self.image_np.shape[:2]
        return view_rect.contains(QtCore.QRectF(0, 0, w, h))
