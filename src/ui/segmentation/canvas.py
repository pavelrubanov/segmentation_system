"""
Интерактивный канвас сегментации на QGraphicsView.

ЛКМ = позитивная точка / box drag, ПКМ = негативная точка.
Ctrl+ЛКМ = панорама, колесо = зум к курсору.
В режиме редактирования ЛКМ = кисть или ластик.
"""

import numpy as np
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from core.predictor import Predictor

# константы

_LMB  = QtCore.Qt.MouseButton.LeftButton
_RMB  = QtCore.Qt.MouseButton.RightButton
_CTRL = QtCore.Qt.KeyboardModifier.ControlModifier
_HAND = QtWidgets.QGraphicsView.DragMode.ScrollHandDrag
_NONE = QtWidgets.QGraphicsView.DragMode.NoDrag

# setBrush/setPen не принимают BrushStyle/PenStyle напрямую, нужна обёртка
_NO_BRUSH = QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush)
_NO_PEN   = QtGui.QPen(QtCore.Qt.PenStyle.NoPen)

_MASK_COLOR = QtGui.QColor(0, 255, 255, 128)   # cyan semi-transparent
_CLICK_THR  = 5                                  # px, порог клик vs drag


class _Overlay(QtWidgets.QGraphicsItem):
    """ARGB QImage поверх изображения. Кисть рисует через QPainter, маску читаем через numpy."""

    def __init__(self, w: int, h: int):
        super().__init__()
        self._w, self._h = w, h
        self.qimg = self._blank(w, h)

    def boundingRect(self):
        return QtCore.QRectF(0, 0, self._w, self._h)

    def paint(self, painter, *_):
        painter.drawImage(0, 0, self.qimg)

    def resize(self, w, h):
        self.prepareGeometryChange()
        self._w, self._h = w, h
        self.qimg = self._blank(w, h)
        self.update()

    def clear(self):
        self.qimg.fill(QtCore.Qt.GlobalColor.transparent)
        self.update()

    # Format_ARGB32 в памяти хранится как BGRA на little-endian

    def fill_from_mask(self, mask: np.ndarray):
        """Заполнить overlay из бинарной маски (H,W) uint8 0/255."""
        arr = self._numpy()
        on = (mask > 0).astype(np.uint8)
        arr[..., 0] = on * _MASK_COLOR.blue()
        arr[..., 1] = on * _MASK_COLOR.green()
        arr[..., 2] = on * _MASK_COLOR.red()
        arr[..., 3] = on * _MASK_COLOR.alpha()
        self.update()

    def extract_mask(self) -> np.ndarray:
        """Извлечь маску из альфа-канала → (H,W) uint8 0/255."""
        return (self._numpy()[..., 3] > 0).astype(np.uint8) * 255

    def _numpy(self):
        ptr = self.qimg.bits()
        ptr.setsize(self.qimg.sizeInBytes())
        return np.frombuffer(ptr, np.uint8).reshape(self._h, self._w, 4)

    @staticmethod
    def _blank(w, h):
        img = QtGui.QImage(w, h, QtGui.QImage.Format.Format_ARGB32)
        img.fill(QtCore.Qt.GlobalColor.transparent)
        return img



class SegmentationCanvas(QtWidgets.QGraphicsView):

    def __init__(self, predictor: Predictor, parent=None):
        super().__init__(parent)
        self.predictor = predictor

        self._sc = QtWidgets.QGraphicsScene(self)
        self.setScene(self._sc)
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing |
                            QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(_NONE)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.viewport().setCursor(QtCore.Qt.CursorShape.CrossCursor)

        # Публичное состояние (читает window.py)
        self.image_np: Optional[np.ndarray] = None
        self.pos_points: list[tuple] = []
        self.neg_points: list[tuple] = []
        self.box: Optional[tuple] = None
        self.current_mask: Optional[np.ndarray] = None
        self.edit_mode = False

        # Кисть
        self._tool = "draw"
        self._brush_r = 30
        self._painting = False
        self._last_pt: Optional[tuple] = None

        # Элементы сцены (создаются лениво в set_image)
        self._pixmap: Optional[QtWidgets.QGraphicsPixmapItem] = None
        self._overlay: Optional[_Overlay] = None
        self._box_item: Optional[QtWidgets.QGraphicsRectItem] = None
        self._drag_rect: Optional[QtWidgets.QGraphicsRectItem] = None
        self._brush_cursor: Optional[QtWidgets.QGraphicsEllipseItem] = None
        self._pt_items: list = []

        # Мышь
        self._press_vp: Optional[QtCore.QPoint] = None
        self._press_sc: Optional[QtCore.QPointF] = None
        self._dragging = False
        self._panning = False

    # --- публичный API ---

    def set_tool(self, t: str):
        if t in ("draw", "erase"):
            self._tool = t

    def set_brush_radius(self, r: int):
        self._brush_r = max(1, min(200, r))

    def set_image(self, img: np.ndarray):
        self.clean()
        self.image_np = img
        self.predictor.set_image(img)
        h, w = img.shape[:2]

        qimg = QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()
        if self._pixmap is None:
            self._pixmap = self._sc.addPixmap(QtGui.QPixmap.fromImage(qimg))
        else:
            self._pixmap.setPixmap(QtGui.QPixmap.fromImage(qimg))

        if self._overlay is None:
            self._overlay = _Overlay(w, h)
            self._overlay.setZValue(1)
            self._sc.addItem(self._overlay)
        else:
            self._overlay.resize(w, h)

        if self._brush_cursor is None:
            self._brush_cursor = QtWidgets.QGraphicsEllipseItem()
            self._brush_cursor.setPen(
                QtGui.QPen(QtGui.QColor(0, 0, 0, 220), 2.0))
            self._brush_cursor.setBrush(_NO_BRUSH)
            self._brush_cursor.setZValue(100)
            self._brush_cursor.setVisible(False)
            self._sc.addItem(self._brush_cursor)

        self._sc.setSceneRect(0, 0, w, h)
        self.resetTransform()
        self.fitInView(self._sc.sceneRect(),
                       QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def clean(self):
        self.pos_points.clear()
        self.neg_points.clear()
        self.box = None
        self.current_mask = None
        self.edit_mode = False
        self._painting = self._dragging = self._panning = False
        self._last_pt = None

        for it in self._pt_items:
            if it.scene() is self._sc:
                self._sc.removeItem(it)
        self._pt_items.clear()

        for it in (self._box_item, self._drag_rect):
            if it and it.scene() is self._sc:
                self._sc.removeItem(it)
        self._box_item = self._drag_rect = None

        if self._overlay:
            self._overlay.clear()
        if self._brush_cursor:
            self._brush_cursor.setVisible(False)
        self.viewport().setCursor(QtCore.Qt.CursorShape.CrossCursor)

    def redraw(self):
        if not self.edit_mode:
            self._predict()

    def start_edit(self) -> bool:
        if self.current_mask is None or self._overlay is None:
            return False
        self.edit_mode = True
        self._overlay.fill_from_mask(self.current_mask)
        if self._brush_cursor:
            self._brush_cursor.setVisible(True)
        self.viewport().setCursor(QtCore.Qt.CursorShape.BlankCursor)
        return True

    def load_mask_for_edit(self, mask: np.ndarray):
        """Загружает готовую маску и сразу входит в режим редактирования."""
        if self._overlay is None:
            return
        self.current_mask = mask
        self.edit_mode = True
        self._overlay.fill_from_mask(mask)
        if self._brush_cursor:
            self._brush_cursor.setVisible(True)
        self.viewport().setCursor(QtCore.Qt.CursorShape.BlankCursor)

    def finish_edit(self) -> Optional[np.ndarray]:
        if not self.edit_mode or self._overlay is None:
            return None
        return self._overlay.extract_mask()

    # клавиатура (Ctrl → панорама)

    def enterEvent(self, e):
        self.setFocus()          # чтобы key events приходили сразу, без клика

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key.Key_Control and not self._painting:
            self.setDragMode(_HAND)
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == QtCore.Qt.Key.Key_Control and not self._panning:
            self.setDragMode(_NONE)
        super().keyReleaseEvent(e)

    # обработка мыши

    def wheelEvent(self, e):
        if self.image_np is None:
            return
        dy = e.angleDelta().y()
        if dy == 0 or (dy < 0 and self._is_fit()):
            return
        f = 1.25 if dy > 0 else 0.8
        self.scale(f, f)

    def mousePressEvent(self, e):
        if self.image_np is None:
            return
        # Ctrl + ЛКМ → pan (Qt делает всё сам)
        if self.dragMode() == _HAND and e.button() == _LMB:
            self._panning = True
            super().mousePressEvent(e)
            return

        self._press_vp = e.position().toPoint()
        self._press_sc = self.mapToScene(self._press_vp)

        if self.edit_mode:
            if e.button() == _LMB:
                self._painting = True
                self._last_pt = (self._press_sc.x(), self._press_sc.y())
                self._brush_stroke(self._press_sc.x(), self._press_sc.y())
            return

        # Prompts: ЛКМ → потенциальный box
        if e.button() == _LMB:
            self._dragging = True
            pen = QtGui.QPen(QtGui.QColor(255, 255, 0), 2)
            self._drag_rect = self._sc.addRect(
                QtCore.QRectF(self._press_sc, self._press_sc), pen, _NO_BRUSH)
            self._drag_rect.setZValue(5)

    def mouseMoveEvent(self, e):
        if self.image_np is None:
            return
        if not self._panning:
            pt = self.mapToScene(e.position().toPoint())
            if self.edit_mode:
                self._show_brush(pt)
                if self._painting:
                    self._brush_stroke(pt.x(), pt.y())
            elif self._dragging and self._drag_rect and self._press_sc:
                self._drag_rect.setRect(
                    QtCore.QRectF(self._press_sc, pt).normalized())
        # нужно для pan и трекинга мыши в AnchorUnderMouse
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self.image_np is None:
            return
        # Конец pan
        if self._panning:
            self._panning = False
            super().mouseReleaseEvent(e)
            if not (e.modifiers() & _CTRL):
                self.setDragMode(_NONE)
            return

        vp = e.position().toPoint()
        click = (self._press_vp is not None
                 and (vp - self._press_vp).manhattanLength() < _CLICK_THR)

        if self.edit_mode:
            self._painting = False
            self._last_pt = None
            return

        if e.button() == _LMB:
            if click and self._press_sc:
                self._add_point(self._press_sc, positive=True)
            elif self._dragging and self._press_sc:
                self._set_box(self._press_sc, self.mapToScene(vp))
            self._dragging = False
            self._safe_remove(self._drag_rect)
            self._drag_rect = None

        elif (e.button() == _RMB and click
              and self._press_sc and not (e.modifiers() & _CTRL)):
            self._add_point(self._press_sc, positive=False)

    # промпты: точки и box

    def _add_point(self, p, positive: bool):
        (self.pos_points if positive else self.neg_points).append(
            (p.x(), p.y()))

        r = 4
        color = QtGui.QColor(0, 255, 0) if positive else QtGui.QColor(255, 0, 0)
        pen = QtGui.QPen(
            QtCore.Qt.GlobalColor.black if positive else color, 1.5)
        brush = QtGui.QBrush(color) if positive else _NO_BRUSH
        dot = self._sc.addEllipse(
            p.x() - r, p.y() - r, 2 * r, 2 * r, pen, brush)
        dot.setZValue(10)
        self._pt_items.append(dot)

        if not positive:                    # крестик ×
            xpen = QtGui.QPen(color, 2)
            for dx, dy in ((1, 1), (1, -1)):
                ln = self._sc.addLine(
                    p.x() - 5*dx, p.y() - 5*dy,
                    p.x() + 5*dx, p.y() + 5*dy, xpen)
                ln.setZValue(11)
                self._pt_items.append(ln)

        self._predict()

    def _set_box(self, p0, p1):
        rect = QtCore.QRectF(p0, p1).normalized()
        if rect.width() < 3 or rect.height() < 3:
            return
        self.box = (rect.left(), rect.top(), rect.right(), rect.bottom())
        pen = QtGui.QPen(QtGui.QColor(255, 255, 0), 2)
        if self._box_item is None:
            self._box_item = self._sc.addRect(rect, pen, _NO_BRUSH)
            self._box_item.setZValue(4)
        else:
            self._box_item.setRect(rect)
        self._predict()

    # вызов предиктора

    def _predict(self):
        if self.image_np is None or self.edit_mode:
            return
        mask = self.predictor.predict(
            self.pos_points, self.neg_points, self.box)
        self.current_mask = mask
        if mask is not None and self._overlay:
            self._overlay.fill_from_mask(mask)
        elif self._overlay:
            self._overlay.clear()

    # рисование кистью через QPainter

    def _brush_stroke(self, x: float, y: float):
        if self._overlay is None:
            return
        p = QtGui.QPainter(self._overlay.qimg)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        p.setPen(_NO_PEN)
        p.setBrush(_MASK_COLOR)
        if self._tool == "draw":
            p.setCompositionMode(
                QtGui.QPainter.CompositionMode.CompositionMode_Source)
        else:
            p.setCompositionMode(
                QtGui.QPainter.CompositionMode.CompositionMode_Clear)

        # Интерполяция, чтобы линия была без разрывов
        lx, ly = self._last_pt or (x, y)
        dist = ((x - lx)**2 + (y - ly)**2) ** 0.5
        n = max(1, int(dist / max(1, self._brush_r // 2)))
        r = float(self._brush_r)
        for i in range(n + 1):
            t = i / n
            p.drawEllipse(
                QtCore.QPointF(lx + (x - lx) * t, ly + (y - ly) * t), r, r)
        p.end()

        self._last_pt = (x, y)
        self._overlay.update()

    def _show_brush(self, pt):
        if self._brush_cursor:
            r = self._brush_r
            self._brush_cursor.setVisible(True)
            self._brush_cursor.setRect(pt.x() - r, pt.y() - r, 2 * r, 2 * r)

    # утилиты

    def _safe_remove(self, item):
        """removeItem с проверкой — PyQt6 падает если item не в этой сцене."""
        if item and item.scene() is self._sc:
            self._sc.removeItem(item)

    def _is_fit(self) -> bool:
        """True если изображение целиком видно (нельзя zoom out дальше)."""
        if self.image_np is None:
            return True
        vr = self.mapToScene(self.viewport().rect()).boundingRect()
        h, w = self.image_np.shape[:2]
        return vr.contains(QtCore.QRectF(0, 0, w, h))
