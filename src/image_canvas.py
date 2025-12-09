import numpy as np
from PyQt5 import QtCore
import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as mpatches
from predictor import Predictor

class ImageCanvas(FigureCanvas):
    """
    Matplotlib canvas внутри PyQt5.
    Показывает картинку, точки, box и текущую маску.
    - ЛКМ drag: RectangleSelector -> box
    - ЛКМ/ПКМ click: точки (через press+release, чтобы не ставить точку при drag)
    """

    def __init__(self, predictor: Predictor):
        self.predictor = predictor
        
        # состояние
        self.image_np = None
        self.pos_points = []
        self.neg_points = []
        self.box = None                 # (x0,y0,x1,y1) или None
        self.current_mask = None        # np.uint8 [H,W] 0/1
        self.image_is_set = False

        fig = Figure()
        self.axes = fig.add_subplot(111)
        #self.axes.set_axis_off()
        super().__init__(fig)

        self.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)

        # состояние для click-vs-drag
        self._press = None  # (x_px, y_px, button)

        # события мыши
        self.mpl_connect("button_press_event", self.on_press)
        self.mpl_connect("button_release_event", self.on_release)

        # RectangleSelector всегда активен, только ЛКМ-drag
        # minspanx/minspany в координатах данных (тут ~ пиксели изображения)
        self.rect_selector = RectangleSelector(
            self.axes,
            self.on_box_select,
            useblit=True,
            button=[1],
            interactive=False,
            spancoords="data",
            minspanx=5,
            minspany=5,
        )
        self.rect_selector.set_active(True)

    # ---------- Input handling ----------

    def on_press(self, event):
        if event.inaxes != self.axes:
            self._press = None
            return
        if event.xdata is None or event.ydata is None:
            self._press = None
            return
        # event.x/event.y — пиксели экрана; удобно мерять смещение
        self._press = (float(event.x), float(event.y), int(event.button))

    def on_release(self, event):
        """
        Ставим точки только если это клик (малое смещение).
        Если это drag ЛКМ, RectangleSelector сам вызовет on_box_select(...)
        """
        if self._press is None:
            return
        if event.inaxes != self.axes:
            return
        if event.xdata is None or event.ydata is None:
            return

        x0_px, y0_px, button = self._press

        CLICK_THR_PX = 3.0
        is_click = (abs(float(event.x) - x0_px) <= CLICK_THR_PX and abs(float(event.y) - y0_px) <= CLICK_THR_PX)
        if not is_click:
            return  # drag

        x = float(event.xdata)
        y = float(event.ydata)

        # 1=ЛКМ -> positive, 3=ПКМ -> negative
        if button == 1:
            self.pos_points.append((x, y))
        elif button == 3:
            self.neg_points.append((x, y))

        self.redraw()

    def on_box_select(self, eclick, erelease):
        if eclick.xdata is None or eclick.ydata is None:
            return
        if erelease.xdata is None or erelease.ydata is None:
            return

        x0, y0 = float(eclick.xdata), float(eclick.ydata)
        x1, y1 = float(erelease.xdata), float(erelease.ydata)

        xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
        ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)

        # safety: хотя minspan уже стоит, фильтр оставим
        if abs(xmax - xmin) < 2 or abs(ymax - ymin) < 2:
            return

        self.box = (xmin, ymin, xmax, ymax)
        self.redraw()

    # ---------- Canvas drawing ----------
    def clean(self):
        self.pos_points = []
        self.neg_points = []
        self.box = None
        self.current_mask = None

    def set_image(self, image_np):
        self.clean()

        self.image_np = image_np
        self.predictor.set_image(self.image_np)
        self.image_is_set = True
        self.redraw()

    def redraw(self):
        self.current_mask = self.predictor.predict(self.pos_points, self.neg_points, self.box) if self.image_is_set else None

        self.axes.clear()
        #self.axes.set_axis_off()

        if self.image_np is not None:
            self.axes.imshow(self.image_np)

        if self.current_mask is not None:
            self.axes.imshow(
                np.ma.masked_where(self.current_mask == 0, self.current_mask),
                alpha=0.5,
                cmap="jet"
            )

        # финальный box
        if self.box is not None:
            x0, y0, x1, y1 = self.box
            rect = mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                fill=False, linewidth=2.0, edgecolor="yellow",
            )
            self.axes.add_patch(rect)

        # точки
        if self.pos_points:
            xs = [p[0] for p in self.pos_points]
            ys = [p[1] for p in self.pos_points]
            self.axes.scatter(xs, ys, c="lime", marker="o", s=60, edgecolors="black", linewidths=1.0)

        if self.neg_points:
            xs = [p[0] for p in self.neg_points]
            ys = [p[1] for p in self.neg_points]
            self.axes.scatter(xs, ys, c="red", marker="x", s=60, linewidths=2.0)

        self.draw_idle()

