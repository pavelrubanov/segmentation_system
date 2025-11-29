import numpy as np
from PyQt5 import QtCore
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as mpatches

class ImageCanvas(FigureCanvas):
    """
    Matplotlib canvas внутри PyQt5.
    Показывает картинку, точки, box и текущую маску.
    - ЛКМ drag: RectangleSelector -> box
    - ЛКМ/ПКМ click: точки (через press+release, чтобы не ставить точку при drag)
    """

    def __init__(self, controller):
        self.controller = controller

        fig = Figure()
        self.ax = fig.add_subplot(111)
        self.ax.set_axis_off()
        super().__init__(fig)

        self.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)

        # состояние для click-vs-drag
        self._press = None  # (x_px, y_px, button, xdata, ydata)

        # события мыши
        self.mpl_connect("button_press_event", self.on_press)
        self.mpl_connect("button_release_event", self.on_release)

        # данные
        self.image = None       # np.uint8 [H,W,3]
        self.mask = None        # np.uint8 [H,W] 0/1
        self.pos_points = []    # list[(x,y)]
        self.neg_points = []    # list[(x,y)]
        self.box = None         # (x0,y0,x1,y1) or None

        # artists (чтобы не делать ax.clear() каждый раз)
        self.img_artist = None
        self.mask_artist = None
        self.pos_scatter = None
        self.neg_scatter = None
        self.box_patch = None

        # RectangleSelector всегда активен, только ЛКМ-drag
        # minspanx/minspany в координатах данных (тут ~ пиксели изображения)
        self.rect_selector = RectangleSelector(
            self.ax,
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
        if event.inaxes != self.ax:
            self._press = None
            return
        if event.xdata is None or event.ydata is None:
            self._press = None
            return
        # event.x/event.y — пиксели экрана; удобно мерять смещение
        self._press = (float(event.x), float(event.y), int(event.button), float(event.xdata), float(event.ydata))

    def on_release(self, event):
        """
        Ставим точки только если это клик (малое смещение).
        Если это drag ЛКМ, RectangleSelector сам вызовет on_box_select(...)
        """
        if self._press is None:
            return
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x0_px, y0_px, button, _, _ = self._press
        dx = abs(float(event.x) - x0_px)
        dy = abs(float(event.y) - y0_px)

        CLICK_THR_PX = 3.0
        is_click = (dx <= CLICK_THR_PX and dy <= CLICK_THR_PX)
        if not is_click:
            return  # drag

        x = float(event.xdata)
        y = float(event.ydata)

        # 1=ЛКМ -> positive, 3=ПКМ -> negative
        if button == 1:
            self.controller.add_point((x, y), positive=True)
        elif button == 3:
            self.controller.add_point((x, y), positive=False)

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

        self.controller.set_box((xmin, ymin, xmax, ymax))

    # ---------- Rendering ----------

    def _ensure_artists(self):
        if self.image is None:
            return

        if self.img_artist is None:
            self.img_artist = self.ax.imshow(self.image)
            self.ax.set_axis_off()

        if self.mask_artist is None:
            dummy = np.ma.masked_where(np.zeros(self.image.shape[:2], dtype=np.uint8) == 0,
                                       np.zeros(self.image.shape[:2], dtype=np.uint8))
            self.mask_artist = self.ax.imshow(
                dummy, alpha=0.5, cmap="jet", interpolation="nearest", visible=False
            )

        if self.pos_scatter is None:
            self.pos_scatter = self.ax.scatter(
                [], [], c="lime", marker="o", s=60, edgecolors="black", linewidths=1.0
            )

        if self.neg_scatter is None:
            self.neg_scatter = self.ax.scatter(
                [], [], c="red", marker="x", s=60, linewidths=2.0
            )

        if self.box_patch is None:
            self.box_patch = mpatches.Rectangle(
                (0, 0), 1, 1, fill=False, linewidth=2.0, edgecolor="yellow", visible=False
            )
            self.ax.add_patch(self.box_patch)

    def set_image(self, img_np):
        self.image = img_np
        self.mask = None
        self.pos_points = []
        self.neg_points = []
        self.box = None

        # сброс artists (проще и надежнее при смене размера картинки)
        self.ax.clear()
        self.ax.set_axis_off()
        self.img_artist = None
        self.mask_artist = None
        self.pos_scatter = None
        self.neg_scatter = None
        self.box_patch = None

        # RectangleSelector продолжает быть привязан к self.ax (ось та же),
        # но после clear() его визуальный прямоугольник исчезает — это нормально.

        self._ensure_artists()
        self._render()
        self.draw()

    def update_overlay(self, mask, pos_points, neg_points, box):
        self.mask = mask
        self.pos_points = list(pos_points)
        self.neg_points = list(neg_points)
        self.box = box
        self._ensure_artists()
        self._render()
        self.draw_idle()

    def clear_points_and_mask(self):
        self.mask = None
        self.pos_points = []
        self.neg_points = []
        self.box = None
        self._ensure_artists()
        self._render()
        self.draw_idle()

    def _render(self):
        if self.image is None:
            return

        # image
        self.img_artist.set_data(self.image)

        # mask
        if self.mask is None:
            self.mask_artist.set_visible(False)
        else:
            m = np.ma.masked_where(self.mask == 0, self.mask)
            self.mask_artist.set_data(m)
            self.mask_artist.set_visible(True)

        # points
        if self.pos_points:
            xs = [p[0] for p in self.pos_points]
            ys = [p[1] for p in self.pos_points]
            self.pos_scatter.set_offsets(np.column_stack([xs, ys]))
        else:
            self.pos_scatter.set_offsets(np.zeros((0, 2)))

        if self.neg_points:
            xs = [p[0] for p in self.neg_points]
            ys = [p[1] for p in self.neg_points]
            self.neg_scatter.set_offsets(np.column_stack([xs, ys]))
        else:
            self.neg_scatter.set_offsets(np.zeros((0, 2)))

        # box
        if self.box is None:
            self.box_patch.set_visible(False)
        else:
            x0, y0, x1, y1 = self.box
            self.box_patch.set_bounds(x0, y0, x1 - x0, y1 - y0)
            self.box_patch.set_visible(True)
