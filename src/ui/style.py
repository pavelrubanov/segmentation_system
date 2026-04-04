from PyQt6 import QtWidgets, QtGui, QtCore


def apply_theme(app: QtWidgets.QApplication) -> None:
    app.setFont(QtGui.QFont("Segoe UI", 10))
    app.setStyleSheet(_QSS)


def make_icon(draw_fn, size: int = 24) -> QtGui.QIcon:
    """Создаёт иконку, рисуя draw_fn(painter, size) на прозрачном QPixmap."""
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    draw_fn(p, size)
    p.end()
    return QtGui.QIcon(pm)


# ── Программные иконки инструментов ──────────────────────────────────────────

def icon_crosshair(size: int = 24) -> QtGui.QIcon:
    def _draw(p: QtGui.QPainter, s: int):
        pen = QtGui.QPen(QtGui.QColor("#1E1F22"), 2)
        p.setPen(pen)
        c = s / 2
        gap = 3
        p.drawLine(QtCore.QPointF(c, 0), QtCore.QPointF(c, c - gap))
        p.drawLine(QtCore.QPointF(c, c + gap), QtCore.QPointF(c, s))
        p.drawLine(QtCore.QPointF(0, c), QtCore.QPointF(c - gap, c))
        p.drawLine(QtCore.QPointF(c + gap, c), QtCore.QPointF(s, c))
    return make_icon(_draw, size)


def icon_brush(size: int = 24) -> QtGui.QIcon:
    def _draw(p: QtGui.QPainter, s: int):
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor("#3574F0"))
        r = s * 0.32
        p.drawEllipse(QtCore.QPointF(s / 2, s / 2), r, r)
    return make_icon(_draw, size)


def icon_eraser(size: int = 24) -> QtGui.QIcon:
    def _draw(p: QtGui.QPainter, s: int):
        pen = QtGui.QPen(QtGui.QColor("#1E1F22"), 2)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        m = s * 0.2
        p.drawRoundedRect(QtCore.QRectF(m, m, s - 2 * m, s - 2 * m), 3, 3)
    return make_icon(_draw, size)


# ── QSS ──────────────────────────────────────────────────────────────────────

_QSS = """
/* ── Глобальные ─────────────────────────────────────────────────── */

QMainWindow, QDialog {
    background: #F5F6FA;
}

QWidget#sidePanel {
    background: #FFFFFF;
    border-left: 1px solid #D4D4D8;
}

/* ── QPushButton ────────────────────────────────────────────────── */

QPushButton {
    background: #FFFFFF;
    border: 1px solid #D4D4D8;
    border-radius: 6px;
    padding: 7px 18px;
    color: #1E1F22;
    min-height: 20px;
}
QPushButton:hover {
    background: #F0F4FF;
    border-color: #3574F0;
}
QPushButton:pressed {
    background: #D8E4FC;
}
QPushButton:disabled {
    background: #F0F0F0;
    color: #A0A0A0;
    border-color: #E0E0E0;
}

QPushButton#primary {
    background: #3574F0;
    color: #FFFFFF;
    border: none;
}
QPushButton#primary:hover {
    background: #4A86F7;
}
QPushButton#primary:pressed {
    background: #2460D4;
}
QPushButton#primary:disabled {
    background: #A0BEF5;
}

QPushButton#success {
    background: #36803F;
    color: #FFFFFF;
    border: none;
}
QPushButton#success:hover {
    background: #3F9A4A;
}
QPushButton#success:pressed {
    background: #2D6B34;
}

QPushButton#flat {
    background: transparent;
    border: none;
    color: #6C707E;
}
QPushButton#flat:hover {
    color: #1E1F22;
}

QPushButton#card {
    background: #FFFFFF;
    border: 1px solid #D4D4D8;
    border-radius: 8px;
    padding: 14px 20px;
    text-align: left;
    font-size: 13px;
}
QPushButton#card:hover {
    background: #F0F4FF;
    border-color: #3574F0;
}
QPushButton#card:pressed {
    background: #D8E4FC;
}

/* ── QToolBar / QToolButton ─────────────────────────────────────── */

QToolBar {
    background: #FFFFFF;
    border-bottom: 1px solid #D4D4D8;
    spacing: 4px;
    padding: 2px 6px;
}

QToolButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px 8px;
    color: #1E1F22;
}
QToolButton:hover {
    background: #F0F4FF;
    border-color: #D4D4D8;
}
QToolButton:checked {
    background: #E8EEF9;
    border-color: #3574F0;
}
QToolButton:disabled {
    color: #A0A0A0;
}

/* ── QMenuBar / QMenu ───────────────────────────────────────────── */

QMenuBar {
    background: #FFFFFF;
    border-bottom: 1px solid #E8E8EC;
    padding: 2px 0;
}
QMenuBar::item {
    padding: 4px 10px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background: #E8EEF9;
}

QMenu {
    background: #FFFFFF;
    border: 1px solid #D4D4D8;
    border-radius: 6px;
    padding: 4px 0;
}
QMenu::item {
    padding: 6px 28px 6px 12px;
}
QMenu::item:selected {
    background: #E8EEF9;
}
QMenu::separator {
    height: 1px;
    background: #E8E8EC;
    margin: 4px 8px;
}

/* ── QGroupBox ──────────────────────────────────────────────────── */

QGroupBox {
    border: 1px solid #D4D4D8;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 6px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: #1E1F22;
}



/* ── QRadioButton ───────────────────────────────────────────────── */

QRadioButton {
    spacing: 6px;
    color: #1E1F22;
}
QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border-radius: 8px;
    border: 2px solid #D4D4D8;
    background: #FFFFFF;
}
QRadioButton::indicator:checked {
    border-color: #3574F0;
    background: #3574F0;
}
QRadioButton::indicator:hover {
    border-color: #3574F0;
}

/* ── QListWidget (masks buffer) ─────────────────────────────────── */

QListWidget {
    background: #FAFAFA;
    border: none;
    outline: none;
}
QListWidget::item {
    border: 2px solid transparent;
    border-radius: 4px;
    padding: 4px;
    margin: 2px;
}
QListWidget::item:selected {
    border-color: #3574F0;
    background: #E8EEF9;
}
QListWidget::item:hover {
    background: #F0F0F5;
}

/* ── QSplitter ──────────────────────────────────────────────────── */

QSplitter::handle {
    background: #E8E8EC;
    width: 3px;
}
QSplitter::handle:hover {
    background: #3574F0;
}

/* ── QStatusBar ─────────────────────────────────────────────────── */

QStatusBar {
    background: #FFFFFF;
    border-top: 1px solid #E8E8EC;
    color: #6C707E;
    font-size: 9pt;
}
QStatusBar::item {
    border: none;
}

/* ── QLabel ──────────────────────────────────────────────────────── */

QLabel#heading {
    font-size: 18px;
    font-weight: 700;
    color: #1E1F22;
}

QLabel#subheading {
    font-size: 11px;
    font-weight: 600;
    color: #1E1F22;
}

QLabel#subtitle {
    font-size: 10pt;
    color: #6C707E;
}

QLabel#secondary {
    font-size: 9pt;
    color: #6C707E;
}

/* ── QProgressDialog ────────────────────────────────────────────── */

QProgressBar {
    border: 1px solid #D4D4D8;
    border-radius: 4px;
    text-align: center;
    background: #F0F0F0;
}
QProgressBar::chunk {
    background: #3574F0;
    border-radius: 3px;
}

/* ── QToolTip ───────────────────────────────────────────────────── */

QToolTip {
    background: #1E1F22;
    color: #FFFFFF;
    border: none;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 9pt;
}
"""
