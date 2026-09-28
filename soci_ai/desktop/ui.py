"""QtWidgets desktop presentation. Capture, inference and reporting live elsewhere."""
from __future__ import annotations

import html
import math
from copy import deepcopy
from datetime import datetime

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPalette, QPen, QTextDocument
from PySide6.QtWidgets import (
    QBoxLayout, QComboBox, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QStackedWidget,
    QTextBrowser, QVBoxLayout, QWidget, QMessageBox,
)

from .tree import LivingTree
from .charts import SessionTimeline, time_label


def numeric(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def fmt(value, suffix='', decimals=1):
    result = numeric(value)
    return '—' if result is None else f'{result:.{decimals}f}{suffix}'


def label(text, name='', wrap=False):
    item = QLabel(text)
    if name:
        item.setObjectName(name)
    item.setWordWrap(wrap)
    return item


class ElidedLabel(QLabel):
    """One readable line with full text available on hover, never layout growth."""
    def __init__(self, text='', name='', parent=None):
        super().__init__(text, parent)
        self.setObjectName(name)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setToolTip(text)

    def setText(self, text):
        super().setText(text)
        self.setToolTip(text)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setFont(self.font())
        painter.setPen(self.palette().color(QPalette.ColorRole.WindowText))
        text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, max(0, self.width()-2))
        painter.drawText(self.rect(), self.alignment() | Qt.AlignmentFlag.AlignVCenter, text)
        painter.end()


class MathFormula(QWidget):
    """Native rich-text math, scaled as a whole instead of broken mid-equation."""
    def __init__(self, equation, caption, color='#99e9cb', height=62):
        super().__init__()
        self.setFixedHeight(height)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setAccessibleName(caption)
        self.setToolTip(caption)
        self.document = QTextDocument()
        self.document.setDefaultFont(QFont('Cambria Math', 15))
        self.document.setHtml('<div style="color:' + color + ';white-space:nowrap">' + equation + '</div>')
        self.document.setTextWidth(-1)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor('#071822'))
        painter.drawRoundedRect(QRectF(self.rect()), 9, 9)
        size = self.document.size()
        scale = min(1, (self.width()-24)/max(1, size.width()), (self.height()-12)/max(1, size.height()))
        painter.translate(12, (self.height()-size.height()*scale)/2)
        painter.scale(max(.1, scale), max(.1, scale))
        self.document.drawContents(painter)
        painter.end()


class CameraPreview(QWidget):
    """Mirrors pixels and normalised face bounds through one aspect-fit mapping."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(160, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image = QImage()
        self.box = None
        self.placeholder = '摄像头尚未启动'

    def set_frame(self, frame_bgr, box=None):
        if frame_bgr is None:
            self.image = QImage()
        else:
            # Copy owns the storage beyond the worker callback. Reverse channels
            # and horizontal pixels together so preview and bounds agree.
            import numpy as np
            rgb = np.ascontiguousarray(frame_bgr[:, ::-1, :3][:, :, ::-1])
            height, width, _ = rgb.shape
            self.image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.box = box
        self.update()

    def image_rect(self):
        if self.image.isNull():
            return QRectF()
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        width, height = self.image.width() * scale, self.image.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def box_rect(self):
        if not self.box or self.image.isNull():
            return QRectF()
        try:
            x = float(self.box.get('x', self.box.get('left', 0)))
            y = float(self.box.get('y', self.box.get('top', 0)))
            width = float(self.box.get('width', self.box.get('w', 0)))
            height = float(self.box.get('height', self.box.get('h', 0)))
            if not all(math.isfinite(v) for v in (x, y, width, height)):
                return QRectF()
            x0, x1 = max(0, x), min(1, x + width)
            y0, y1 = max(0, y), min(1, y + height)
            if x1 <= x0 or y1 <= y0:
                return QRectF()
            target = self.image_rect()
            return QRectF(target.x() + (1 - x1) * target.width(),
                          target.y() + y0 * target.height(),
                          (x1 - x0) * target.width(), (y1 - y0) * target.height())
        except (TypeError, ValueError):
            return QRectF()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor('#030c13'))
        painter.setPen(QPen(QColor('#1d3d46'), 1))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(.5, .5, -.5, -.5), 12, 12)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 11, 11)
        painter.setClipPath(clip)
        if self.image.isNull():
            painter.setPen(QColor('#759aab'))
            painter.setFont(QFont('Microsoft YaHei UI', 10))
            painter.drawText(self.rect().adjusted(16, 16, -16, -16), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                             self.placeholder + '\n\n开始后显示本机实时画面')
        else:
            painter.drawImage(self.image_rect(), self.image)
            bbox = self.box_rect()
            if not bbox.isEmpty():
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor('#64e2b8'), 1.6))
                painter.drawRoundedRect(bbox, 7, 7)
        painter.end()


class Metric(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        name = label(title, 'muted')
        name.setMinimumWidth(62)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(5)
        self.value = label('—', 'metricValue')
        self.value.setFixedWidth(36)
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight)
        row.addWidget(name)
        row.addWidget(self.bar, 1)
        row.addWidget(self.value)

    def set_value(self, value, scale=100):
        number = numeric(value)
        self.bar.setValue(0 if number is None else round(max(0, min(100, number * scale))))
        self.value.setText('—' if number is None else f'{max(0, min(100, number * scale)):.0f}')


class ScoreRing(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(152)
        self.value = None

    def set_value(self, value):
        self.value = numeric(value)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        diameter = min(154, self.width()-14, self.height()-20)
        box = QRectF((self.width()-diameter)/2, 5, diameter, diameter)
        p.setPen(QPen(QColor('#1b3540'), 5))
        p.drawEllipse(box)
        if self.value is not None:
            # These boundaries match core mode: SBI <25 / <60 / >=60.
            color = QColor('#68dfb0') if self.value > 75 else QColor('#e9b66e') if self.value > 40 else QColor('#ff586f')
            p.setPen(QPen(color, 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(box, 90 * 16, -round(max(0, min(100, self.value)) / 100 * 360 * 16))
        p.setPen(QColor('#ff8796') if self.value is not None and self.value <= 40 else QColor('#e6f8f1'))
        compact = self.height() <= 100
        number_font = QFont('Segoe UI', -1, QFont.Weight.Light)
        number_font.setPixelSize(max(10, round(diameter * (.36 if compact else .30))))
        p.setFont(number_font)
        p.drawText(box if compact else box.adjusted(0, 0, 0, -34),
                   Qt.AlignmentFlag.AlignCenter, fmt(self.value, decimals=0))
        p.setPen(QColor('#7197a2'))
        caption_font = QFont('Microsoft YaHei UI')
        caption_font.setPixelSize(9 if self.height() < 100 else 11)
        p.setFont(caption_font)
        caption = (QRectF(0, box.bottom()+1, self.width(), 14) if compact else
                   QRectF(box.left(), box.bottom()-29, diameter, 18))
        p.drawText(caption, Qt.AlignmentFlag.AlignCenter, '友好度 / 100')
        p.end()


STYLE = '''
QWidget { color: #d7e9ed; font-family: "Microsoft YaHei UI", "Segoe UI"; font-size: 13px; }
QMainWindow, QWidget#appRoot, QWidget#scrollPage { background: #06111b; }
QWidget#sidebar { background: #07131e; border-right: 1px solid #19303c; }
QFrame#card { background: #0b1d29; border: 1px solid #1b3641; border-radius: 15px; }
QLabel { background: transparent; border: none; }
QLabel#brand { color: #e8faf7; font-size: 23px; font-weight: 650; }
QLabel#pageTitle { color: #e6f6f4; font-size: 24px; font-weight: 550; }
QLabel#cardTitle { color: #bcd8df; font-size: 13px; font-weight: 550; }
QLabel#introTitle { font-size: 21px; font-weight: 700; }
QLabel#introBody { color: #b4cdd6; font-size: 14px; }
QLabel#kicker { color: #6f929f; font-size: 10px; }
QLabel#muted { color: #86a7b5; font-size: 12px; }
QLabel#metricValue { color: #a9cad1; font-size: 11px; }
QLabel#status { color: #71dfbb; font-size: 12px; }
QLabel#telemetry { color: #718e9d; font-size: 10px; }
QLabel#statValue { color: #e4f4ef; font-size: 29px; font-weight: 450; }
QLabel#reviewNote { color: #a2bfca; font-size: 13px; }
QLabel#momentTime { color: #77dabd; font-size: 12px; }
QFrame#moment { background: #10232f; border: 1px solid #213b46; border-radius: 9px; }
QLabel#suggestionTitle { color: #8de6c9; font-size: 16px; font-weight: 700; }
QLabel#suggestionBody { color: #d9eee8; font-size: 16px; font-weight: 600; }
QFrame#feedback { background: #10282c; border: 1px solid #2b514d; border-radius: 12px; }
QFrame#feedback[tone="language"] { background: #301d27; border: 2px solid #ed6b7e; }
QFrame#feedback[tone="pending"] { background: #2b261c; border: 1px solid #caa55e; }
QLabel#feedbackTitle { color: #d7f4e7; font-size: 17px; font-weight: 700; }
QFrame#feedback[tone="language"] QLabel#feedbackTitle { color: #ff95a5; }
QFrame#feedback[tone="pending"] QLabel#feedbackTitle { color: #f0cf83; }
QLabel#feedbackTags { color: #edc4b0; font-size: 12px; font-weight: 650; }
QLabel#feedbackQuote { color: #b6cdd3; font-size: 13px; }
QLabel#feedbackAction { color: #e8f5e9; font-size: 16px; font-weight: 650; }
QPushButton { background: #102733; border: 1px solid #27424c; border-radius: 9px; color: #b5d1d9; padding: 9px 14px; }
QPushButton:hover { border-color: #55bdaa; color: #e3f5f0; }
QPushButton:disabled { color: #46626f; border-color: #19333e; background: #0c202c; }
QPushButton#primary { background: #62dfb8; color: #06221c; border-color: #62dfb8; font-weight: 650; }
QPushButton#primary:disabled { background: #174237; color: #77aa9b; border-color: #255549; }
QPushButton#nav { border: none; text-align: left; background: transparent; color: #7797a7; padding: 13px 10px; }
QPushButton#nav:checked { color: #75e1c1; background: #102e35; border-left: 3px solid #6edfb9; border-radius: 5px; }
QComboBox { background: #081923; border: 1px solid #25414d; border-radius: 7px; padding: 6px 9px; color: #afccd4; }
QComboBox:disabled { color: #486674; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView { background: #102833; color: #d0e7e9; selection-background-color: #24564e; }
QProgressBar { background: #1b3540; border: none; border-radius: 2px; }
QProgressBar::chunk { background: #68dbb1; border-radius: 2px; }
QTextBrowser { background: #0b1d29; border: 1px solid #1b3641; border-radius: 12px; padding: 14px; color: #b8d1d9; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #0b1d29; width: 6px; }
QScrollBar::handle:vertical { background: #2e505c; border-radius: 3px; min-height: 25px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QDialog { background: #07131e; }
'''


class StudioWindow(QMainWindow):
    start_requested = Signal()
    stop_requested = Signal()
    calibrate_requested = Signal()
    export_requested = Signal()
    closing = Signal()
    history_selected = Signal(object)
    history_refresh_requested = Signal()
    history_more_requested = Signal()
    history_delete_requested = Signal(str)
    history_retry_requested = Signal()
    history_save_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle('SOCI AI · 实时交互自检 · 09.27 分层提示版')
        self.resize(1360, 860)
        self.setMinimumSize(760, 480)
        self.setStyleSheet(STYLE)
        self._snapshot = {}
        self._report = {}
        self._history_items = []
        self._history_selection = None
        self._save_available = False
        self._deferred_close = False
        self._close_notified = False
        central = QWidget()
        central.setObjectName('appRoot')
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setSpacing(0)
        outer.setContentsMargins(0, 0, 0, 0)
        sidebar = self.sidebar = QWidget()
        sidebar.setObjectName('sidebar')
        sidebar.setFixedWidth(176)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(20, 29, 18, 20)
        self.brand_label = label('SOCI AI', 'brand')
        side.addWidget(self.brand_label)
        side.addWidget(label('交流自检助手', 'kicker'))
        side.addSpacing(28)
        self.nav = []
        for index, text in enumerate(('实时自检', '会话回顾', '系统介绍')):
            button = QPushButton(f'0{index + 1}   {text}')
            button.setObjectName('nav')
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, i=index: self._show_page(i))
            self.nav.append(button)
            side.addWidget(button)
        side.addStretch()
        self.sidebar_status = label('●  等待开始', 'status')
        side.addWidget(self.sidebar_status)
        side.addSpacing(12)
        outer.addWidget(sidebar)
        main = QWidget()
        layout = QVBoxLayout(main)
        self.main_layout = layout
        layout.setContentsMargins(24, 24, 24, 18)
        layout.setSpacing(12)
        top = self.header_layout = QGridLayout()
        top.setHorizontalSpacing(12)
        top.setVerticalSpacing(10)
        self.heading = QWidget()
        title = QVBoxLayout(self.heading)
        title.setContentsMargins(0, 0, 0, 0)
        self.page_kicker = label('', 'kicker')
        self.page_kicker.hide()
        self.page_title = label('让每一次表达，温和而清晰', 'pageTitle')
        self.page_title.setWordWrap(True)
        title.addWidget(self.page_title)
        top.addWidget(self.heading, 0, 0)
        self.header_controls = QWidget()
        controls = QHBoxLayout(self.header_controls)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)
        controls.addStretch()
        self.calibrate_button = QPushButton('重新记录自然表情')
        self.calibrate_button.setToolTip(
            '每次开始会自动记住你放松时的表情，后续眉嘴动作与它比较。\n'
            '若开始时已皱眉，或换人、光线明显变化，请先放松，再点击此按钮。\n'
            '点击后保持自然表情约 1 秒，程序会重新记录面部参照。')
        self.calibrate_button.clicked.connect(self.calibrate_requested.emit)
        self.start_button = QPushButton('开始检测')
        self.start_button.setObjectName('primary')
        self.start_button.clicked.connect(self.start_requested.emit)
        self.stop_button = QPushButton('结束会话')
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.settings_button = QPushButton('设置')
        self.settings_button.setToolTip('选择摄像头、麦克风与表情参照')
        self.settings_button.clicked.connect(self._open_settings)
        for button in (self.settings_button, self.start_button, self.stop_button):
            controls.addWidget(button)
        top.addWidget(self.header_controls, 0, 1)
        top.setColumnStretch(0, 1)
        layout.addLayout(top)
        self.pages = QStackedWidget()
        layout.addWidget(self.pages, 1)
        outer.addWidget(main, 1)
        self._build_settings()
        self._build_live()
        self._build_review()
        self._build_mechanism()
        self._show_page(0)
        self.set_state({'status': 'ready'})
        self._adapt_layout()

    def _card(self, kicker, title):
        card = QFrame()
        card.setObjectName('card')
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        tag = label(kicker, 'kicker')
        tag.setVisible(bool(kicker))
        layout.addWidget(tag)
        layout.addWidget(label(title, 'cardTitle'))
        return card, layout

    def _build_settings(self):
        self.device_dialog = QDialog(self)
        self.device_dialog.setWindowTitle('设备与表情设置')
        self.device_dialog.setMinimumWidth(360)
        settings = QVBoxLayout(self.device_dialog)
        settings.setContentsMargins(22, 22, 22, 22)
        settings.setSpacing(14)
        settings.addWidget(label('设备与表情参照', 'introTitle'))
        self.camera_combo = QComboBox()
        self.camera_combo.addItem('摄像头 0', 0)
        self.microphone_combo = QComboBox()
        self.microphone_combo.addItem('系统默认麦克风', None)
        for title, combo in (('摄像头', self.camera_combo), ('麦克风', self.microphone_combo)):
            settings.addWidget(label(title, 'cardTitle'))
            settings.addWidget(combo)
        settings.addWidget(label('每次开始会自动记录自然表情。换人或光线变化后，可以在放松眉嘴时重新记录。', 'muted', True))
        self.baseline_status = label('等待开始', 'status', True)
        settings.addWidget(self.baseline_status)
        settings.addWidget(self.calibrate_button)
        done = QPushButton('完成')
        done.clicked.connect(self.device_dialog.close)
        settings.addWidget(done)

    def _open_settings(self):
        self.device_dialog.show()
        self.device_dialog.raise_()

    @staticmethod
    def _scroll(page):
        page.setObjectName('scrollPage')
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(page)
        return scroll

    def _build_live(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.live_layout = layout
        session = QHBoxLayout()
        self.session_status = label('●  等待开始', 'status')
        self.elapsed = label('00:00', 'telemetry')
        session.addWidget(self.session_status)
        session.addStretch()
        session.addWidget(self.elapsed)
        layout.addLayout(session)
        self.feedback_card = QFrame()
        self.feedback_card.setObjectName('feedback')
        feedback_layout = QGridLayout(self.feedback_card)
        feedback_layout.setContentsMargins(14, 8, 14, 8)
        feedback_layout.setSpacing(3)
        self.feedback_title = ElidedLabel('表达反馈', 'feedbackTitle')
        self.feedback_tags = ElidedLabel('', 'feedbackTags')
        self.feedback_tags.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.feedback_quote = ElidedLabel('', 'feedbackQuote')
        self.feedback_rewrite = ElidedLabel('开始后显示语言、表情与语气提醒。', 'feedbackAction')
        self.feedback_meta = label('', 'kicker', True)
        self.feedback_meta.hide()
        self.feedback_title.setFixedHeight(23)
        self.feedback_tags.setFixedHeight(23)
        self.feedback_rewrite.setFixedHeight(24)
        self.feedback_quote.setFixedHeight(20)
        for item in (self.feedback_tags, self.feedback_quote):
            policy = item.sizePolicy()
            policy.setRetainSizeWhenHidden(True)
            item.setSizePolicy(policy)
        feedback_layout.addWidget(self.feedback_title, 0, 0)
        feedback_layout.addWidget(self.feedback_tags, 0, 1)
        feedback_layout.addWidget(self.feedback_rewrite, 1, 0, 1, 2)
        feedback_layout.addWidget(self.feedback_quote, 2, 0, 1, 2)
        feedback_layout.setColumnStretch(0, 3)
        feedback_layout.setColumnStretch(1, 2)
        self.feedback_card.setFixedHeight(92)
        layout.addWidget(self.feedback_card)
        center = self.live_grid = QGridLayout()
        center.setSpacing(12)
        camera, camera_layout = self._card('', '本机摄像头')
        self.camera_card = camera
        camera_layout.setSpacing(7)
        camera.setFixedWidth(228)
        self.camera = CameraPreview()
        camera_layout.addWidget(self.camera, 1)
        self.camera_status = ElidedLabel('等待摄像头信号', 'status')
        self.camera_status.setFixedHeight(20)
        camera_layout.addWidget(self.camera_status)
        self.face_metrics = {}
        for key, text in (('brow_tension', '眉部紧张'), ('mouth_downturn', '嘴角下压')):
            metric = Metric(text)
            metric.setToolTip('归一化面部动作强度 0–100，不是情绪概率或识别准确率。')
            self.face_metrics[key] = metric
            camera_layout.addWidget(metric)
        center.addWidget(camera, 0, 0)
        self.tree = LivingTree()
        self.tree.setMinimumSize(120, 160)
        center.addWidget(self.tree, 0, 1)
        center.setColumnStretch(1, 1)
        scores, score_layout = self._card('', '当前交流状态')
        self.score_card = scores
        scores.setFixedWidth(228)
        self.score_body = QBoxLayout(QBoxLayout.Direction.TopToBottom)
        self.score_body.setSpacing(10)
        self.ring = ScoreRing()
        self.score_body.addWidget(self.ring)
        details = QWidget()
        detail_layout = QVBoxLayout(details)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(12)
        self.metrics = {}
        for key, text in (('visual_tension', '面部动作'), ('tone_pressure', '语音压力'), ('semantic_bias', '语义倾向')):
            metric = Metric(text)
            self.metrics[key] = metric
            detail_layout.addWidget(metric)
        detail_layout.addStretch(1)
        self.focus_card = QFrame()
        self.focus_card.setObjectName('sessionFocus')
        self.focus_card.setMinimumWidth(0)
        focus_layout = QVBoxLayout(self.focus_card)
        focus_layout.setContentsMargins(10, 8, 10, 8)
        focus_layout.setSpacing(5)
        self.focus_title = ElidedLabel('本场重点', 'focusTitle')
        self.focus_title.setFixedHeight(20)
        focus_layout.addWidget(self.focus_title)
        self.focus_note = ElidedLabel('开始后汇总本场表现', 'muted')
        self.focus_note.setFixedHeight(18)
        focus_layout.addWidget(self.focus_note)
        self.focus_labels = []
        for _ in range(3):
            hint = ElidedLabel('', 'focusHint')
            hint.setFixedHeight(28)
            hint.hide()
            self.focus_labels.append(hint)
            focus_layout.addWidget(hint)
        self.focus_card.setStyleSheet('QFrame#sessionFocus {background:#102b32; border:1px solid #29505a; border-radius:12px;} '
                                     'QLabel#focusTitle {color:#b7e2e9; font-size:14px; font-weight:600;}')
        detail_layout.addWidget(self.focus_card)
        self.score_body.addWidget(details, 1)
        score_layout.addLayout(self.score_body, 1)
        center.addWidget(scores, 0, 2)
        layout.addLayout(center, 1)
        voice, voice_layout = self._card('', '声音输入与实时文字')
        self.voice_card = voice
        voice_row = QHBoxLayout()
        mic_column = QVBoxLayout()
        self.audio_status = ElidedLabel('麦克风待机', 'status')
        self.audio_status.setFixedHeight(18)
        self.mic_level = Metric('输入电平')
        mic_column.addWidget(self.audio_status)
        mic_column.addWidget(self.mic_level)
        voice_row.addLayout(mic_column, 1)
        transcript_column = QVBoxLayout()
        self.transcript_status = ElidedLabel('识别器待机', 'kicker')
        self.transcript_status.setFixedHeight(16)
        self.transcript = ElidedLabel('等待真实语音输入…', 'cardTitle')
        self.transcript.setFixedHeight(24)
        transcript_column.addWidget(self.transcript_status)
        transcript_column.addWidget(self.transcript)
        voice_row.addLayout(transcript_column, 2)
        voice_layout.addLayout(voice_row)
        layout.addWidget(voice)
        self.live_scroll = self._scroll(page)
        self.pages.addWidget(self.live_scroll)

    def _build_review(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        history_card, history_layout = self._card('', '本机会话历史')
        history_row = QGridLayout()
        self.history_combo = QComboBox()
        self.history_combo.setMinimumWidth(0)
        self.history_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.history_combo.addItem('当前会话', None)
        self.history_combo.currentIndexChanged.connect(self._history_changed)
        self.history_refresh = QPushButton('刷新')
        self.history_refresh.clicked.connect(self.history_refresh_requested.emit)
        self.history_more = QPushButton('更多记录')
        self.history_more.clicked.connect(self.history_more_requested.emit)
        self.history_more.hide()
        self.history_delete = QPushButton('删除本条')
        self.history_delete.setEnabled(False)
        self.history_delete.clicked.connect(self._request_history_delete)
        history_row.addWidget(self.history_combo, 0, 0, 1, 3)
        history_row.addWidget(self.history_refresh, 0, 3)
        history_row.setColumnStretch(0, 1)
        self.history_status = label('结束后点击“保存到历史”，下次打开仍可查看。', 'muted', True)
        history_row.addWidget(self.history_status, 1, 0)
        self.history_retry = QPushButton('重试保存')
        self.history_retry.clicked.connect(self.history_retry_requested.emit)
        self.history_retry.hide()
        history_row.addWidget(self.history_retry, 1, 1)
        history_row.addWidget(self.history_more, 1, 2)
        history_row.addWidget(self.history_delete, 1, 3)
        history_layout.addLayout(history_row)
        layout.addWidget(history_card)
        row = QHBoxLayout()
        self.review_status = label('每一次回顾，都从真实记录出发。', 'muted', True)
        row.addWidget(self.review_status, 1)
        self.history_save = QPushButton('保存到历史')
        self.history_save.setEnabled(False)
        self.history_save.clicked.connect(self.history_save_requested.emit)
        row.addWidget(self.history_save)
        self.export_button = QPushButton('导出会话报告')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_requested.emit)
        row.addWidget(self.export_button)
        layout.addLayout(row)
        stats = QHBoxLayout()
        stats.setSpacing(12)
        self.review_stats = {}
        for key, kicker, title in (('duration', '', '会话时长'), ('average', '', '平均 SBI'), ('peak', '', '峰值 SBI')):
            card, contents = self._card(kicker, title)
            value = label('—', 'statValue')
            contents.addWidget(value)
            self.review_stats[key] = value
            stats.addWidget(card, 1)
        layout.addLayout(stats)
        chart, contents = self._card('', '表达变化 · SBI 时间线')
        legend = QHBoxLayout()
        for title, color in (('●  平稳', SessionTimeline.MINT), ('●  留意', '#edc678'), ('●  高风险', SessionTimeline.CORAL)):
            item = label(title)
            item.setStyleSheet('color:' + color + ';')
            legend.addWidget(item)
        legend.addStretch()
        legend.addWidget(label('时间 →   ·   悬停查看记录', 'muted'))
        contents.addLayout(legend)
        self.timeline = SessionTimeline()
        self.timeline.setFixedHeight(300)
        contents.addWidget(self.timeline)
        contents.addWidget(label('横轴：会话时间　纵轴：SBI（0–100）　空白间隔表示缺少有效信号。', 'muted', True))
        layout.addWidget(chart)
        layout.addWidget(label('下一次，可以这样表达', 'cardTitle'))
        self.advice_grid = QGridLayout()
        self.advice_grid.setSpacing(12)
        self.advice_cards = []
        for index, (kicker, title) in enumerate((('', '语言表达'), ('', '表情动作'), ('', '声音与节奏'))):
            card, contents = self._card(kicker, title)
            body = label('完成会话后，根据实际观察给出建议。', 'reviewNote', True)
            contents.addWidget(body)
            contents.addStretch()
            self.advice_cards.append((card, contents.itemAt(1).widget(), body))
            self.advice_grid.addWidget(card, 0, index)
        layout.addLayout(self.advice_grid)
        moments, contents = self._card('', '值得回看的时刻')
        self.moments_layout = QVBoxLayout()
        self.moments_layout.setSpacing(9)
        contents.addLayout(self.moments_layout)
        self.moments_layout.addWidget(label('完成会话后，关键变化会出现在这里。', 'muted', True))
        layout.addWidget(moments)
        self.transcripts_toggle = QPushButton('展开语音转录')
        self.transcripts_toggle.setCheckable(True)
        self.transcripts_toggle.setVisible(False)
        self.transcripts_toggle.toggled.connect(self._toggle_transcripts)
        layout.addWidget(self.transcripts_toggle)
        self.review = QTextBrowser()
        self.review.setOpenExternalLinks(False)
        self.review.setMinimumHeight(180)
        self.review.setMaximumHeight(300)
        self.review.hide()
        layout.addWidget(self.review)
        self.review_footnote = label('本页仅呈现真实会话记录。SBI 是可解释的原型规则评分，不代表人格、意图或识别准确率。', 'muted', True)
        layout.addWidget(self.review_footnote)
        self.review_scroll = self._scroll(page)
        self.pages.addWidget(self.review_scroll)

    def _toggle_transcripts(self, expanded):
        self.review.setVisible(expanded)
        count = len(self._report.get('transcripts') or [])
        self.transcripts_toggle.setText(('收起' if expanded else '展开') + f'语音转录 · {count} 条')

    @staticmethod
    def _history_title(item):
        stamp = str(item.get('started_at') or '')
        try:
            stamp = datetime.fromisoformat(stamp).astimezone().strftime('%m月%d日 %H:%M:%S')
        except ValueError:
            pass
        state = {'running': '进行中', 'completed': '已结束', 'interrupted': '中断',
                 'unsaved': '未保存 · 可导出'}.get(item.get('status'), '未知')
        return f"{stamp} · {time_label(numeric(item.get('elapsed_seconds')) or 0)} · 均值 {fmt(item.get('average_sbi'))} / 峰值 {fmt(item.get('peak_sbi'))} · {state}"

    def set_history_items(self, items, selected_id, *, has_more=False):
        retained = next((row for row in self._history_items if row['id'] == selected_id), None)
        self._history_items = list(items)
        if retained and not any(row['id'] == selected_id for row in items):
            # Anchor the visible selection separately from the controller's
            # paged rows; it must not count toward the next database offset.
            self._history_items.append(retained)
        self.history_combo.blockSignals(True)
        self.history_combo.clear()
        self.history_combo.addItem('当前会话', None)
        for item in self._history_items:
            self.history_combo.addItem(self._history_title(item), item['id'])
        self.history_combo.blockSignals(False)
        self.set_history_selection(selected_id)
        self.history_more.setVisible(has_more)

    def set_history_selection(self, session_id):
        self._history_selection = session_id
        self.set_history_save_available(self._save_available)
        self.history_combo.blockSignals(True)
        index = self.history_combo.findData(session_id)
        self.history_combo.setCurrentIndex(max(0, index))
        self.history_combo.blockSignals(False)
        item = next((row for row in self._history_items if row['id'] == session_id), None)
        self.history_delete.setEnabled(bool(item and item.get('status') in ('completed', 'interrupted')))

    def _history_changed(self, index):
        key = self.history_combo.itemData(index)
        self.set_history_selection(key)
        self.set_report({})
        self.history_selected.emit(key)

    def _request_history_delete(self):
        item = next((row for row in self._history_items if row['id'] == self._history_selection), None)
        if not item or item.get('status') not in ('completed', 'interrupted'):
            return
        answer = QMessageBox.question(self, '删除本条历史？', self._history_title(item) +
            '\n\n仅删除本条本机历史，不删除已导出的报告。\n删除后无法在应用内撤销。',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.history_delete_requested.emit(item['id'])

    def set_history_status(self, text, *, error=False):
        self.history_status.setText(text)
        self.history_status.setStyleSheet('color:' + ('#ff8197' if error else '#86b7b6') + ';')
        self.history_retry.setVisible(error)

    def set_history_save_available(self, available):
        self._save_available = bool(available)
        self.history_save.setEnabled(self._save_available and self._history_selection is None)

    def displayed_report(self):
        return deepcopy(self._report)

    def _build_mechanism(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        documents = QHBoxLayout()
        help_button = QPushButton('使用说明')
        help_button.clicked.connect(lambda: self._open_document('docs/desktop-quickstart.md', '使用说明'))
        self.notice_button = QPushButton('第三方软件与模型声明')
        self.notice_button.clicked.connect(lambda: self._open_document('resources/licenses/THIRD-PARTY-NOTICES.txt', '第三方软件与模型声明'))
        documents.addWidget(help_button)
        documents.addWidget(self.notice_button)
        documents.addStretch()
        layout.addLayout(documents)
        hero, contents = self._card('', '让细微的表达变化，被温和地看见')
        hero.layout().itemAt(1).widget().setObjectName('introTitle')
        contents.addWidget(label('面部动作、声音与实际语音文字汇聚为同一条时间线。交流时看生命树与简短提醒，结束后回看变化，练习更具体、更尊重的表达。', 'introBody', True))
        layout.addWidget(hero)
        self.intro_grid = QGridLayout()
        self.intro_grid.setSpacing(14)
        cards = (
            ('01 · 分数含义', 'SBI 是什么？', '#8ae4c4',
             '<p><b>交流偏差信号指数</b>将可观察的表达线索映射为 0–100 分。数值越高，越值得停顿并检查当前表达。</p>'
             '<p><font color="#78e2b9"><b>0–24　平稳交流</b></font></p>'
             '<p><font color="#edc678"><b>25–59　留意变化</b></font></p>'
             '<p><font color="#ff8197"><b>60–100　先停顿，再澄清</b></font></p>'
             '<p>友好度是同一结果的反向刻度。缺少有效输入时显示「—」，不把缺失信号当作友好。</p>',
             (('友好度 = 100 − SBI', '友好度与 SBI 是互补刻度'),)),
            ('02 · 多模态观察', '系统看到了什么？', '#89c9f0',
             '<p><font color="#8ae4c4"><b>面部动作</b></font>　眉部紧张与嘴角下压相对本人自然表情的变化。正常说话中的抿唇、噘嘴不参与负面评分。</p>'
             '<p><font color="#edc678"><b>语言表达</b></font>　实际转录中的群体概括、能力贬低、排斥等表达，结合否定语境给出标签与改写。</p>'
             '<p><font color="#b9b0ef"><b>声音与语气</b></font>　相对声音基线的音量升高，作为辅助线索。</p>'
             '<p>单一表情或大音量不能证明冒犯；需结合实际措辞、语境与对方反馈。</p>', ()),
            ('03 · 个体化参照', '表情强度如何计算？', '#b9b0ef',
             '<p>开始后记录 8 帧自然表情。用基线中位数 <i>m</i>、波动 <i>d</i> 与当前动作系数 <i>x</i> 计算相对变化，抑制微小抖动。</p>'
             '<p><i>h</i> 为动作响应幅度：眉部与嘴角 0.18。左右侧分别比较后取较强值，显示为 <b>100 × f</b>。</p>'
             '<p>动作强度达到 0.18 且保持至少 250 毫秒后，再结合稳定性形成持续线索。</p>',
             (('g = max(0.006, 4.4478d)', '噪声门限 g，由自然表情波动确定'),
              ('z = max(0, x − m − g) / max(h, 6g)', '相对基线与噪声门限归一化'),
              ('f = min(1, z<sup>0.75</sup>)', '动作强度限制在零至一之间'))),
            ('04 · 可解释融合', '从信号到 SBI', '#edc678',
             '<p>以 <i>V</i>、<i>S</i>、<i>A</i> 表示 0–1 的面部、语言与声音线索。较强的主线索与跨模态协同项共同形成目标。</p>'
             '<p>分数采用非对称时间平滑：上升响应 <b>0.18 秒</b>，回落 <b>1.40 秒</b>。过期信号退出计算，微笑不抵消语言偏见。</p>',
             (('u = max(0.65V, 0.90S)', '主线索 u'),
              ('T = min(1, u + 0.12 min(V, S) + 0.10A)', '融合目标 T：主线索、协同与声音辅助'),
              ('r<sub>t</sub> = r<sub>t−1</sub> + (T − r<sub>t−1</sub>)(1 − e<sup>−Δt/τ</sup>)', '随时间跟随融合目标，快升慢降'),
              ('SBI = 100r<sub>t</sub>', '映射为零至一百分'))),
            ('05 · 语言练习', '把概括换成具体表达', '#ff9bac',
             '<p><font color="#ff9bac"><b>“女生不适合……”</b></font><br>→ 根据兴趣、经历与任务要求讨论。</p>'
             '<p><font color="#ff9bac"><b>“学历低的人更适合跑腿”</b></font><br>→ 按经验、能力与本人意愿分工。</p>'
             '<p><font color="#ff9bac"><b>“别这么敏感”</b></font><br>→ 我想理解，哪部分让你不舒服？</p>'
             '<p><b>300 条原创风险样例 + 60 条对照</b>，覆盖 12 类场景。样例匹配与语境规则参与语义评分；不是采集的真实会议数据或已验证的模型效果。</p>'
             '<p>分类参考、来源与样例库内置于程序；开发工程中可阅读相应语料文件。误听、引用、反驳等情况仍需核对原话。</p>', ()),
            ('06 · 使用方式', '让每次反馈更有帮助', '#8ae4c4',
             '<p><b>开始前</b>　面向镜头、放松眉嘴，确认麦克风。换人或光线明显变化后可在“设置”中重新记录参照。</p>'
             '<p><b>交流时</b>　看生命树和简短提醒，描述具体事实，保留停顿，也听取对方的实际反馈。</p>'
             '<p><b>结束后</b>　回看渐变曲线与少量关键时刻，对照真实文字，挑一个具体说法练习改写。</p>'
             '<p><font color="#89a5b2">这是反思表达的原型工具。SBI 是规则评分，不是识别准确率，也不用于判断人格、意图或诊断情绪。</font></p>', ()),
        )
        self.intro_cards = []
        for index, (kicker, title, color, body, formulas) in enumerate(cards):
            card, contents = self._card(kicker, title)
            heading = contents.itemAt(1).widget()
            heading.setObjectName('introTitle')
            heading.setStyleSheet('color:' + color + ';')
            text = label(body, 'introBody', True)
            text.setTextFormat(Qt.TextFormat.RichText)
            text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            contents.addWidget(text)
            for equation, caption in formulas:
                contents.addWidget(MathFormula(equation, caption, color, height=54))
            contents.addStretch()
            self.intro_cards.append(card)
            self.intro_grid.addWidget(card, index // 2, index % 2)
        layout.addLayout(self.intro_grid)
        self.mechanism = self._scroll(page)
        self.pages.addWidget(self.mechanism)

    def _open_document(self, relative, title):
        from .resources import resource_root
        self.document_dialog = QDialog(self)
        self.document_dialog.setWindowTitle(title)
        self.document_dialog.resize(min(850, self.width()), min(600, self.height()))
        contents = QVBoxLayout(self.document_dialog)
        self.document_browser = QTextBrowser()
        self.document_browser.setOpenExternalLinks(False)
        try:
            self.document_browser.setPlainText((resource_root() / relative).read_text(encoding='utf-8'))
        except OSError as exc:
            self.document_browser.setPlainText('无法读取内置声明：' + str(exc))
        contents.addWidget(self.document_browser)
        done = QPushButton('关闭')
        done.clicked.connect(self.document_dialog.close)
        contents.addWidget(done)
        self.document_dialog.show()

    @staticmethod
    def _document(body):
        return '<html><head><style>body{font-family:"Microsoft YaHei UI";font-size:12px;color:#b8d1d9}h2{color:#def2eb;font-size:22px;font-weight:500}h3{color:#73dcbc;font-size:15px;margin-top:26px}p{line-height:1.7}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #25414b;text-align:left}th{color:#73dcbc}small{color:#7394a1}</style></head><body>' + body + '</body></html>'

    def _show_page(self, index):
        self.pages.setCurrentIndex(index)
        for i, button in enumerate(self.nav):
            button.setChecked(i == index)
        self.page_title.setText(('让每一次表达，温和而清晰', '回顾这一次真实交流', '理解每一条反馈的来源')[index])
        self.page_kicker.setText(('实时交互自检', '会话回顾', '系统介绍')[index])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'intro_cards'):
            self._adapt_layout()

    def _adapt_layout(self):
        compact = self.width() < 1120
        tiny = self.width() < 900 or self.height() < 540
        # The focus block shares the existing card's vertical budget. Switch
        # to compact spacing earlier rather than growing the live page.
        short = self.height() < 800
        very_short = self.height() < 560
        extra_short = self.height() < 470
        self.sidebar.setFixedWidth(112 if tiny else 148 if compact else 176)
        self.sidebar.layout().setContentsMargins(10 if tiny else 18, 20, 10, 16)
        self.brand_label.setStyleSheet('font-size: 16px;' if tiny else 'font-size: 20px;' if compact else '')
        for button in self.nav:
            button.setStyleSheet('padding: 12px 4px; font-size: 12px;' if tiny else '')
        margin = 12 if short or compact else 24
        self.main_layout.setContentsMargins(margin, 8 if very_short else 10 if short else 20,
                                           margin, 8 if very_short else 10 if short else 16)
        self.main_layout.setSpacing(6 if very_short else 8 if short else 12)
        self.header_layout.setHorizontalSpacing(8)
        self.header_layout.setColumnStretch(1, 0)
        # Button style/text changes must not alter the page's vertical budget.
        header_height = 23 if very_short else 28 if short else 44
        self.heading.setFixedHeight(header_height)
        self.header_controls.setFixedHeight(header_height)
        self.page_title.setStyleSheet('font-size: 15px;' if very_short else 'font-size: 17px;' if tiny else 'font-size: 21px;' if compact else '')
        for button in (self.settings_button, self.start_button, self.stop_button):
            button.setStyleSheet('padding: 5px 8px; font-size: 11px;' if very_short else 'padding: 7px 10px; font-size: 12px;' if compact or short else '')
        width = 164 if tiny else 195 if compact else 228
        self.camera_card.setFixedWidth(width)
        self.score_card.setFixedWidth(width)
        self.live_grid.setSpacing(8 if compact else 12)
        self.live_layout.setSpacing(5 if very_short else 6 if short else 10)
        self.feedback_card.layout().setContentsMargins(14, 5 if very_short else 8, 14, 5 if very_short else 8)
        self.feedback_card.setFixedHeight(74 if very_short else 80 if tiny else 92)
        self.feedback_title.setFixedHeight(20 if tiny else 23)
        self.feedback_tags.setFixedHeight(20 if tiny else 23)
        self.feedback_rewrite.setFixedHeight(21 if tiny else 24)
        self.feedback_quote.setFixedHeight(17 if tiny else 20)
        self.feedback_title.setStyleSheet('font-size:15px;' if tiny else '')
        self.feedback_rewrite.setStyleSheet('font-size:14px;' if tiny else '')
        self.voice_card.setFixedHeight(78 if very_short else 82 if short else 98)
        for card in (self.camera_card, self.score_card, self.voice_card):
            card.layout().setContentsMargins(9 if short else 14, 6 if very_short else 8 if short else 14,
                                            9 if short else 14, 6 if very_short else 8 if short else 14)
            card.layout().setSpacing(3 if very_short else 5 if short else 9)
            card.setSizePolicy(QSizePolicy.Policy.Fixed if card != self.voice_card else QSizePolicy.Policy.Expanding,
                               QSizePolicy.Policy.Expanding if card != self.voice_card else QSizePolicy.Policy.Fixed)
        self.camera.setMinimumSize(100, 40 if very_short else 55 if short else 100)
        self.camera_status.setFixedHeight(16 if very_short else 20)
        self.tree.setMinimumHeight(125 if very_short else 160)
        self.score_body.setSpacing(2 if very_short else 3 if short else 10)
        self.ring.setFixedHeight(48 if extra_short else 64 if very_short else 100 if short else 175)
        for metric in (*self.face_metrics.values(), *self.metrics.values()):
            metric.setFixedHeight(16 if very_short else 18 if short else 25)
            metric.layout().setSpacing(4 if tiny else 7)
        self.score_body.itemAt(1).widget().layout().setSpacing(4 if very_short else 6 if short else 12)
        self.focus_card.layout().setContentsMargins(6 if short else 10, 4 if short else 8,
                                                    6 if short else 10, 4 if short else 8)
        self.focus_card.layout().setSpacing(2 if short else 5)
        self.focus_title.setFixedHeight(14 if very_short else 16 if short else 20)
        self.focus_note.setFixedHeight(14 if short else 18)
        # Reserve a stable block even while collecting or displaying fewer
        # priorities. Extremely short work areas show the highest priority;
        # all retained items remain available on hover.
        self.focus_card.setFixedHeight(44 if extra_short else 86 if very_short else 100 if short else 140)
        for hint in self.focus_labels:
            hint.setFixedHeight(18 if very_short else 22 if short else 28)
        self._paint_focus(self._snapshot)
        advice_count = len(self._report.get('advice') or []) or 3
        advice_columns = 1 if compact else (2 if advice_count > 3 else min(3, advice_count))
        for index, (card, _, _) in enumerate(self.advice_cards):
            self.advice_grid.removeWidget(card)
            self.advice_grid.addWidget(card, index // advice_columns, index % advice_columns)
        for index in range(3):
            self.advice_grid.setColumnStretch(index, 1 if index < advice_columns else 0)
        for index, card in enumerate(self.intro_cards):
            self.intro_grid.removeWidget(card)
            self.intro_grid.addWidget(card, index if compact else index // 2, 0 if compact else index % 2)
        self.intro_grid.setColumnStretch(0, 1)
        self.intro_grid.setColumnStretch(1, 0 if compact else 1)
        # Resolve the nested minimum heights bottom-up. QScrollArea otherwise
        # temporarily retains the previous screen size after several resizes,
        # displaying a stale scrollbar even though the compact content fits.
        for widget in (self.focus_card, self.score_body.itemAt(1).widget(),
                       self.camera_card, self.score_card, self.voice_card):
            widget.layout().invalidate()
            widget.layout().activate()
        self.live_grid.invalidate()
        self.live_layout.invalidate()
        self.live_layout.activate()
        page = self.live_scroll.widget()
        page.setMinimumHeight(self.live_layout.minimumSize().height())
        page.resize(self.live_scroll.viewport().size())

    def set_devices(self, cameras, microphones):
        for combo, devices in ((self.camera_combo, cameras), (self.microphone_combo, microphones)):
            previous = combo.currentData()
            combo.clear()
            for name, device_id in devices:
                name = str(name)
                display = name if not any('a' <= c.lower() <= 'z' for c in name) else ('麦克风 ' if combo is self.microphone_combo else '摄像头 ') + str(device_id)
                combo.addItem(display, device_id)
                combo.setItemData(combo.count()-1, name, Qt.ItemDataRole.ToolTipRole)
            position = combo.findData(previous)
            if position >= 0:
                combo.setCurrentIndex(position)

    def selected_camera(self):
        return self.camera_combo.currentData()

    def selected_microphone(self):
        return self.microphone_combo.currentData()

    def set_frame(self, frame_bgr, box=None):
        self.camera.set_frame(frame_bgr, box)

    def _paint_focus(self, snapshot):
        focus = snapshot.get('session_focus') or {}
        items = focus.get('items') or []
        status = snapshot.get('status', 'ready')
        self.focus_note.setText(str(focus.get('status') or
            ('正在积累有效信号' if status in ('running', 'starting', 'calibrating') else '开始后汇总本场表现')))
        self.focus_note.setVisible(not items)
        colors = {'amber':'#f1d08a', 'blue':'#9cdbea', 'mint':'#8de4c0'}
        limit = 1 if self.height() < 470 else 3
        for index, hint in enumerate(self.focus_labels):
            item = items[index] if index < len(items) else {}
            hint.setText(str(item.get('text') or ''))
            hint.setToolTip(str(item.get('detail') or ''))
            size = 11 if self.height() < 560 else 12 if self.height() < 800 else 15
            hint.setStyleSheet(f"color:{colors.get(item.get('tone'), colors['blue'])}; font-size:{size}px; font-weight:700;")
            hint.setVisible(bool(item) and index < limit)
        self.focus_card.setToolTip('本场截至目前的累计重点，每 10 秒重新评估。结束后保留，下一场开始清空；与顶部即时反馈相互独立。'
                                  + ''.join('\n' + str(item.get('text') or '') + '：' + str(item.get('detail') or '') for item in items))

    def set_state(self, snapshot):
        snapshot = snapshot or {}
        self._snapshot = snapshot
        self._paint_focus(snapshot)
        state = str(self._snapshot.get('status', 'ready'))
        running = state in ('running', 'starting', 'calibrating')
        status = {'running': '●  检测运行中', 'starting': '●  正在启动', 'stopped': '●  会话已结束',
                  'ready': '●  等待开始', 'error': '●  检测遇到错误', 'calibrating': '●  校准中'}.get(state, state)
        self.session_status.setText(status)
        self.sidebar_status.setText(status)
        self.start_button.setEnabled(not running)
        self.start_button.setText('检测进行中' if running else '开始检测')
        self.stop_button.setEnabled(running)
        self.camera_combo.setEnabled(not running)
        self.microphone_combo.setEnabled(not running)
        elapsed = max(0, (numeric(snapshot.get('elapsed_ms')) or 0) / 1000)
        self.elapsed.setText(f'{int(elapsed) // 60:02}:{int(elapsed) % 60:02}')
        self.tree.set_state(snapshot.get('tree'), snapshot.get('sbi') if running else None)
        self.ring.set_value(snapshot.get('friendliness') if running else None)
        metrics = snapshot.get('metrics') or {}
        for key, widget in self.metrics.items():
            widget.set_value(metrics.get(key) if running else None)
        alert = snapshot.get('language_alert') if running else None
        tone = 'neutral'
        self.feedback_title.setText(str(snapshot.get('suggestion_title') or '表达反馈'))
        self.feedback_rewrite.setText(str(snapshot.get('suggestion') or '开始后显示语言、表情与语气提醒。'))
        if alert:
            tone = 'pending' if alert.get('review_required') or not alert.get('is_final') else 'language'
            self.feedback_tags.setText('  ·  '.join(map(str, alert.get('tags') or [])))
            quote = str(alert.get('text') or '')
            self.feedback_quote.setText('原句：' + quote)
            self.feedback_quote.setToolTip(quote)
            self.feedback_rewrite.setText('建议：' + str(alert.get('rewrite') or '请结合原话与语境核对。'))
            note = ('近似转写，请核对字幕 · 不计入 SBI' if alert.get('review_required') else
                    '刚才的语言提醒 · 已确认文字' if alert.get('is_final') else
                    '临时转写 · 待确认，字幕仍可能修改')
            self.feedback_meta.setText(note)
            detail = str(alert.get('explanation') or '') + '\n建议改写：' + str(alert.get('full_rewrite') or '')
            self.feedback_card.setToolTip(note + '\n' + detail)
        else:
            self.feedback_tags.clear()
            self.feedback_quote.clear()
            self.feedback_meta.clear()
            self.feedback_card.setToolTip('表情、声音和规则匹配只是线索，请结合实际语境理解。')
        for item in (self.feedback_tags, self.feedback_quote):
            item.setVisible(bool(alert))
        if self.feedback_card.property('tone') != tone:
            self.feedback_card.setProperty('tone', tone)
            self.feedback_card.style().unpolish(self.feedback_card)
            self.feedback_card.style().polish(self.feedback_card)
            self.feedback_title.style().unpolish(self.feedback_title)
            self.feedback_title.style().polish(self.feedback_title)
        explanation = snapshot.get('explanation')
        if isinstance(explanation, (list, tuple)):
            explanation = '\n'.join(map(str, explanation))
        self.score_card.setToolTip(str(explanation or '融合结果取决于实际可用信号。'))
        vision = snapshot.get('vision') or {}
        features = vision.get('features') or {}
        found = bool(vision.get('face_detected'))
        self.camera_status.setText('●  已检测到面部' if running and found else '○  未检测到面部' if running else '等待开始')
        self.camera.box = vision.get('box') if found else None
        if not running:
            self.camera.set_frame(None)
        self.camera.placeholder = '正在读取摄像头' if running else '摄像头尚未启动'
        self.camera.update()
        for key, metric in self.face_metrics.items():
            metric.set_value(features.get(key) if running and found else None)
        self.calibrate_button.setEnabled(running and found and bool(features.get('baseline_ready')))
        if not running:
            baseline = '开始后自动记住自然表情'
        elif features.get('baseline_ready'):
            baseline = '自然表情已记录 · 动作与此比较'
        elif found:
            baseline = '请放松约 1 秒 · 正在记住自然表情'
        else:
            baseline = '面向镜头后自动记录自然表情'
        self.baseline_status.setText(baseline)
        audio = snapshot.get('audio') or {}
        self.audio_status.setText('麦克风 · ' + str(audio.get('status') or '待机'))
        self.mic_level.set_value(audio.get('rms') if running else None)
        self.mic_level.setToolTip('输入均方根：' + fmt(audio.get('rms'), decimals=4) + ' / 峰值：' + fmt(audio.get('peak'), decimals=4))
        transcript = snapshot.get('transcript') or {}
        self.transcript.setText(str(transcript.get('text') or '等待真实语音输入…'))
        text_status = str(transcript.get('status') or '待机')
        self.transcript_status.setText('已确认文字' if transcript.get('is_final') else '正在识别 · ' + text_status if running else '语音识别待机')
        telemetry = snapshot.get('telemetry') or {}
        self.settings_button.setToolTip('设备与表情设置\n摄像头 ' + fmt(telemetry.get('camera_fps')) + ' 帧/秒\n面部处理 ' + fmt(telemetry.get('vision_latency_ms'), decimals=0) + ' 毫秒\n音频队列 ' + fmt(telemetry.get('audio_backlog_ms'), decimals=0) + ' 毫秒\n语音识别 ' + fmt(telemetry.get('asr_latency_ms'), decimals=0) + ' 毫秒')

    @staticmethod
    def _table(headers, rows):
        esc = lambda value: html.escape(str(value))
        return '<table width="100%" cellspacing="0"><tr>' + ''.join('<th>' + esc(v) + '</th>' for v in headers) + '</tr>' + ''.join('<tr>' + ''.join('<td>' + esc(v) + '</td>' for v in row) + '</tr>' for row in rows) + '</table>'

    def set_report(self, report):
        report = report or {}
        self._report = report
        self.export_button.setEnabled(bool(report))
        elapsed = numeric(report.get('elapsed_seconds'))
        self.review_stats['duration'].setText('—' if elapsed is None else time_label(elapsed))
        self.review_stats['average'].setText(fmt(report.get('average_sbi')))
        self.review_stats['average'].setToolTip('所有有效界面更新样本的均值；缺失信号不计入。')
        self.review_stats['peak'].setText(fmt(report.get('peak_sbi')))
        self.review_stats['peak'].setToolTip('覆盖全部有效更新的峰值，包含历史抽样间的短暂变化。')
        self.timeline.set_history(report.get('history') or [], elapsed)
        advice = report.get('advice') or []
        while len(self.advice_cards) < len(advice):
            card, contents = self._card('', '本次建议')
            body = label('', 'reviewNote', True)
            contents.addWidget(body)
            contents.addStretch()
            self.advice_cards.append((card, contents.itemAt(1).widget(), body))
            self.advice_grid.addWidget(card, len(self.advice_cards) - 1, 0)
        for index, (card, title, body) in enumerate(self.advice_cards):
            item = advice[index] if index < len(advice) else None
            card.setVisible(bool(item) or (not report and index < 3))
            if item:
                title.setText(str(item.get('title') or '本次建议'))
                body.setText(str(item.get('body') or ''))
            else:
                title.setText(('语言表达', '表情动作', '声音与节奏')[index % 3])
                body.setText('完成会话后，根据实际观察给出建议。')
        while self.moments_layout.count():
            item = self.moments_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        moments = (report.get('moments') or [])[:3]
        for item in moments:
            card = QFrame()
            card.setObjectName('moment')
            row = QHBoxLayout(card)
            row.setContentsMargins(14, 11, 14, 11)
            at = time_label((numeric(item.get('at_ms')) or 0) / 1000)
            moment_time = label(at, 'momentTime')
            moment_time.setFixedWidth(54)
            row.addWidget(moment_time, 0, Qt.AlignmentFlag.AlignTop)
            text = QVBoxLayout()
            text.setSpacing(5)
            text.addWidget(label(str(item.get('title') or '表达变化'), 'cardTitle', True))
            text.addWidget(label(str(item.get('detail') or ''), 'reviewNote', True))
            if item.get('end_ms', 0) > item.get('start_ms', 0):
                interval = time_label(item['start_ms']/1000) + '—' + time_label(item['end_ms']/1000) + ' · 同类变化合并'
                text.addWidget(label(interval, 'kicker'))
            row.addLayout(text, 1)
            self.moments_layout.addWidget(card)
        if not moments:
            self.moments_layout.addWidget(label('本次没有可回看的关键变化。' if report else '完成会话后，关键变化会出现在这里。', 'muted', True))
        transcripts = report.get('transcripts') or []
        transcript_body = self._table(('时间', '文字'), [(time_label((numeric(row.get('at_ms')) or 0) / 1000), row.get('text', '')) for row in transcripts]) if transcripts else '<p>没有转录记录。</p>'
        self.review.setHtml(self._document(transcript_body))
        self.transcripts_toggle.setVisible(bool(transcripts))
        expanded = bool(transcripts) and self.transcripts_toggle.isChecked()
        self.transcripts_toggle.setChecked(expanded)
        self._toggle_transcripts(expanded)
        self.review_status.setText(f'本次会话 · {len(transcripts)} 条语音记录 · {len(moments)} 个回看时刻' if report else '每一次回顾，都从真实记录出发。')
        self.review_footnote.setText(str(report.get('limitations') or '本页仅呈现真实会话记录。SBI 是可解释的原型规则评分，不代表人格、意图或识别准确率。'))
        self._adapt_layout()

    def closeEvent(self, event):
        if self._deferred_close:
            event.ignore()
            if not self._close_notified:
                self._close_notified = True
                self.closing.emit()
            return
        self.tree._timer.stop()
        if not self._close_notified:
            self.closing.emit()
        super().closeEvent(event)

    def set_deferred_close(self, enabled):
        self._deferred_close = enabled
        self._close_notified = False

    def allow_close(self):
        self._deferred_close = False
        self._close_notified = True
        self.close()
