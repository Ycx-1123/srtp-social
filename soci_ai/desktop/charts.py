"""Native, dependency-free session timeline; gaps remain gaps."""
from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


def finite_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _timeline_rows(history):
    """Filter and sort the whole record so labels never drift from scores."""
    rows = []
    for row in history:
        if not isinstance(row, dict):
            continue
        at = finite_number(row.get('at_ms'))
        if at is None or at < 0:
            continue
        rows.append(row)
    return sorted(rows, key=lambda row: float(row['at_ms']))


def timeline_points(history):
    """Return measured samples sorted by time without filling missing scores."""
    points = []
    for row in _timeline_rows(history):
        values = []
        for key in ('sbi', 'friendliness'):
            value = finite_number(row.get(key))
            values.append(None if value is None else max(0, min(100, value)))
        points.append((float(row['at_ms']) / 1000, *values))
    return points


def sample_caption(row):
    """Short recorded cue labels only: no transcript or guessed past cause."""
    sbi = finite_number(row.get('sbi'))
    if sbi is None:
        return '暂无有效信号'
    cues = row.get('cues')
    if isinstance(cues, list):
        labels = list(dict.fromkeys(label.strip() for label in cues if isinstance(label, str) and label.strip()))[:3]
        if labels:
            phases = row.get('cue_phases')
            if isinstance(phases, dict):
                if all(phases.get(label) == 'recovering' for label in labels):
                    return ' · '.join(labels) + '（回落中）'
                return ' · '.join(label + ('（回落中）' if phases.get(label) == 'recovering' else '') for label in labels)
            caption = ' · '.join(labels)
            return caption + ('（回落中）' if row.get('cue_phase') == 'recovering' else '')
        if sbi < 25:
            return '当前信号平稳'
    return '未记录该点的具体线索'


def series_segments(points, column):
    """Split each series at missing measurements (including leading gaps)."""
    result, segment = [], []
    for point in points:
        if point[column] is None:
            if segment:
                result.append(segment)
                segment = []
        else:
            segment.append((point[0], point[column]))
    if segment:
        result.append(segment)
    return result


def time_label(seconds):
    seconds = max(0, int(round(seconds)))
    minutes, second = divmod(seconds, 60)
    return f'{minutes:02}:{second:02}'


class SessionTimeline(QWidget):
    """QPainter timeline with real time spacing and inspectable samples."""

    CORAL = '#ff647e'
    MINT = '#69e4bb'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setAccessibleName('会话分数时间线')
        self.setAccessibleDescription('横轴为会话时间，纵轴为 SBI，零至一百分。绿黄红渐变表示信号增强，60 分及以上标红。缺失数据保持断开。')
        self.points = []
        self.captions = []
        self.duration = 1.0
        self.hover_index = None

    def set_history(self, history, elapsed_seconds=None):
        rows = _timeline_rows(history or [])
        self.points = timeline_points(rows)
        self.captions = [sample_caption(row) for row in rows]
        elapsed = finite_number(elapsed_seconds) or 0
        self.duration = max(1, elapsed, self.points[-1][0] if self.points else 0)
        self.hover_index = None
        self.setToolTip('')
        self.update()

    def event(self, event):
        # The custom chart card is immediate; do not show a duplicate Qt popup.
        if event.type() == QEvent.Type.ToolTip:
            return True
        return super().event(event)

    def plot_rect(self):
        return QRectF(44, 25, max(1, self.width() - 66), max(1, self.height() - 65))

    def _position(self, at, value):
        area = self.plot_rect()
        return QPointF(area.left() + at / self.duration * area.width(), area.bottom() - value / 100 * area.height())

    def mouseMoveEvent(self, event):
        area = self.plot_rect()
        if self.points and area.contains(event.position()):
            at = (event.position().x() - area.left()) / area.width() * self.duration
            self.hover_index = min(range(len(self.points)), key=lambda index: abs(self.points[index][0] - at))
        else:
            self.hover_index = None
        self.setToolTip(self.hover_text())
        self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.hover_index = None
        self.setToolTip('')
        self.update()
        super().leaveEvent(event)

    def hover_text(self):
        if self.hover_index is None:
            return ''
        at, sbi, _ = self.points[self.hover_index]
        score = '—' if sbi is None else f'{sbi:.1f}'
        return f'会话时间  {time_label(at)}\nSBI  {score}\n{self.captions[self.hover_index]}'

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            area = self.plot_rect()
            # A quiet risk zone is a threshold guide, never an inferred event.
            threshold_y = self._position(0, 60).y()
            risk_zone = QLinearGradient(0, area.top(), 0, threshold_y)
            risk_zone.setColorAt(0, QColor(255, 91, 119, 18))
            risk_zone.setColorAt(1, QColor(255, 91, 119, 6))
            painter.fillRect(QRectF(area.left(), area.top(), area.width(), threshold_y - area.top()), risk_zone)
            painter.setFont(QFont('Microsoft YaHei UI', 8))
            for value in (0, 25, 60, 100):
                y = self._position(0, value).y()
                painter.setPen(QPen(QColor('#693744' if value == 60 else '#233942'), 1, Qt.PenStyle.DashLine))
                painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
                painter.setPen(QColor('#ec889b' if value == 60 else '#7797a6'))
                painter.drawText(QRectF(0, y - 9, 32, 18), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, str(value))
            painter.setPen(QColor('#dc8e9d'))
            painter.drawText(QRectF(area.right()-140, area.top()-19, 140, 18), Qt.AlignmentFlag.AlignRight, '高风险区 · SBI ≥ 60')
            tick_count = 4 if self.width() < 600 else 5
            for index in range(tick_count):
                at = index / (tick_count - 1) * self.duration
                x = self._position(at, 0).x()
                painter.setPen(QColor('#7797a6'))
                tick = f'{at:.1f}秒' if self.duration < 10 else time_label(at)
                painter.drawText(QRectF(x - 25, area.bottom() + 12, 50, 18), Qt.AlignmentFlag.AlignCenter, tick)
            painter.save()
            painter.setClipRect(area.adjusted(-5, -5, 5, 5))
            stroke = QLinearGradient(0, area.bottom(), 0, area.top())
            for stop, color in ((0, self.MINT), (.25, '#83e0b4'), (.45, '#edc678'), (.60, self.CORAL), (1, '#ff526f')):
                stroke.setColorAt(stop, QColor(color))
            for segment in series_segments(self.points, 1):
                path = QPainterPath(self._position(*segment[0]))
                previous = self._position(*segment[0])
                for sample in segment[1:]:
                    current = self._position(*sample)
                    # Horizontal tangents stay inside both observed values:
                    # smooth joins without overshooting measured peaks.
                    offset = (current.x() - previous.x()) / 3
                    path.cubicTo(QPointF(previous.x()+offset, previous.y()),
                                 QPointF(current.x()-offset, current.y()), current)
                    previous = current
                if len(segment) > 1:
                    fill = QPainterPath(path)
                    fill.lineTo(self._position(segment[-1][0], 0))
                    fill.lineTo(self._position(segment[0][0], 0))
                    fill.closeSubpath()
                    gradient = QLinearGradient(0, area.top(), 0, area.bottom())
                    gradient.setColorAt(0, QColor(255, 100, 126, 35))
                    gradient.setColorAt(.55, QColor(237, 198, 120, 13))
                    gradient.setColorAt(1, QColor(105, 228, 187, 3))
                    painter.fillPath(fill, gradient)
                glow = QLinearGradient(0, area.bottom(), 0, area.top())
                for stop, color in stroke.stops():
                    color.setAlpha(26)
                    glow.setColorAt(stop, color)
                painter.setPen(QPen(glow, 10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                painter.drawPath(path)
                painter.setPen(QPen(stroke, 4.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                painter.setBrush(stroke)
                if len(segment) == 1:
                    painter.drawEllipse(self._position(*segment[0]), 3, 3)
                else:
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawPath(path)
            painter.restore()
            if not any(point[1] is not None for point in self.points):
                painter.setPen(QColor('#91b2bf'))
                painter.setFont(QFont('Microsoft YaHei UI', 10))
                painter.drawText(area, Qt.AlignmentFlag.AlignCenter, '完成一次会话后，在这里看见表达的变化')
            if self.hover_index is not None:
                self._paint_hover(painter, area)
        finally:
            painter.end()

    def _paint_hover(self, painter, area):
        at, sbi, _ = self.points[self.hover_index]
        x = self._position(at, 0).x()
        painter.setPen(QPen(QColor('#7a99a8'), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
        color = self.CORAL if sbi is not None and sbi >= 60 else '#edc678' if sbi is not None and sbi >= 25 else self.MINT
        if sbi is not None:
            painter.setPen(QPen(QColor('#0b1d29'), 2))
            painter.setBrush(QColor(color))
            painter.drawEllipse(self._position(at, sbi), 5, 5)
        width = min(268, area.width())
        box = QRectF(min(max(area.left(), x + 12), area.right() - width), area.top() + 9, width, 96)
        painter.setPen(QPen(QColor('#35515d'), 1))
        painter.setBrush(QColor('#102631'))
        painter.drawRoundedRect(box, 8, 8)
        painter.setFont(QFont('Microsoft YaHei UI', 8))
        painter.setPen(QColor('#d7e9ed'))
        painter.drawText(QRectF(box.left()+12, box.top()+8, width-24, 19), Qt.AlignmentFlag.AlignLeft, '会话时间  ' + time_label(at))
        painter.setPen(QColor(color))
        text = '—' if sbi is None else f'{sbi:.1f}'
        painter.drawText(QRectF(box.left()+12, box.top()+30, width-24, 19), Qt.AlignmentFlag.AlignLeft, 'SBI  ' + text)
        painter.setPen(QColor('#a2bfca'))
        painter.drawText(QRectF(box.left()+12, box.top()+52, width-24, 36),
                         Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap, self.captions[self.hover_index])
