"""A small native botanical scene. Only real controller input changes its state."""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QRadialGradient, QTransform
from PySide6.QtWidgets import QWidget


def _number(value, default=0.0):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _unit(value, default=0.0):
    value = _number(value, default)
    return max(0.0, min(1.0, value / 100 if value > 1 else value))


def _mix(first, second, amount):
    amount = max(0.0, min(1.0, amount))
    return QColor(*(round(a + (b - a) * amount) for a, b in zip(first.getRgb(), second.getRgb())))


def _point(curve, t):
    a, b, c, d = curve
    s = 1 - t
    return QPointF(s ** 3 * a[0] + 3 * s * s * t * b[0] + 3 * s * t * t * c[0] + t ** 3 * d[0],
                   s ** 3 * a[1] + 3 * s * s * t * b[1] + 3 * s * t * t * c[1] + t ** 3 * d[1])


def _stem(curve, width):
    """Cache a tapered silhouette and its softly lit edge, in design coordinates."""
    left, right = [], []
    for step in range(25):
        t = step / 24
        point = _point(curve, t)
        before, after = _point(curve, max(0, t - .015)), _point(curve, min(1, t + .015))
        dx, dy = after.x() - before.x(), after.y() - before.y()
        length = max(.001, math.hypot(dx, dy))
        radius = (width * (1 - t) ** 1.25 + .42) / 2
        normal = QPointF(-dy / length * radius, dx / length * radius)
        left.append(point + normal)
        right.append(point - normal)
    shape = QPainterPath(left[0])
    for point in left[1:] + list(reversed(right)):
        shape.lineTo(point)
    shape.closeSubpath()
    edge = QPainterPath(left[0])
    for point in left[1:]:
        edge.lineTo(point)
    return shape, edge


def _leaf_shape(length, width):
    leaf = QPainterPath(QPointF(0, 0))
    leaf.cubicTo(length * .35, -width * .68, length * .77, -width * .43, length, 0)
    leaf.cubicTo(length * .68, width * .49, length * .20, width * .61, 0, 0)
    return leaf


@dataclass(slots=True)
class _Leaf:
    path: QPainterPath
    shade: int
    keep: float


@dataclass(slots=True)
class _Bough:
    origin: QPointF
    stem: QPainterPath
    edge: QPainterPath
    twigs: QPainterPath
    leaves: list[_Leaf]
    buds: list[QPointF]
    side: float


def _geometry():
    """Seeded botanical geometry is built once, never in the animation loop."""
    rng = random.Random(1739)
    trunk_curve = ((0, 1), (-33, -107), (22, -167), (4, -269))
    curves = [
        (((-10, -126), (-72, -124), (-155, -145), (-205, -177)), 6.0),
        (((-9, -111), (-61, -160), (-151, -174), (-192, -221)), 7.8),
        (((-3, -167), (-63, -189), (-113, -236), (-155, -263)), 6.6),
        (((1, -216), (-31, -267), (-71, -291), (-99, -319)), 5.4),
        (((4, -252), (-20, -297), (-27, -336), (-13, -362)), 3.7),
        (((5, -235), (42, -276), (68, -316), (54, -345)), 4.8),
        (((3, -198), (57, -218), (110, -271), (116, -293)), 6.0),
        (((-2, -157), (71, -171), (126, -193), (161, -240)), 7.0),
        (((-11, -113), (51, -128), (138, -144), (190, -184)), 7.4),
    ]
    boughs = []
    for index, (curve, width) in enumerate(curves):
        # Attach each branch to the actual curved trunk, not its centre axis.
        low, high = 0.0, 1.0
        for _ in range(16):
            middle = (low + high) / 2
            if _point(trunk_curve, middle).y() > curve[0][1]:
                low = middle
            else:
                high = middle
        attachment = _point(trunk_curve, (low + high) / 2)
        curve = ((attachment.x(), attachment.y()), *curve[1:])
        stem, edge = _stem(curve, width)
        twigs, leaves, buds = QPainterPath(), [], []
        side = -1 if curve[-1][0] < 0 else 1
        for spray in range(6):
            fraction = .28 + spray * .135
            origin = _point(curve, fraction)
            ahead = _point(curve, min(1, fraction + .03))
            angle = math.atan2(ahead.y() - origin.y(), ahead.x() - origin.x())
            angle += (-1 if spray % 2 else 1) * rng.uniform(.65, 1.12)
            # Shoots tend toward light, while keeping an irregular crown outline.
            if math.sin(angle) > .20:
                angle = -angle
            length = rng.uniform(29, 48) * (1.05 - .25 * fraction)
            tip = origin + QPointF(math.cos(angle) * length, math.sin(angle) * length)
            bend = origin + QPointF(math.cos(angle + .16) * length * .58,
                                    math.sin(angle + .16) * length * .58)
            twigs.moveTo(origin)
            twigs.quadTo(bend, tip)
            for pair in range(4):
                along = .25 + pair * .20
                position = origin * ((1 - along) ** 2) + bend * (2 * along * (1 - along)) + tip * (along ** 2)
                for direction in (-1, 1):
                    turn = angle + direction * rng.uniform(.55, .95)
                    size = rng.uniform(12, 22) * (1.13 - pair * .09)
                    transform = QTransform()
                    transform.translate(position.x(), position.y())
                    transform.rotate(math.degrees(turn))
                    leaves.append(_Leaf(transform.map(_leaf_shape(size, size * rng.uniform(.38, .57))),
                                        rng.randrange(8), rng.random()))
            transform = QTransform()
            transform.translate(tip.x() - math.cos(angle) * 3, tip.y() - math.sin(angle) * 3)
            transform.rotate(math.degrees(angle))
            leaves.append(_Leaf(transform.map(_leaf_shape(rng.uniform(14, 22), 7)), rng.randrange(8), rng.random()))
            if (spray + index) % 3 == 0:
                buds.append(tip)
        boughs.append(_Bough(QPointF(*curve[0]), stem, edge, twigs, leaves, buds, side))
    trunk, trunk_edge = _stem(trunk_curve, 22)
    roots = []
    for curve, width in [(((0, -14), (-15, -1), (-36, 3), (-72, 7)), 11),
                         (((4, -13), (23, 1), (38, 2), (57, 6)), 9),
                         (((-2, -8), (-9, 3), (-14, 7), (-26, 12)), 7),
                         (((3, -7), (14, 4), (18, 7), (27, 11)), 5)]:
        roots.append(_stem(curve, width)[0])
    return boughs, trunk, trunk_edge, roots


class LivingTree(QWidget):
    """An asymmetric tree with quiet, signal-driven growth and recovery."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 390)
        self._state = {}
        self._score = None
        self._started = time.monotonic()
        self._last_frame = self._started
        self._target = dict(health=.8, risk=0.0, bloom=.12, wind=.08, observing=1.0, alert=0.0)
        self._current = self._target.copy()
        self._boughs, self._trunk, self._trunk_edge, self._roots = _geometry()
        self._falling_leaf = _leaf_shape(12, 5)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._animate)

    def showEvent(self, event):
        super().showEvent(event)
        self._last_frame = time.monotonic()
        self._timer.start()

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def _animate(self):
        if not self.isVisible():
            return
        now = time.monotonic()
        elapsed = min(.1, max(0, now - self._last_frame))
        self._last_frame = now
        blend = 1 - math.exp(-elapsed / .85)
        for key, target in self._target.items():
            # Colour should respond promptly when the readout enters red;
            # branch movement retains the slower botanical transition.
            response = 1 - math.exp(-elapsed / .16) if key == 'alert' else blend
            self._current[key] += (target - self._current[key]) * response
        self.update()

    def set_state(self, tree: dict | None, sbi=None):
        self._state = dict(tree) if isinstance(tree, dict) else {}
        self._score = None if sbi is None else _number(sbi, None)
        risk = _unit(self._state.get('risk'), _number(self._score) / 100)
        health = _unit(self._state.get('health'), 1 - risk)
        observing = self._score is None or self._state.get('mode') in ('observing', 'idle')
        alert = not observing and (self._score >= 60 or self._state.get('mode') in ('risk', 'critical'))
        self._target = dict(health=health, risk=risk,
                            bloom=_unit(self._state.get('bloom'), .12 if observing else health * .65),
                            wind=_unit(self._state.get('wind'), .08), observing=float(observing), alert=float(alert))
        if not self.isVisible():
            self._current = self._target.copy()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            self._paint_scene(painter)
        finally:
            # Release native painting before widget shutdown, including on exceptions.
            painter.end()

    def _paint_scene(self, painter):
        w, h = self.width(), self.height()
        bounds = QRectF(.5, .5, w - 1, h - 1)
        alert = self._current['alert']
        painter.setPen(QPen(_mix(QColor('#1c3a43'), QColor('#e04b63'), alert), 1 + alert * .6))
        background = QLinearGradient(0, 0, w, h)
        background.setColorAt(0, QColor('#091c24'))
        background.setColorAt(1, QColor('#07171e'))
        painter.setBrush(background)
        painter.drawRoundedRect(bounds, 20, 20)
        clip = QPainterPath()
        clip.addRoundedRect(bounds, 20, 20)
        painter.setClipPath(clip)

        values = self._current
        risk, health, bloom = values['risk'], values['health'], values['bloom']
        observing, wind = values['observing'], values['wind']
        t = time.monotonic() - self._started
        accent = _mix(QColor('#82d9b4'), QColor('#deb67c'), risk / .55)
        if risk > .55:
            accent = _mix(QColor('#deb67c'), QColor('#e68e88'), (risk - .55) / .45)
        accent = _mix(accent, QColor('#80a99e'), observing * .42)
        accent = _mix(accent, QColor('#ff586f'), alert)

        glow = QRadialGradient(QPointF(w * .50, h * .45), max(w * .55, h * .40))
        light = QColor(accent)
        # Slow breathing light, well below flashing frequency, makes the
        # risk state readable at a glance without generating sensor scores.
        breath = .88 + .12 * math.sin(t * 1.8)
        light.setAlpha(round(20 + 52 * alert * breath))
        glow.setColorAt(0, light)
        glow.setColorAt(.60, _mix(QColor(20, 62, 59, 10), QColor(131, 27, 52, 24), alert))
        glow.setColorAt(1, QColor(7, 23, 30, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawRect(bounds)

        compact = h < 360
        scale = max(.05, min((w - 28) / 490, (h - (96 if compact else 122)) / 406))
        base_x, base_y = w * .50, h - (39 if compact else 61)
        painter.save()
        painter.translate(base_x, base_y)
        painter.scale(scale, scale)
        self._paint_tree(painter, t, accent, health, risk, bloom, wind, observing, alert)
        painter.restore()

        painter.setPen(QColor('#a8c7c2'))
        painter.setFont(QFont('Microsoft YaHei UI', 9 if compact else 11, QFont.Weight.Medium))
        painter.drawText(QRectF(17, 14, w - 108, 24), Qt.AlignmentFlag.AlignLeft, '生命树')
        painter.setPen(_mix(QColor('#688c91'), QColor('#ff9aa5'), alert))
        painter.setFont(QFont('Microsoft YaHei UI', 7))
        if self._target['alert']:
            painter.drawText(QRectF(17, 39, w - 24, 16), Qt.AlignmentFlag.AlignLeft, '高风险 · 建议停顿')
        painter.setFont(QFont('Segoe UI', 19 if compact else 23, QFont.Weight.Light))
        painter.setPen(_mix(QColor('#d4ebe2'), QColor('#ff8796'), alert))
        score = '—' if self._score is None else f'{self._score:.1f}'
        painter.drawText(QRectF(w - 100, 11, 80, 32), Qt.AlignmentFlag.AlignRight, score)
        painter.setFont(QFont('Microsoft YaHei UI', 7))
        painter.setPen(QColor('#698a90'))
        painter.drawText(QRectF(w - 105, 43, 85, 16), Qt.AlignmentFlag.AlignRight, '当前 SBI')

        mode = str(self._state.get('mode') or 'observing')
        captions = {'flourishing': '舒展生长', 'healthy': '平稳交流', 'stressed': '感知交流压力',
                    'withering': '停顿，让表达舒缓', 'calm': '平稳交流', 'idle': '静候交流，慢慢生长',
                    'warning': '感知交流压力', 'critical': '停顿，让表达舒缓',
                    'observing': '静候交流，慢慢生长', 'friendly': '表达舒展，枝叶向光',
                    'signal': '觉察变化，留一点呼吸', 'risk': '停顿，让表达舒缓'}
        painter.setPen(_mix(QColor('#c3dbd2'), QColor('#ffb1b8'), alert))
        painter.setFont(QFont('Microsoft YaHei UI', 8 if compact else 11))
        painter.drawText(QRectF(12, h - 28, w - 24, 22), Qt.AlignmentFlag.AlignCenter, captions.get(mode, '觉察表达变化'))

    def _paint_tree(self, painter, t, accent, health, risk, bloom, wind, observing, alert):
        painter.setPen(Qt.PenStyle.NoPen)
        ground = QRadialGradient(QPointF(0, 3), 133)
        ground.setColorAt(0, _mix(QColor(79, 151, 123, 28), QColor(228, 55, 81, 45), alert))
        ground.setColorAt(.65, _mix(QColor(42, 109, 92, 12), QColor(135, 30, 53, 18), alert))
        ground.setColorAt(1, QColor(19, 50, 48, 0))
        painter.setBrush(ground)
        painter.drawEllipse(QRectF(-142, -10, 284, 32))
        painter.setPen(QPen(_mix(QColor(96, 154, 135, 22), QColor(243, 91, 109, 48), alert), .8))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(QRectF(-125, -11, 250, 30), 186 * 16, 160 * 16)

        # A sparse, deterministic dust field is decorative, never a sensor signal.
        for i in range(25):
            phase = i * 2.399963
            x = math.sin(phase) * (85 + (i * 31) % 157) + math.sin(t * .14 + i) * 4
            y = -26 - (i * 53 % 352) + math.sin(t * .19 + phase) * 5
            alpha = int((12 + 20 * (.5 + .5 * math.sin(t * .5 + i))) * (1 - risk * .55))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_mix(QColor(151, 207, 171, alpha), QColor(255, 137, 147, alpha + 8), alert))
            painter.drawEllipse(QPointF(x, y), 1.1 if i % 3 else 1.6, 1.1 if i % 3 else 1.6)

        painter.save()
        painter.rotate(math.sin(t * .55) * (.20 + wind * .45))
        bark = QLinearGradient(-18, 0, 30, -270)
        bark.setColorAt(0, _mix(QColor('#507a6b'), QColor('#91354d'), alert))
        bark.setColorAt(.42, _mix(QColor('#6f9580'), QColor('#df5369'), alert))
        bark.setColorAt(1, _mix(QColor('#749782'), QColor('#ff8894'), alert))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(bark)
        for root in self._roots:
            painter.drawPath(root)
        painter.drawPath(self._trunk)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(_mix(QColor(178, 202, 157, 72), QColor(255, 185, 188, 125), alert), .8))
        painter.drawPath(self._trunk_edge)

        palette = []
        for shade in range(8):
            base = _mix(QColor('#265f54'), accent, .29 + shade * .094)
            red = _mix(QColor('#c32c48'), QColor('#ff9ca4'), shade / 7)
            base = _mix(base, red, alert)
            neutral_alpha = (116 + shade * 16) * (1 - observing * .12) * (1 - risk * .18)
            base.setAlpha(round(neutral_alpha + (205 + shade * 7 - neutral_alpha) * alert))
            palette.append(base)
        retained = max(.29 + .21 * alert, 1 - risk * .53 - (1 - health) * .12)
        droop = max(0, (risk - .26) / .74)
        for index, bough in enumerate(self._boughs):
            painter.save()
            painter.translate(bough.origin)
            sway = math.sin(t * (.56 + index * .017) + index * .83) * (.30 + wind * 1.3)
            painter.rotate(sway + bough.side * droop * (7 + index % 3 * 2))
            painter.translate(-bough.origin)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_mix(QColor('#5c826d'), QColor('#e06579'), alert))
            painter.drawPath(bough.stem)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(_mix(QColor(169, 199, 156, 58), QColor(255, 175, 181, 115), alert), .62))
            painter.drawPath(bough.edge)
            painter.setPen(QPen(_mix(QColor(117, 155, 115, 112), QColor(250, 124, 139, 185), alert), .65, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPath(bough.twigs)
            painter.setPen(Qt.PenStyle.NoPen)
            for leaf in bough.leaves:
                opacity = max(0, min(1, (retained - leaf.keep) * 8))
                if opacity <= .01:
                    continue
                painter.setOpacity(opacity)
                painter.setBrush(palette[leaf.shade])
                painter.drawPath(leaf.path)
            painter.setOpacity(1)
            for bud_index, position in enumerate(bough.buds):
                strength = max(0, min(1, bloom * 1.6 - ((index * 7 + bud_index * 3) % 11) / 15))
                strength *= (1 - risk * .80)
                if strength > .025:
                    self._paint_bud(painter, position, strength, t + index * .7)
            painter.restore()

        if risk > .47:
            for i in range(9):
                progress = (t * .055 + i * .137) % 1
                x = math.sin(i * 2.7) * 168 + math.sin(progress * 7 + i) * 16
                y = -235 + progress * 251
                alpha = min(1, progress * 7, (1 - progress) * 6) * (risk - .47) * .85
                painter.save()
                painter.translate(x, y)
                painter.rotate(i * 39 + progress * 145)
                painter.setOpacity(alpha)
                painter.setBrush(accent)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawPath(self._falling_leaf)
                painter.restore()
        painter.restore()

    @staticmethod
    def _paint_bud(painter, position, strength, t):
        pulse = .92 + .08 * math.sin(t * .95)
        radius = 10 + 6 * strength
        halo = QRadialGradient(position, radius)
        halo.setColorAt(0, QColor(244, 215, 143, round(42 * strength * pulse)))
        halo.setColorAt(1, QColor(244, 215, 143, 0))
        painter.setBrush(halo)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(position, radius, radius)
        painter.setOpacity(strength)
        painter.setBrush(QColor('#e7d9a6'))
        size = 1.4 + strength * 1.05
        for petal in range(5):
            angle = petal * math.tau / 5 - math.pi / 2
            center = position + QPointF(math.cos(angle) * size * .90, math.sin(angle) * size * .90)
            painter.drawEllipse(center, size * .82, size * 1.02)
        painter.setBrush(QColor('#ffdf91'))
        painter.drawEllipse(position, 1.25, 1.25)
        painter.setOpacity(1)
