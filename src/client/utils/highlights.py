from __future__ import annotations

from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtCore import QRect


from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from utils.paint import load_class_config, hex2rgb, build_palette


class HighlightsRenderer:
    def __init__(self):
        self._highlights: set[tuple[int, int]] = set()
        self.palette = build_palette()

    def _get_color_for_class_id(self, class_id: int) -> QColor:
        color = self.palette[class_id % len(self.palette)]
        return QColor(*color)

    def add_highlight(self, value: int, class_id: int):
        self._highlights.add((value, class_id))

    def remove_highlight(self, value: int, class_id: int):
        self._highlights.discard((value, class_id))

    def clear(self):
        self._highlights.clear()

    def set_highlights(self, highlights):
        self._highlights = set(highlights)

    def paint(self, painter: QPainter, groove_rect: QRect, min_val: int, max_val: int):
        if max_val == min_val or not self._highlights:
            return

        groove_x = groove_rect.x()
        groove_width = groove_rect.width()
        line_center_y = groove_rect.center().y()
        line_half_height = 1

        for (class_id, hl_value) in self._highlights:
            pen_color = self._get_color_for_class_id(class_id)
            pen = QPen(pen_color)
            pen.setWidth(1)
            painter.setPen(pen)
            if min_val <= hl_value <= max_val:
                hl_ratio = (hl_value - min_val) / (max_val - min_val)
                hl_x = groove_x + int(hl_ratio * groove_width)
                painter.drawLine(hl_x, line_center_y - line_half_height, hl_x, line_center_y + line_half_height)

    def find_clicked_highlight(self, click_x: int, groove_rect: QRect, min_val: int, max_val: int,
                               tolerance: int = 3) -> tuple[int, int] | None:
        if max_val == min_val or not self._highlights:
            return None

        groove_x = groove_rect.x()
        groove_width = groove_rect.width()

        for (class_id, hl_value) in self._highlights:
            hl_ratio = (hl_value - min_val) / (max_val - min_val)
            hl_x = groove_x + int(hl_ratio * groove_width)
            if abs(click_x - hl_x) <= tolerance:
                return class_id, hl_value
        return None