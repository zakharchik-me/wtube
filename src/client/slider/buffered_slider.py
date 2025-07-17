from PyQt6.QtWidgets import QSlider, QStyle, QStyleOptionSlider, QToolTip
from PyQt6.QtCore import Qt, QPointF, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPainter, QColor
from ..utils.highlights import HighlightsRenderer
from typing import List


try:
    CC_Slider = QStyle.ComplexControl.CC_Slider
    SC_SliderGroove = QStyle.SubControl.SC_SliderGroove
    SC_SliderHandle = QStyle.SubControl.SC_SliderHandle
except AttributeError:
    CC_Slider = 1
    SC_SliderGroove = 4
    SC_SliderHandle = 16


class BufferedSlider(QSlider):
    pauseRequested = pyqtSignal()
    highlightRequested = pyqtSignal(int, int)
    highlightClicked = pyqtSignal(int, int)

    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.buffer_value = 0
        self._is_dragging = False
        self.highlightRequested.connect(self.add_highlight)

        self.highlights_renderer = HighlightsRenderer()

        self.setMouseTracking(True)

    def setBufferValue(self, value: int):
        self.buffer_value = value
        self.update()

    def styleOption(self) -> QStyleOptionSlider:
        opt = QStyleOptionSlider()
        opt.initFrom(self)
        opt.orientation = self.orientation()
        opt.minimum = self.minimum()
        opt.maximum = self.maximum()
        opt.sliderPosition = self.sliderPosition()
        opt.sliderValue = self.value()
        opt.singleStep = self.singleStep()
        opt.pageStep = self.pageStep()
        opt.upsideDown = self.invertedAppearance()
        opt.tickPosition = self.tickPosition()
        opt.tickInterval = self.tickInterval()
        return opt

    def paintEvent(self, event):
        painter = QPainter(self)
        opt = self.styleOption()

        groove_rect = self.style().subControlRect(
            CC_Slider,
            opt,
            SC_SliderGroove,
            self
        )

        min_val = self.minimum()
        max_val = self.maximum()
        if max_val == min_val:
            painter.end()
            return

        buffer_ratio = (self.buffer_value - min_val) / (max_val - min_val)
        current_ratio = (self.value() - min_val) / (max_val - min_val)

        groove_x = groove_rect.x()
        groove_width = groove_rect.width()
        groove_center_y = groove_rect.center().y()

        buffer_px = int(buffer_ratio * groove_width)
        current_px = int(current_ratio * groove_width)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#808080"))  # серый
        painter.drawRect(groove_x, groove_center_y - 2, current_px, 4)

        visible_buffer_width = max(0, buffer_px - current_px)

        if visible_buffer_width > 0:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#C0C0C0"))  # светло-серый буфер
            painter.drawRect(groove_x + current_px, groove_center_y - 2, visible_buffer_width, 4)

        self.highlights_renderer.paint(painter, groove_rect, min_val, max_val)

        handle_rect = self.style().subControlRect(
            CC_Slider,
            opt,
            SC_SliderHandle,
            self
        )
        painter.setBrush(QColor(128, 128, 128))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(handle_rect)

        painter.end()

    def pixelPosToRangeValue(self, pos: QPointF) -> int:
        opt = self.styleOption()
        groove_rect = self.style().subControlRect(
            CC_Slider,
            opt,
            SC_SliderGroove,
            self
        )

        min_val = self.minimum()
        max_val = self.maximum()

        if self.orientation() == Qt.Orientation.Horizontal:
            slider_length = self.style().pixelMetric(self.style().PixelMetric.PM_SliderLength, opt, self)
            slider_min = groove_rect.x()
            slider_max = groove_rect.right() - slider_length + 1

            pos_x = pos.x()
            if pos_x < slider_min:
                pos_x = slider_min
            elif pos_x > slider_max:
                pos_x = slider_max

            ratio = (pos_x - slider_min) / (slider_max - slider_min)
            value = min_val + (max_val - min_val) * ratio
            return round(value)

        else:
            slider_length = self.style().pixelMetric(self.style().PixelMetric.PM_SliderLength, opt, self)
            slider_min = groove_rect.y()
            slider_max = groove_rect.bottom() - slider_length + 1

            pos_y = pos.y()
            if pos_y < slider_min:
                pos_y = slider_min
            elif pos_y > slider_max:
                pos_y = slider_max

            ratio = (pos_y - slider_min) / (slider_max - slider_min)
            value = min_val + (max_val - min_val) * ratio
            return round(value)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            new_val = self.pixelPosToRangeValue(event.position())

            opt = self.styleOption()
            groove_rect = self.style().subControlRect(
                CC_Slider,
                opt,
                SC_SliderGroove,
                self
            )
            clicked_highlight = self.highlights_renderer.find_clicked_highlight(
                int(event.position().x()), groove_rect, self.minimum(), self.maximum()
            )

            if clicked_highlight is not None:
                class_id, frame = clicked_highlight
                self.highlightClicked.emit(frame, class_id)
                event.accept()
                return

            self.setValue(new_val)
            self._is_dragging = True
            self.pauseRequested.emit()
            event.accept()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        opt = self.styleOption()
        groove_rect = self.style().subControlRect(
            CC_Slider,
            opt,
            SC_SliderGroove,
            self
        )
        hovered_highlight = self.highlights_renderer.find_clicked_highlight(
            int(event.position().x()), groove_rect, self.minimum(), self.maximum()
        )
        if hovered_highlight is not None:
            global_pos = self.mapToGlobal(event.position().toPoint())
            tooltip_pos = global_pos + QPointF(0, 20).toPoint()
            QToolTip.showText(tooltip_pos, f"Highlight: {hovered_highlight[1]}", self)
        else:
            QToolTip.hideText()

        if self._is_dragging:
            new_val = self.pixelPosToRangeValue(event.position())
            self.setValue(new_val)
            event.accept()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = False
            event.accept()
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)

    @pyqtSlot(int)
    def add_highlight(self, value: int, class_id: int):
        self.highlights_renderer.add_highlight(value, class_id)
        self.update()

    def set_highlights(self, highlights: List[int]):
        self.highlights_renderer.set_highlights(highlights)
        self.update()