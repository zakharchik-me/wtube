from PyQt6.QtWidgets import QSlider, QStyle, QStyleOptionSlider
from PyQt6.QtCore import Qt, QPointF, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPainter, QColor


try:
    CC_Slider = QStyle.ComplexControl.CC_Slider
    SC_SliderGroove = QStyle.SubControl.SC_SliderGroove
except AttributeError:
    CC_Slider = 1
    SC_SliderGroove = 4



class BufferedSlider(QSlider):
    pauseRequested = pyqtSignal()


    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.buffer_value = 0
        self._is_dragging = False

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
        super().paintEvent(event)

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

        visible_buffer_width = max(0, buffer_px - current_px)

        if visible_buffer_width > 0:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 100))
            painter.drawRect(groove_x + current_px, groove_center_y - 2, visible_buffer_width, 4)

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
            self.setValue(new_val)
            self._is_dragging = True
            self.pauseRequested.emit()
            event.accept()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
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
