from typing import List, Tuple, Optional, Union, Set
import numpy as np
from PyQt6.QtCore import QRect, Qt, pyqtSlot
from PyQt6.QtGui import QImage, QPainter, QPaintEvent, QPen, QPixmap, QColor
from PyQt6.QtWidgets import QWidget

from utils.image_utils import convert_to_qimage
from utils.paint import load_class_config, hex2rgb, build_palette


class ImageViewer(QWidget):
    """Widget for displaying an image with overlaid bounding boxes and labels."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap = QPixmap()
        self._annotations: List[Tuple[QRect, float, str, int]] = []

        self._raw_boxes = []
        self._raw_scores = []
        self._raw_labels = []

        self.palette = build_palette()
        self.id_to_label = load_class_config()

        self.confidence_threshold: float = 0.3

        self._include_class = {i: True for i in range(80)}

    def set_confidence_threshold(self, confidence_threshold: float) -> None:
        self.confidence_threshold = confidence_threshold
        self._reprocess_annotations()  # Reapply filtering
        self.update()

    def set_include_class(self, class_id: int, include_class: bool) -> None:
        self._include_class[class_id] = include_class
        self.update()

    def set_include_classes(self, classes_to_include: Set[int]) -> None:
        for class_id in self._include_class.keys():
            self._include_class[class_id] = class_id in classes_to_include
        self.update()

    def _get_color_for_class_id(self, class_id: int) -> QColor:
        color = self.palette[class_id % len(self.palette)]
        return QColor(*color)

    def _process_boxes(
        self,
        boxes: List[List[int]],
        scores: List[float],
        labels: List[Union[int, str]],
    ) -> List[Tuple[int, int, int, int, float, str, int]]:
        processed = []
        for box, score, label in zip(boxes, scores, labels):
            if len(box) != 4 or score < self.confidence_threshold:
                continue
            try:
                x, y, w, h = box
                score_f = float(score)

                class_id = int(label)
                label_s = self.id_to_label.get(class_id, str(class_id))

                processed.append((x, y, w, h, score_f, label_s, class_id))
            except (ValueError, TypeError):
                print("[ImageViewer] Error while parsing bbox/scores/labels.")
                continue
        return processed

    def _reprocess_annotations(self) -> None:
        """Reprocess annotations after threshold/class filter changes."""
        processed = self._process_boxes(self._raw_boxes, self._raw_scores, self._raw_labels)
        self._annotations = [
            (QRect(x, y, w, h), score, label, class_id)
            for x, y, w, h, score, label, class_id in processed
        ]

    @pyqtSlot(object, object, object, object)
    def update_image(
        self,
        img: QImage,
        boxes: List[List[int]],
        scores: List[float],
        labels: List[Union[int, str]],
    ) -> None:
        if isinstance(img, np.ndarray):
            img = convert_to_qimage(img)

        self._pixmap = QPixmap.fromImage(img)
        self._raw_boxes = boxes
        self._raw_scores = scores
        self._raw_labels = labels
        self._reprocess_annotations()
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)

        if not self._pixmap.isNull():
            painter.drawPixmap(self.rect(), self._pixmap)

            font = painter.font()
            font.setPointSize(10)
            painter.setFont(font)

            for rect, score, label, class_id in self._annotations:
                if not self._include_class.get(class_id, False):
                    continue
                ratio_w = self.rect().width() / self._pixmap.width()
                ratio_h = self.rect().height() / self._pixmap.height()

                new_rect = QRect(
                    int(rect.x() * ratio_w),
                    int(rect.y() * ratio_h),
                    int(rect.width() * ratio_w),
                    int(rect.height() * ratio_h),
                )

                pen_color = self._get_color_for_class_id(class_id)
                pen = QPen(pen_color)
                pen.setWidth(2)
                painter.setPen(pen)

                painter.drawRect(new_rect)
                text = f"{label}: {score:.2f}"
                text_x = new_rect.x()
                text_y = new_rect.y() - 5 if new_rect.y() - 5 >= 0 else new_rect.y() + 15
                painter.drawText(text_x, text_y, text)

        painter.end()
