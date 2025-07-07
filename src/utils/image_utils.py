from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from PyQt6.QtGui import QImage


def convert_to_qimage(frame: NDArray[np.uint8]) -> QImage:
    """
    Convert an OpenCV BGR numpy array to a QImage (RGB888 format).
    """
    rgb: NDArray[np.uint8] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    bytes_per_line: int = channels * width
    image: QImage = QImage(
        rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    )
    return image.copy()

