import numpy as np
import cv2
from skimage.util import img_as_ubyte
from abc import ABC
from utils.registry import register

from .helpers import erode_dilate, plot_contours, join_intersecting_near_boxes

from collections import namedtuple
DetectionResult = namedtuple("DetectionResult", ["boxes", "scores", "labels"])


@register("postprocess")
class BeesPostprocessor(ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, batch, pred, original_frames):
        pred_cpu, dets, dets2 = self._prepare_prediction(pred)

        cnts = self._get_contours(dets2)
        existing_boxes = self._extract_boxes(cnts, original_frames)

        self._shift_X(batch)

        return batch, DetectionResult(
            boxes=existing_boxes,
            scores=[0.5] * len(existing_boxes),
            labels=[0] * len(existing_boxes)
        )

    def _prepare_prediction(self, pred):
        kernel = np.ones((3, 3), np.uint8)
        pred_cpu = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
        dets = erode_dilate(pred_cpu, THR=112, kernel=kernel, iterations=1)
        dets2 = cv2.bitwise_not(img_as_ubyte(dets))
        return pred_cpu, dets, dets2

    def _get_contours(self, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cnts

    def _extract_boxes(self, contours, original_frame):
        boxes = []
        for c in contours:
            try:
                _, _, _, color, x, y, w, h = plot_contours(c, original_frames=original_frame, sigma=20)
                boxes.append((x, y, x + w, y + h))
            except TypeError as e:
                print(f"Contour error: {e}")
        return join_intersecting_near_boxes(boxes, distance_threshold=10)

    def _shift_X(self, X):
        for i in range(4):
            X[0, i] = X[0, i + 1].clone()
