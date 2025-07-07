import os
import cv2
import numpy as np
import json
from collections import deque
from PyQt6.QtGui import QImage
from typing import List, Tuple, Dict
from utils.image_utils import convert_to_qimage
from utils.video_saver import VideoSaver
from client.buffer.base import BufferFactory


class RollingBuffer(BufferFactory):
    def __init__(self, max_size, video_saver: VideoSaver):
        self.buffer_image = deque()  # stores (frame_index, image)
        self.buffer_index = {}  # frame_index -> image
        self.buffer_log = deque()  # stores (frame_index, log)
        self.log_index = {}     # frame_index -> log (dict)
        self.max_size = max_size
        self.video_saver = video_saver

    def add_image(self, image: np.ndarray, frame_index: int):
        if len(self.buffer_image) >= self.max_size:
            oldest_index, oldest_image = self.buffer_image.popleft()
            self.buffer_index.pop(oldest_index, None)
            if self.video_saver:
                self.video_saver.add_frame(oldest_index, oldest_image)

        self.buffer_image.append((frame_index, image))
        self.buffer_index[frame_index] = image

    def add_ann(self, log: dict, frame_index: int):
        if len(self.buffer_log) >= self.max_size:
            oldest_index, oldest_log = self.buffer_log.popleft()
            self.log_index.pop(oldest_index, None)
            if self.video_saver:
                self.video_saver.add_log2disk(oldest_index, oldest_log)

        self.buffer_log.append((frame_index, log))
        self.log_index[frame_index] = log

    def load_log(self, frame_index):
        if frame_index in self.log_index:
            return self.log_index[frame_index]
        else:
            return None

    def flush(self):
        while self.buffer_image and self.buffer_log:
            frame_index_frame, image = self.buffer_image.popleft()
            frame_index_log, log = self.buffer_log.popleft()
            self.buffer_index.pop(frame_index_frame, None)
            self.log_index.pop(frame_index_log, None)
            if self.video_saver:
                self.video_saver.add_log2disk(frame_index_log, log)
                self.video_saver.add_frame(frame_index_frame, image)

    def _load_image(self, bgr_frame: np.ndarray) -> QImage:
        return convert_to_qimage(bgr_frame)

    def get_image(self, frame_index: int) -> QImage:
        if frame_index in self.buffer_index:
            return self._load_image(self.buffer_index[frame_index])
        else:
            return self.video_saver.get_image(frame_index)

    def _process_frame(
            self,
            frame_index: int
        ) -> Tuple[QImage, List[List[int]], List[float], List[str]]:
            qimage = self.get_image(frame_index)
            log = self.load_log(frame_index)
            boxes, scores, labels = self.load_annotation(log)
            return qimage, boxes, scores, labels

    def load_annotation(
            self,
            data
    ) -> Tuple[List[List[int]], List[float], List[str]]:
        if not isinstance(data, dict):
            print("[RollingBuffer] Invalid data format: not a dict")
            return [], [], []

        raw_anns = data.get("logs", {}).get("annotations", [])

        boxes = []
        scores = []
        labels = []

        for ann in raw_anns:
            if not isinstance(ann, dict):
                print(f"[RollingBuffer] Skipping non-dict annotation: {ann}")
                continue

            bbox = ann.get("bbox")
            score = ann.get("score")
            label = ann.get("label")

            if not bbox or len(bbox) != 4:
                print(f"[RollingBuffer] Invalid bbox: {bbox}")
                continue

            try:
                x, y, x2, y2 = map(int, bbox)
                w = x2 - x
                h = y2 - y
                boxes.append([x, y, w, h])
                scores.append(float(score))
                labels.append(str(label))
            except Exception as e:
                print(f"[RollingBuffer] Error parsing annotation {ann}: {e}")
                continue
        return boxes, scores, labels

    def get_all_indices(self) -> List[int]:
        return list(self.log_index.keys())
