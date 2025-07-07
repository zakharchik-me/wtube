import os
import glob
import time
import json
from typing import Tuple, List, Dict
from PyQt6.QtGui import QImage
from .base import BufferFactory
from utils.registry import register
from utils.image_utils import convert_to_qimage
import cv2


@register("buffer")
class FolderBuffer(BufferFactory):
    def __init__(self, path: str):
        """
        :param path: Path to a directory containing .jpg images and .json annotation files.
        """
        # print("path: ", path)
        self.image_folder = os.path.join(path, 'images')
        self.ann_folder = os.path.join(path, 'annotations')
        self.index = 1

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[QImage, List[List[int]], List[float], List[str]]:
        """
        Returns:
            A tuple (QImage, boxes, scores, labels), where:
              - boxes is List[List[int]] of [x, y, w, h]
              - scores is List[float]
              - labels is List[str]
        If we run out of images, we sleep briefly and try again (blocking until new files appear).
        """
        ann_path = os.path.join(self.ann_folder, f"{self.index:06d}.json")
        while not os.path.exists(ann_path):
            print("Index: ", self.index)
            time.sleep(0.5)
            self.__next__()
        data = self._load_json(ann_path)
        image_path = os.path.join(self.image_folder, f"{self.index:06d}.jpg")
        try:
            bgr_frame = cv2.imread(image_path)
        except ImportError:
            raise StopIteration

        qimage = self._load_image(bgr_frame)
        boxes, scores, labels = self._load_annotation(data)

        self.index += 1
        return qimage, boxes, scores, labels

    def _load_image(self, bgr_frame) -> QImage:
        qimage: QImage = convert_to_qimage(bgr_frame)
        return qimage

    def _load_annotation(self, data: Dict[str, List]) -> Tuple[List[List[int]], List[float], List[str]]:
        """
        Reads the corresponding JSON annotation file (same base name as img_path) and returns:
          - boxes: List of [x, y, w, h]
          - scores: List of floats
          - labels: List of strings
        If no annotation file exists, returns three empty lists.
        """
        raw_anns = data.get("annotations", [])
        if not isinstance(raw_anns, list):
            print(f"[FolderBuffer] JSON {json_path} has no 'annotations' list.")
            return [], [], []

        boxes: List[List[int]] = []
        scores: List[float] = []
        labels: List[str] = []

        for ann in raw_anns:
            bbox = ann.get("bbox", None)
            score = ann.get("score", None)
            label = ann.get("label", None)

            if (not isinstance(bbox, list)) or len(bbox) != 4:
                continue
            try:
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                score_f = float(score)
                label_s = str(label)
            except (ValueError, TypeError):
                continue

            boxes.append([x, y, w, h])
            scores.append(score_f)
            labels.append(label_s)

        return boxes, scores, labels

    def _load_json(self, json_path: str) -> Dict[str, List]:
        """Loads annotation with images names and annotations"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    def load_log(self, number: int) -> Dict[str, List]:
        """Loads annotation with images names and annotations"""
        path = f"{self.ann_folder}/{number:09d}.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data
