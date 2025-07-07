import json
import os
import shutil
import time
from collections import deque
from typing import Tuple, List, Dict
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage
from client.buffer.base import BufferFactory
from utils.image_utils import convert_to_qimage
from utils.registry import register
from utils.logging import save_log, save_img


class FrameReader(QThread):
    def __init__(self, path: str):
        super().__init__()
        self.image_folder = os.path.join(path, "images")
        self.ann_folder = os.path.join(path, "annotations")

    def get_frame(self, frame_index: int) -> np.ndarray:
        image_path = os.path.join(self.image_folder, f"{frame_index:09d}.jpg")
        start = time.time()
        bgr_frame = cv2.imread(image_path)
        print("[FrameReader]: Elapsed: {}".format(time.time() - start))
        return bgr_frame

    def close(self):
        self.quit()
        self.wait()

class ImageFlushWorker(QThread):
    flush_requested = pyqtSignal()

    def __init__(self, saver: 'VideoSaver'):
        super().__init__()
        self.saver = saver
        self.flush_requested.connect(self.flush)

    @pyqtSlot()
    def flush(self):
        self.saver.flush_images()
        print("[ImageFlushWorker] flush_images done.")


class LogFlushWorker(QThread):
    flush_requested = pyqtSignal()

    def __init__(self, saver: 'VideoSaver'):
        super().__init__()
        self.saver = saver
        self.flush_requested.connect(self.flush)

    @pyqtSlot()
    def flush(self):
        self.saver.flush_logs()
        print("[LogFlushWorker] flush_logs done.")


@register("buffer")
class VideoSaver(BufferFactory):
    def __init__(self, path: str, max_size: int = 300, fps: int = 30, frame_size: tuple = (1280, 720)):
        self.buffer_image: deque[Tuple[int, np.ndarray]] = deque()
        self.buffer_index: Dict[int, np.ndarray] = {}
        self.buffer_log: deque[Tuple[int, np.ndarray]] = deque()
        self.log_index: Dict[int, Dict] = {}
        self.max_size = max_size

        self.fps = fps
        self.frame_size = frame_size
        self.path = os.path.splitext(path)[0]


        self.img_folder = os.path.join(self.path, "images")
        os.makedirs(self.img_folder, exist_ok=True)
        self.clear_img_dir(self.img_folder)
        self.ann_folder = os.path.join(self.path, 'annotations')
        os.makedirs(self.ann_folder, exist_ok=True)
        self.clear_log_dir(self.ann_folder)

        self.frame_reader = FrameReader(path=self.path)
        self.reader_thread: Optional[QThread] = QThread()
        self.frame_reader.moveToThread(self.reader_thread)
        self.reader_thread.start()
        self.async_frame_cache: Dict[int, QImage] = {}

        self.image_flush_thread = QThread()
        self.image_flush_worker = ImageFlushWorker(self)
        self.image_flush_worker.moveToThread(self.image_flush_thread)
        self.image_flush_thread.start()

        self.log_flush_thread = QThread()
        self.log_flush_worker = LogFlushWorker(self)
        self.log_flush_worker.moveToThread(self.log_flush_thread)
        self.log_flush_thread.start()

    def add_image(self, idx: int, frame: np.ndarray):
        self.add_frame(frame, idx)

    def add_ann(self, idx: int, log: dict):
        self.add_log2disk(idx, log)

    def add_frame(self, idx, frame):
        if len(self.buffer_image) >= self.max_size:
            self.flush_images()
        self.buffer_image.append((idx, frame))
        self.buffer_index[idx] = frame

    def add_log2disk(self, idx, log):
        if len(self.buffer_log) >= self.max_size:
            self.flush_logs()
        self.buffer_log.append((idx, log))
        self.log_index[idx] = log

    def load_log(self, number: int) -> Dict[str, List]:
        if number in self.log_index.keys():
            return self.log_index[number]
        else:
            path = f"{self.ann_folder}/{number:09d}.json"
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    def flush(self):
        self.flush_images_async()
        self.flush_logs_async()

    def flush_logs(self):
        log_copy = list(self.buffer_log)
        for idx, log in log_copy:
            self.save_log(log, idx)

        self.buffer_log.clear()
        self.log_index.clear()

    def flush_images(self):
        buffer_copy = list(self.buffer_image)
        for idx, frame in buffer_copy:
            save_img(frame=frame, frame_num=idx, images_dir=self.img_folder)

        self.buffer_image.clear()
        self.buffer_index.clear()

    def _load_image(self, bgr_frame: np.ndarray) -> QImage:
        return convert_to_qimage(bgr_frame)

    def get_image(self, frame_index: int) -> QImage:
        if frame_index in self.buffer_index:
            return self._load_image(self.buffer_index[frame_index])
        else:
            start = time.time()
            frame = self._load_image(self.frame_reader.get_frame(frame_index))
            print("[VideoSaver]: Frame loading from folder index {} elapsed: {}". format(frame_index, time.time() - start))
            return frame

    def _process_frame_disk(
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
        raw_anns = data.get("logs", {}).get("annotations", [])

        boxes = []
        scores = []
        labels = []

        for ann in raw_anns:
            if not isinstance(ann, dict):
                print(f"[Disk] Skipping non-dict annotation: {ann}")
                continue

            bbox = ann.get("bbox")
            score = ann.get("score")
            label = ann.get("label")

            if not bbox or len(bbox) != 4:
                print(f"[Disk] Invalid bbox: {bbox}")
                continue

            try:
                x, y, x2, y2 = map(int, bbox)
                w = x2 - x
                h = y2 - y
                boxes.append([x, y, w, h])
                scores.append(float(score))
                labels.append(str(label))
            except Exception as e:
                print(f"[Disk Buffer] Error parsing annotation {ann}: {e}")
                continue
        return boxes, scores, labels

    def save_log(self, log: dict, frame_num: int) -> None:
        save_log(data=log, frame_num=frame_num, annons_dir=self.ann_folder)

    def clear_log_dir(self, ann_folder: str):
        import os
        [os.remove(os.path.join(ann_folder, f)) for f in os.listdir(ann_folder) if f.endswith(".json")]

    def clear_img_dir(self, img_folder: str):
        import os
        [os.remove(os.path.join(img_folder, f)) for f in os.listdir(img_folder) if f.endswith(".jpg")]


    def flush_images_async(self):
        self.image_flush_worker.flush_requested.emit()

    def flush_logs_async(self):
        self.log_flush_worker.flush_requested.emit()

    def close(self):
        self.flush()

        self.image_flush_thread.quit()
        self.image_flush_thread.wait()
        self.log_flush_thread.quit()
        self.log_flush_thread.wait()

        self.frame_reader.close()