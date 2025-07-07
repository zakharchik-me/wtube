import os
import cv2
import json
from typing import List, Tuple
from glob import glob

from utils.paint import load_class_config, build_palette
from utils.logging import load_log

class Export:
    def __init__(self, source, path: str, output_path: str, conf_thresh: float = 0.3):
        self.path = path
        self.image_dir = os.path.join(path, 'images')
        self.annot_dir = os.path.join(path, 'annotations')
        self.output_path = output_path
        self.conf_thresh = conf_thresh

        self.id_to_label = load_class_config()
        self.palette = build_palette()

        if source is not None:
            self.fps = source.get_fps()
        else:
            self.fps = 30

    def _get_color(self, class_id: int) -> tuple:
        rgb = self.palette[class_id % len(self.palette)]
        return rgb[::-1]

    def _draw_annotations(
        self, image, boxes: List[List[int]], scores: List[float], labels: List[int]
    ):
        for bbox, score, label in zip(boxes, scores, labels):
            if score < self.conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            class_id = int(label)
            label_name = self.id_to_label.get(class_id, str(class_id))
            color = self._get_color(class_id)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            text = f"{label_name}: {score:.2f}"
            cv2.putText(
                image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )

    def _clear_dir(self):
        for root, dirs, files in os.walk(self.path, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.path)

    def run(self):
        image_paths = sorted(glob(os.path.join(self.image_dir, "*.*")))
        if not image_paths:
            raise RuntimeError("No images found!")

        first_image = cv2.imread(image_paths[0])
        height, width = first_image.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))

        for idx, img_path in enumerate(image_paths, start=1):
            image = cv2.imread(img_path)
            if image is None:
                print(f"[Warning] Skipping unreadable image: {img_path}")
                continue

            boxes, scores, labels = load_log(self.annot_dir, idx)
            self._draw_annotations(image, boxes, scores, labels)
            out.write(image)

        out.release()
        self._clear_dir()
        print(f"[Export] Video saved to: {self.output_path}")


if __name__ == "__main__":
    exporter = Export(
        source=None,
        path="/home/daniil/PycharmProjects/Wtube/src/logs/2022_02_17_2_MH1x 05 100fps",
        output_path='/home/daniil/PycharmProjects/Wtube/src/logs/test.mp4'
    )
    exporter.run()