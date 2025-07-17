from .base import DetPreprocessor
from wtube.utils.instance import PreprocDetData
from utils.registry import register
import numpy as np
from typing import Tuple
import cv2
import math


@register("preprocess")
class YoloOnnxPreprocessor(DetPreprocessor):
    model_width: int
    model_height: int

    def __init__(self, *args, **kwargs):
        self.model_width = kwargs.get('model_width') or 640
        self.model_height = kwargs.get('model_height') or 640
        self.tiling = kwargs.get('tiling') or False
        self.scaled = kwargs.get('scaled') or False

    def __call__(
            self,
            frame: np.ndarray,
    ) -> "PreprocDetData":

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32)
        frame /= 255.0

        if self.tiling:
            patches, cords, scales = self.tile_img(frame, min_overlap=0.1)
            if self.scaled:
                s_patches, s_cords, s_scales = self.resize_pad(frame)
                patches += s_patches
                cords += s_cords
                scales += s_scales
        else:
            patches, cords, scales = self.resize_pad(frame)

        data = np.stack(patches, axis=0), np.stack(cords, axis=0), np.stack(scales, axis=0)
        preproc_det_data = PreprocDetData(data, tensor_format='nhwc')
        return preproc_det_data


    def resize_pad(self, frame: np.ndarray) -> Tuple[list, list, list]:
            height, width = frame.shape[:2]
            new_im = np.full((self.model_height, self.model_width, 3), 0.447, dtype=np.float32)
            scale = min(self.model_width / width, self.model_height / height)
            new_img_w = int(width * scale)
            new_img_h = int(height * scale)
            image = cv2.resize(frame, (new_img_w, new_img_h), interpolation=cv2.INTER_LINEAR)
            off_x, off_y = (self.model_width - new_img_w) // 2, (self.model_height - new_img_h) // 2
            new_im[off_y:off_y + image.shape[0], off_x:off_x + image.shape[1], :] = image
            return [new_im], [(-off_y, -off_x, -off_y + self.model_height, -off_x + self.model_width)], [scale]

    def tile_img(self, img: np.ndarray, min_overlap: float) -> Tuple[list, list, list]:
        tiles = []
        cords = []
        scales = []
        offset_x1, offset_x2, offset_y1, offset_y2 = 0, 0, 0, 0
        height, width = img.shape[:2]

        if width <= self.model_width or height <= self.model_height:
            return self.resize_pad(img)

        frame_width = width - offset_x2 - offset_x1
        frame_height = height - offset_y2 - offset_y1

        x_tiles, y_tiles = math.ceil(frame_width / self.model_width), math.ceil(frame_height / self.model_height)
        x_rem, y_rem = x_tiles - frame_width / self.model_width, y_tiles - frame_height / self.model_height

        if x_tiles > 1 and x_rem / (x_tiles - 1) < min_overlap:
            x_tiles += 1
            x_rem = x_tiles - frame_width / self.model_width
        if y_tiles > 1 and y_rem / (y_tiles - 1) < min_overlap:
            y_tiles += 1
            y_rem = y_tiles - frame_height / self.model_height

        x_overlap = x_rem / (x_tiles - 1) * self.model_width if x_tiles > 1 else 0
        y_overlap = y_rem / (y_tiles - 1) * self.model_height if y_tiles > 1 else 0

        x_cord_list = [int(offset_x1 + (self.model_width - x_overlap) * i) for i in range(x_tiles)]
        y_cord_list = [int(offset_y1 + (self.model_height - y_overlap) * i) for i in range(y_tiles)]

        for i in y_cord_list:
            for j in x_cord_list:
                y_min, x_min, y_max, x_max = i, j, min(i + self.model_height, height - offset_y2), min(
                    j + self.model_width, width - offset_x2)
                tiles.append(img[y_min:y_max, x_min:x_max, :])
                cords.append((y_min, x_min, y_max, x_max))
                scales.append(1.)

        return tiles, cords, scales