from abc import ABC
import numpy as np
import torch
import cv2
from utils.registry import register


def preprocess(im, new_shape):
    """
    Prepares input image before inference.

    Args:
        im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    else:
        img = im.clone()
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # add border

    return img


@register("preprocess")
class YoloPreprocessor(ABC):
    def __init__(self, height: int = 720, width: int = 720):
        self.height = height
        self.width = width

    def __call__(self, frame: np.ndarray) -> torch.Tensor:
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(frame)}")
        # img = cv2.resize(frame, (self.width, self.height))
        img = preprocess(frame, (self.height, self.width))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        tensor = tensor.unsqueeze(0)
        return tensor
