import torch
from abc import ABC
import numpy as np
from torchvision.ops import nms
from collections import namedtuple
from utils.registry import register

DetectionResult = namedtuple("DetectionResult", ["boxes", "scores", "labels"])

@register("postprocess")
class OnnxPostprocessor(ABC):
    def __init__(
        self,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        device: str = "cpu",
    ):
        super().__init__()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device

    def __call__(self, batch, raw_results, frame):
        if isinstance(raw_results, (tuple, list)):
            preds = raw_results[0]
        else:
            preds = raw_results

        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds)
        preds = preds.to(self.device)

        preds = preds.squeeze(0)
        boxes = preds[:4, :].T  #
        class_probs = preds[4:, :]

        scores, labels = class_probs.max(0)

        keep = scores > self.conf_thres
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if boxes.shape[0] > 0:
            keep_idx = nms(boxes, scores, self.iou_thres)
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]
            labels = labels[keep_idx]

        all_boxes = boxes.cpu().tolist()
        all_scores = scores.cpu().tolist()
        all_labels = labels.cpu().tolist()

        return batch, DetectionResult(
            boxes=all_boxes,
            scores=all_scores,
            labels=all_labels
        )
