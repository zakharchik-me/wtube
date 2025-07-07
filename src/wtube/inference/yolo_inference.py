import torch
from ultralytics import YOLO
from .base import InferenceEngine
from utils.registry import register

@register("inference")
class YoloInference(InferenceEngine):
    def __init__(
        self,
        model: str,
        weights_path: str,
        device: str = 'cuda:0',
    ):
        super().__init__()
        self.weights_path = weights_path
        self.device = device
        self.model: YOLO = None

        self.load_weights(weights_path)

    def load_weights(self, path: str):
        self.model = YOLO(path)
        self.model.to(self.device)
        self.model.overrides.update({
            'device': self.device,
        })

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        results = self.model(x, verbose=False)

        return results

    def train_step(self, *args, **kwargs):
        raise NotImplementedError("YOLOv8 training не реализован в этом классе.")
