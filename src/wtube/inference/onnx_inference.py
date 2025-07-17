import torch
import onnxruntime as ort
from .base import DetInference
from utils.registry import register

@register("inference")
class OnnxInference(DetInference):
    def __init__(
        self,
        model: str,
        weights_path: str,
        device: str = 'cpu',
    ):
        super().__init__()
        self.weights_path = weights_path
        self.device = device
        self.session = None
        self.load_weights(weights_path, device)

    def load_weights(self, path: str, device: str = 'cpu'):
        providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy() if self.device == 'cpu' else x.cuda().numpy()
        if x.ndim == 3:
            x = x[None]
        outputs = self.session.run(None, {self.input_name: x})

        if len(outputs) == 1:
            return torch.from_numpy(outputs[0])
        else:
            return tuple(torch.from_numpy(o) for o in outputs)

    def train_step(self, *args, **kwargs):
        raise NotImplementedError("ONNX training не реализован в этом классе.")
