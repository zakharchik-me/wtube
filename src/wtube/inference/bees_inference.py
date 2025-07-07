from .base import InferenceEngine
from utils.registry import register, REGISTRY
import torch

@register("inference")
class BeesInference(InferenceEngine):
    def __init__(self, model, input_channels=5, out_channels=1, weights_path=None):
        if isinstance(model, str):
            model_cls = REGISTRY["model"][model]
        else:
            model_cls = model

        self.model = model_cls(
            input_channels=input_channels,
            out_channels=out_channels,
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()

        if weights_path:
            self.load_weights(weights_path)

    def __call__(self, x):
        with torch.no_grad():
            x = x.to(self.device)
            return self.model(x)

    def load_weights(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)

def test() -> None:
    print("Imported well")

if __name__ == "__main__":
    test()