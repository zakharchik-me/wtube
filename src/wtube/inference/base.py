from abc import ABC, abstractmethod
import torch

class InferenceEngine(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def load_weights(self, path: str):
        pass
