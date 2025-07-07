from abc import ABC, abstractmethod
import numpy as np
import torch

class Preprocessor(ABC):
    @abstractmethod
    def __init__(self, height: int, width: int):
        pass

    def __call__(self, frame: np.ndarray) -> torch.Tensor:
        pass

class InferenceEngine(ABC):
    @abstractmethod
    def __call__(self, img):
        pass

class Postprocessor(ABC):
    @abstractmethod
    def __call__(self, result):
        pass

class FrameSource(ABC):
    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass
