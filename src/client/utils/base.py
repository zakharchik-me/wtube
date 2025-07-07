from abc import ABC, abstractmethod
import numpy as np
import torch


class FrameSource(ABC):
    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass
