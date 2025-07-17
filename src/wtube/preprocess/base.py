from abc import ABC, abstractmethod
from utils.registry import register
from wtube.utils.instance import PreprocDetData
import numpy as np


@register("preprocess")
class DetPreprocessor(ABC):

    @abstractmethod
    def __call__(self, frame: np.ndarray) -> "PreprocDetData":
        raise NotImplementedError