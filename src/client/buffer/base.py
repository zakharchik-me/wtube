from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from PyQt6.QtGui import QImage
import numpy as np


class BufferFactory(ABC):
    @abstractmethod
    def add_image(self, idx: int, frame: np.ndarray):
        pass

    @abstractmethod
    def add_ann(self, idx: int, log: dict):
        pass

    @abstractmethod
    def load_log(self, frame_index: int) -> dict:
        pass

    @abstractmethod
    def get_image(self, frame_index: int) -> QImage:
        pass

    @abstractmethod
    def load_annotation(self, data: dict) -> Tuple[List[List[int]], List[float], List[str]]:
        pass

    @abstractmethod
    def flush(self):
        pass
