from .base import BasePreprocessor
from utils.registry import register
import torch
from skimage.color import rgb2gray
import cv2

@register("preprocess")
class BeesPreprocessor(BasePreprocessor):
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.X = torch.zeros(1, 5, height, width, dtype=torch.float32, device=self.device)

    def __call__(self, frame):
        self.X[0, 0] = self.X[0, 1].clone()
        self.X[0, 1] = self.X[0, 2].clone()
        self.X[0, 2] = self.X[0, 3].clone()
        self.X[0, 3] = self.X[0, 4].clone()

        gray = rgb2gray(frame)
        gray = cv2.resize(gray, (self.width, self.height))
        self.X[0, 4] = torch.from_numpy(gray).float().to(self.device)

        return self.X
