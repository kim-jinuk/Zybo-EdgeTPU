# processing/enhancers.py
import cv2
import numpy as np
from typing import Callable

class Preprocessor:
    def __call__(self, frame):
        raise NotImplementedError

class GammaContrast(Preprocessor):
    def __init__(self, gamma: float = 1.5):
        self.gamma = gamma
        self.table = np.array([(i / 255.0) ** self.gamma * 255 for i in range(256)]).astype("uint8")

    def __call__(self, frame):
        return cv2.LUT(frame, self.table)

class UnsharpMask(Preprocessor):
    def __init__(self, strength: float = 1.0):
        self.strength = strength

    def __call__(self, frame):
        blur = cv2.GaussianBlur(frame, (0, 0), sigmaX=3)
        return cv2.addWeighted(frame, 1 + self.strength, blur, -self.strength, 0)

class Compose(Preprocessor):
    def __init__(self, steps: list[Preprocessor]):
        self.steps = steps

    def __call__(self, frame):
        for step in self.steps:
            frame = step(frame)
        return frame