# processing/enhancers.py
import cv2
import numpy as np
from typing import Callable

class Preprocessor:
    def __call__(self, frame):
        raise NotImplementedError

class LightCLAHE(Preprocessor):
    """
    CLAHE lite : tile 4×4, clipLimit 1.5   ≈ 1.0 ms @640×480
    (기본 CLAHE 8×8, 2.0 → 1.8 ms)
    """
    def __init__(self, clip_limit: float = 1.5, tile: int = 4):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                     tileGridSize=(tile, tile))

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        lab        = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b    = cv2.split(lab)
        l          = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

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

class GaussianDenoise(Preprocessor):
    """
    Simple Gaussian blur denoise  (≈ 0.4 ms @ 640×480)

    ksize : 3·5·7 …  홀수 권장 (3이면 기본 노이즈 제거 + 엣지 보존)
    sigma : 0 → OpenCV가 자동 설정 (권장)
    """
    def __init__(self, ksize: int = 3, sigma: float = 0.0):
        self.ksize, self.sigma = ksize | 1, sigma

    def __call__(self, frame: np.ndarray) -> np.ndarray:          # BGR in/out
        return cv2.GaussianBlur(frame,
                                (self.ksize, self.ksize),
                                self.sigma,
                                borderType=cv2.BORDER_DEFAULT)

class Compose(Preprocessor):
    def __init__(self, steps: list[Preprocessor]):
        self.steps = steps

    def __call__(self, frame):
        for step in self.steps:
            frame = step(frame)
        return frame