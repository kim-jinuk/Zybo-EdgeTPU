"""
Enhancers – Image pre‑processing operators for the Zybo‑EdgeTPU pipeline
=======================================================================
A *single* module that gives you **contrast enhancement, edge sharpening, noise
suppression, de‑blurring**, and **background clutter removal** – every block is
callable and composable.

▶ *Why?*  Better frames → stronger detector scores → steadier tracking IDs.

Linux + USB camera friendly • Plain OpenCV • No exotic deps.

--------------------------------------------------------------------
Quick demo
--------------------------------------------------------------------
```bash
python enhancers.py --video 0 --stack --disp-scale 0.5   # half‑size preview
```
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Type

import cv2
import numpy as np

###############################################################################
#  Base class & registry helper                                                #
###############################################################################

class Preprocessor(Callable):
    """Interface: `processed = processor(frame)` returning **BGR uint8**."""

    def __call__(self, frame: np.ndarray) -> np.ndarray:  # noqa: D401
        raise NotImplementedError


REGISTRY: Dict[str, Type[Preprocessor]] = {}

def register(cls: Type[Preprocessor]):
    REGISTRY[cls.__name__] = cls
    return cls


###############################################################################
#  Operators                                                                   #
###############################################################################

@register
class GammaContrast(Preprocessor):
    """
    Fixed-gamma contrast stretch  (≈ 0.3 – 0.6 ms @ 640×480)

    γ < 1 → 밝기·콘트라스트 ↑   (권장 0.6 ~ 0.8)
    γ > 1 → 밝기·콘트라스트 ↓
    """
    def __init__(self, gamma: float = 0.75):
        inv = 1.0 / max(gamma, 1e-6)
        lut  = np.array([((i / 255.0) ** inv) * 255 for i in range(256)],
                        dtype=np.uint8)
        self._lut = lut.reshape((256, 1))

    def __call__(self, frame: np.ndarray) -> np.ndarray:          # BGR in/out
        return cv2.LUT(frame, self._lut)

@register
class AutoContrast(Preprocessor):
    """
    Percentile stretch (2–98 %)  ≈ 0.5 ms @ 640×480
    - CLAHE보다 3× 빠르고, Gamma보다 '표준편차'가 확실히 늘어남
    """
    def __init__(self, lo_pct: float = 2.0, hi_pct: float = 98.0):
        self.lo, self.hi = lo_pct, hi_pct

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(frame, (self.lo, self.hi))
        scale   = 255.0 / max(hi - lo, 1)
        return cv2.convertScaleAbs(frame, alpha=scale, beta=-lo * scale)
    
@register
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
        
@register
class CLAHEContrast(Preprocessor):
    def __init__(self, clip_limit: float = 2.0, tile_grid: tuple[int, int] = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    def __call__(self, frame: np.ndarray) -> np.ndarray:  # noqa: D401
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


@register
class UnsharpMask(Preprocessor):
    def __init__(self, ksize: int = 5, amount: float = 1.0):
        self.ksize, self.amount = ksize | 1, amount  # ksize must be odd

    def __call__(self, frame: np.ndarray) -> np.ndarray:  # noqa: D401
        blur = cv2.GaussianBlur(frame, (self.ksize, self.ksize), 0)
        return cv2.addWeighted(frame, 1 + self.amount, blur, -self.amount, 0)

@register
class BilateralDenoise(Preprocessor):
    """
    Edge-preserving denoise – bilateral filter
    d            : 필터 직경(픽셀)
    sigma_color  : 색 공간 시그마 (값 차이)
    sigma_space  : 거리 시그마 (좌표 차이)
    """
    def __init__(self, d: int = 5, sigma_color: int = 75, sigma_space: int = 75):
        self.d, self.sc, self.ss = d, sigma_color, sigma_space

    def __call__(self, frame: np.ndarray) -> np.ndarray:            # BGR in/out
        return cv2.bilateralFilter(frame, self.d, self.sc, self.ss)

@register
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

@register
class FastDenoise(Preprocessor):
    def __init__(self, h: int = 10, template_window_size: int = 7, search_window_size: int = 21):
        self.h, self.tmpl, self.search = h, template_window_size, search_window_size

    def __call__(self, frame: np.ndarray) -> np.ndarray:  # noqa: D401
        return cv2.fastNlMeansDenoisingColored(frame, None, self.h, self.h, self.tmpl, self.search)
    
@register
class LaplacianDeblur(Preprocessor):
    """
    Very-light deblur (high-pass sharpen)

    α : 라플라시안 계수 (0.0 ~ 2.0) – 크면 더 강하게 복원
    ks: 라플라시안 커널 크기 (1·3·5 … 홀수) – 3이면 충분
    """
    def __init__(self, alpha: float = 1.0, ks: int = 3):
        self.alpha, self.ks = alpha, ks | 1

    def __call__(self, frame: np.ndarray) -> np.ndarray:      # BGR in/out
        # 1) Gray 라플라시안 추출
        lap = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                            cv2.CV_16S, ksize=self.ks)
        lap = cv2.convertScaleAbs(lap)                        # uint8

        # 2) 원본에 역-가중치 합성 (sharpen)
        sharpen = cv2.addWeighted(
            frame, 1.0 + self.alpha,                         # 앞쪽 ↑
            cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR), -self.alpha,
            0)
        return sharpen

@register
class WienerDeblur(Preprocessor):
    def __init__(self, kernel: int = 9, K: float = 0.01):
        self.kernel_size, self.K = kernel | 1, K
        self.kernel = np.ones((self.kernel_size, self.kernel_size), np.float32) / (
            self.kernel_size**2
        )

    def __call__(self, frame: np.ndarray) -> np.ndarray:  # noqa: D401
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pad_h, pad_w = gray.shape[0] // 2, gray.shape[1] // 2
        padded = cv2.copyMakeBorder(gray, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
        fft = np.fft.fft2(padded)
        kfft = np.fft.fft2(self.kernel, s=padded.shape)
        wiener = np.conj(kfft) / (np.abs(kfft) ** 2 + self.K)
        deconv = np.real(np.fft.ifft2(fft * wiener))
        deconv = np.clip(deconv, 0, 255).astype(np.uint8)[pad_h:-pad_h, pad_w:-pad_w]
        return cv2.cvtColor(deconv, cv2.COLOR_GRAY2BGR)


@register
class ClutterRemoval(Preprocessor):
    def __init__(self, history: int = 50, var_threshold: int = 25, detect_shadows: bool = False):
        self.bg = cv2.createBackgroundSubtractorMOG2(history, var_threshold, detect_shadows)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def __call__(self, frame: np.ndarray) -> np.ndarray:  # noqa: D401
        mask = self.bg.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        return cv2.bitwise_and(frame, frame, mask=mask)


###############################################################################
#  Compose helper                                                             #
###############################################################################

class Compose(Preprocessor):
    def __init__(self, steps: List[Preprocessor]):
        self.steps = steps

    def __call__(self, frame: np.ndarray) -> np.ndarray:  # noqa: D401
        for step in self.steps:
            frame = step(frame)
        return frame

###############################################################################
#  PRESETS helper                                                             #
###############################################################################

PRESETS = {
    "Normal": Compose([
        GammaContrast(gamma=0.8),
    ]),
    "Night": Compose([
        GammaContrast(gamma=0.65),
        GaussianDenoise(ksize=3),
        UnsharpMask(5, 1.0),
    ]),
    "Fog": Compose([
        GammaContrast(gamma=0.75),
        UnsharpMask(5, 1.8),
    ]),
    "Motion": Compose([
        GammaContrast(gamma=0.80),
        LaplacianDeblur(alpha=1.3, ks=3),
        UnsharpMask(5, 0.7),
    ]),
    "IR": Compose([
        GammaContrast(gamma=0.80),
        ClutterRemoval(),
        UnsharpMask(5, 1.0),
    ]),
}

###############################################################################
#  YAML‑style factory                                                         #
###############################################################################

def get_preset(name: str) -> Preprocessor:
    if name not in PRESETS:
        raise ValueError(f"[enhancers] unknown preset: {name}")
    return PRESETS[name]


def build_preprocessing(cfg: Dict[str, dict]) -> Preprocessor:
    """
    cfg 예시
    --------
    preprocess:
      preset: Night                 # ← PRESETS 사용
      # or manual
      contrast_enhance:
        gamma: 0.8
      edge_enhance:
        ksize: 5
        amount: 1.2
    """
    # 1️⃣ preset이 지정돼 있으면 그걸로 끝
    if "preset" in cfg:
        return get_preset(cfg["preset"])

    # 2️⃣ 없으면 키별 매핑으로 수동 조합
    mapping = {
        "contrast_enhance": "GammaContrast",
        "edge_enhance"   : "UnsharpMask",
        "denoise"        : "GaussianDenoise",
        "deblur"         : "LaplacianDeblur",
        "clutter_removal": "ClutterRemoval",
    }
    steps = [REGISTRY[mapping[k]](**v) for k, v in cfg.items()]
    return Compose(steps)

###############################################################################
#  Demo utilities                                                             #
###############################################################################

SUPPORT_FOURCC = ("MJPG", "YUYV", "H264")


def _find_fourcc(cap: cv2.VideoCapture, desired: str | None) -> str | None:
    if desired and cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*desired)):
        return desired
    for cc in SUPPORT_FOURCC:
        if cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cc)):
            ok, _ = cap.read()
            if ok:
                return cc
    return None


def _resize(img: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


###############################################################################
#  Main                                                                       #
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live preview for enhancer stack")
    parser.add_argument("--video", default="0", help="Camera index or video file path")
    parser.add_argument("--width", type=int, default=640, help="Capture width (USB cam)")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--fourcc", default=None, help="Force FOURCC e.g. MJPG")
    parser.add_argument("--stack", action="store_true", help="Show original | processed")
    parser.add_argument("--no-gui", action="store_true", help="Skip imshow (perf profile)")
    parser.add_argument("--disp-scale", type=float, default=1.0, help="Display resize factor")
    args = parser.parse_args()

    # Source open -----------------------------------------------------------
    try:
        src = int(args.video)
        is_cam = True
    except ValueError:
        src = str(Path(args.video).expanduser())
        is_cam = False

    cap = cv2.VideoCapture(src, cv2.CAP_V4L2 if is_cam else 0)
    if not cap.isOpened():
        sys.exit(f"[ERR] Cannot open source {args.video}")

    if is_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        chosen = _find_fourcc(cap, args.fourcc)
        msg = f"[INFO] FOURCC set to {chosen}" if chosen else "[WARN] FOURCC fallback failed"
        print(msg)

    # Default processing chain – tweak as needed
    # processor = Compose([CLAHEContrast(3.0), UnsharpMask(5, 1.2), FastDenoise(7)])
    processor = Compose([GammaContrast(gamma=0.75), 
                         UnsharpMask(5, 1.2),
                         GaussianDenoise(ksize=3),
                         LaplacianDeblur(alpha=1.2, ks=3), 
                         ClutterRemoval()])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed; exiting …")
            break

        processed = processor(frame)
        if args.no_gui:
            continue  # just churn frames for perf testing

        view = np.hstack((frame, processed)) if args.stack else processed
        view = _resize(view, args.disp_scale)
        cv2.imshow("Enhancers", view)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC quits
            break

    cap.release()
    cv2.destroyAllWindows()

