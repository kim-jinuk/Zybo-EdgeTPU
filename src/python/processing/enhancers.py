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
class FastDenoise(Preprocessor):
    def __init__(self, h: int = 10, template_window_size: int = 7, search_window_size: int = 21):
        self.h, self.tmpl, self.search = h, template_window_size, search_window_size

    def __call__(self, frame: np.ndarray) -> np.ndarray:  # noqa: D401
        return cv2.fastNlMeansDenoisingColored(frame, None, self.h, self.h, self.tmpl, self.search)


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
#  YAML‑style factory                                                         #
###############################################################################

def build_preprocessing(cfg: Dict[str, dict]) -> Preprocessor:
    mapping = {
        "contrast_enhance": "CLAHEContrast",
        "edge_enhance": "UnsharpMask",
        "denoise": "FastDenoise",
        "deblur": "WienerDeblur",
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
    parser.add_argument("--width", type=int, default=1280, help="Capture width (USB cam)")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
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
    processor = Compose([CLAHEContrast(3.0), UnsharpMask(5, 1.2), FastDenoise(7)])

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
