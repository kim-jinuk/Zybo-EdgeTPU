# eval_ops.py ── Original vs. Each Pre-proc : quality metrics
'''
python eval_ops.py                  # USB 캠 640×480, 200프레임
python eval_ops.py --video clip.mp4 --frames 100
'''
from pathlib import Path
import argparse, cv2, numpy as np
from collections import defaultdict

# 우리 모듈
from enhancers import (
    GammaContrast, UnsharpMask, GaussianDenoise,
    LaplacianDeblur, ClutterRemoval
)

###############################################################################
# -- 지표 함수들                                                              #
###############################################################################

def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def rms_contrast(img):
    """RMS contrast = 표준편차(밝기)"""
    return _gray(img).std()

def lap_var(img):
    """Laplacian variance = 선명도 / 에지 강도"""
    return cv2.Laplacian(_gray(img), cv2.CV_64F).var()

def noise_est(img):
    """고주파 잔차 평균절댓값 = 노이즈 레벨 근사"""
    g = _gray(img).astype(np.float32)
    return np.mean(np.abs(g - cv2.GaussianBlur(g, (3,3), 0)))

def clutter_removed(orig, proc, thr=10):
    """원본 대비 마스크 픽셀(밝기>thr) 제거 비율"""
    o = (_gray(orig) > thr).astype(np.uint8)
    p = (_gray(proc)  > thr).astype(np.uint8)
    removed = (o == 1) & (p == 0)
    total = o.sum()
    return removed.sum() / total if total else 0.0

###############################################################################
# -- 블록 & 지표 매핑                                                         #
###############################################################################

OPS = [
    ("CONTRAST",  GammaContrast(gamma=0.75),          lambda o, p: rms_contrast(p) / rms_contrast(o) - 1),
    ("SHARPEN",   UnsharpMask(5, 1.2),         lambda o, p: lap_var(p)      / lap_var(o)      - 1),
    ("DENOISE",   GaussianDenoise(3),            lambda o, p: 1 - noise_est(p)/ noise_est(o)        ),
    ("DEBLUR",    LaplacianDeblur(1.2, 3),       lambda o, p: lap_var(p)      / lap_var(o)      - 1),
    ("CLUTTER",   ClutterRemoval(),            clutter_removed),
]
"""
• CONTRAST  → +% (RMS contrast 향상)  
• SHARPEN   → +% (Laplacian variance ↑)  
• DENOISE   → +% (노이즈 ↓)  
• DEBLUR    → +% (선명도 ↑)  
• CLUTTER   → 0‥1 (제거 비율)  
"""

###############################################################################
# -- 소스 열기                                                                #
###############################################################################

def open_source(src, w, h, fourcc):
    cam = cv2.VideoCapture(src, cv2.CAP_V4L2 if isinstance(src, int) else 0)
    if not cam.isOpened():
        raise RuntimeError(f"cannot open {src}")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if fourcc:
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    return cam

###############################################################################
# -- 메인                                                                     #
###############################################################################

def main():
    ap = argparse.ArgumentParser("quality gain evaluator")
    ap.add_argument("--video", default="0")
    ap.add_argument("--width",  type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fourcc", default=None)
    ap.add_argument("--frames", type=int, default=200,
                    help="평가에 쓸 프레임 수")
    args = ap.parse_args()

    try:     src = int(args.video)
    except: src = str(Path(args.video).expanduser())

    cap = open_source(src, args.width, args.height, args.fourcc)
    gains, n = defaultdict(float), 0

    while n < args.frames:
        ok, frame = cap.read()
        if not ok: break
        for name, op, metric in OPS:
            proc   = op(frame)
            gains[name] += metric(frame, proc)
        n += 1

    cap.release()
    print(f"\n★★ {n} frames, {args.width}×{args.height} ★★")
    for name, tot in gains.items():
        gain = tot / n
        if name == "CLUTTER":
            print(f"{name:<8}: {gain*100:6.1f} %  pixels removed")
        elif name == "DENOISE":
            print(f"{name:<8}: {gain*100:6.1f} %  noise ↓")
        else:
            print(f"{name:<8}: {gain*100:6.1f} %  increase")

if __name__ == "__main__":
    main()
