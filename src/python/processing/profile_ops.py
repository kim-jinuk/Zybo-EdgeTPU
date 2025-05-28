# profile_ops.py  ―  원본 + 5개 전처리 블록 개별 타이밍
import time, collections, argparse, cv2, numpy as np
from pathlib import Path
from enhancers import (
    LightCLAHE, CLAHEContrast, 
    GammaContrast, UnsharpMask, GaussianDenoise,
    LaplacianDeblur, ClutterRemoval
)
'''
original
OPS = [
    ("CONTRAST",  GammaContrast(gamma=0.75)),
    ("SHARPEN",   UnsharpMask(5, 1.2)),
    ("DENOISE",   GaussianDenoise(ksize=3)),
    ("DEBLUR",    LaplacianDeblur(alpha=1.2, ks=3)),
    ("CLUTTER",   ClutterRemoval()),
]
'''

OPS = [
    ("CONTRAST",  GammaContrast(gamma=0.75)),
    ("SHARPEN",   LightCLAHE()),
    ("DENOISE",   CLAHEContrast()),
]

def grab_source(src, w, h, fourcc):
    cam = cv2.VideoCapture(src, cv2.CAP_V4L2 if isinstance(src, int) else 0)
    if not cam.isOpened():
        raise RuntimeError(f"cannot open {src}")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if fourcc:
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    return cam

def main():
    ap = argparse.ArgumentParser("measure op latency")
    ap.add_argument("--video", default="0")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fourcc", default=None)
    ap.add_argument("--frames", type=int, default=30,
                    help="샘플 프레임 수")
    args = ap.parse_args()

    try:  src = int(args.video)
    except ValueError: src = str(Path(args.video).expanduser())

    cap   = grab_source(src, args.width, args.height, args.fourcc)
    stats = collections.Counter()
    total = 0

    for i in range(args.frames):
        ok, f = cap.read()
        if not ok: break
        for name, op in OPS:
            t0 = time.perf_counter()
            _  = op(f)
            stats[name] += time.perf_counter() - t0
        total += 1

    cap.release()
    print(f"\n★★ {total} frames 샘플 결과 (640×480) ★★")
    for n, s in stats.items():
        print(f"{n:<8}: {s/total*1000:5.1f} ms")
    fast = 1000 / max(s/total*1000 for s in stats.values())
    print(f"\n**이론적 최저 FPS (가장 느린 블록 기준) ≈ {fast:4.1f}**")

if __name__ == "__main__":
    main()
