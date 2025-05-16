# profile_presets.py ── Zybo-EdgeTPU 전처리 Preset latency checker
import cv2, time, argparse, collections, numpy as np
from pathlib import Path

# 우리 모듈
from enhancers import (
    GammaContrast, UnsharpMask, GaussianDenoise,
    LaplacianDeblur, ClutterRemoval, Compose
)

# ──────────────────────────────────────────────────────────────
#  프리셋 정의 (A~F)  – 수정해도 OK
# ──────────────────────────────────────────────────────────────
PRESETS = {
    "A_day": Compose([
        GammaContrast(),
        GaussianDenoise(),
        LaplacianDeblur(),
        UnsharpMask(),
        ClutterRemoval(),
    ]),
    "B_night": Compose([
        GammaContrast(gamma=0.65),
        GaussianDenoise(ksize=3),
        UnsharpMask(5, 1.0),
    ]),
    "C_fog": Compose([
        GammaContrast(gamma=0.75),
        UnsharpMask(5, 1.8),
    ]),
    "D_motion": Compose([
        GammaContrast(gamma=0.80),
        LaplacianDeblur(alpha=1.3, ks=3),
        UnsharpMask(5, 0.7),
    ]),
    "E_ir": Compose([
        GammaContrast(gamma=0.80),
        ClutterRemoval(),
        UnsharpMask(5, 1.0),
    ]),
    "F_ultra": Compose([
        GammaContrast(gamma=0.8),
    ]),
}

# ──────────────────────────────────────────────────────────────
def open_source(src, w, h, fourcc):
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2 if isinstance(src, int) else 0)
    if not cap.isOpened(): raise RuntimeError(f"cannot open {src}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    return cap

def run_bench(preset, cap, num_frames):
    total = 0.0
    for _ in range(num_frames):
        ok, f = cap.read()          # ← 프레임 먼저 확보 (대기 포함)
        if not ok: break
        t0 = time.perf_counter()    # ← 여기서부터 전처리만!
        _  = preset(f)
        total += time.perf_counter() - t0
    done = max(1, num_frames)
    return total / done * 1000      # ms/frame (pure processing time)


# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser("preset latency profiler")
    ap.add_argument("--video", default="0")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fourcc", default=None)
    ap.add_argument("--frames", type=int, default=200,
                    help="샘플 프레임 수")
    args = ap.parse_args()

    try: src = int(args.video)
    except ValueError: src = str(Path(args.video).expanduser())

    results = {}
    for name, preset in PRESETS.items():
        cap = open_source(src, args.width, args.height, args.fourcc)
        ms   = run_bench(preset, cap, args.frames)
        cap.release()
        results[name] = ms

    print(f"\n★★ {args.frames} frames, {args.width}×{args.height} ★★")
    for n, ms in results.items():
        fps = 1000.0 / ms
        print(f"{n:<9}: {ms:6.2f} ms  ({fps:5.1f} FPS)")

if __name__ == "__main__":
    main()
