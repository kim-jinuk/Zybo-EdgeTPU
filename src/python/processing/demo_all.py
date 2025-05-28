# demo_all.py ── Original + 5 operators 동시 미리보기
'''
usage:
python demo_all.py                          # 기본 USB 캠 640×480
python demo_all.py --video 0 --scale 0.6    # 창 크기 60 %
python demo_all.py --video ./clip.mp4       # 파일도 OK
'''
from pathlib import Path
import argparse, sys, cv2, numpy as np
from enhancers import (
    BilateralDenoise,
    GammaContrast, UnsharpMask, GaussianDenoise,
    LaplacianDeblur, ClutterRemoval
)

# ---------------------------- 설정 -----------------------------
OPS = [
    ("ORIG",   lambda x: x),
    ("CONTR",  GammaContrast(gamma=0.75)),
    ("SHARP",  UnsharpMask(ksize=5, amount=1.2)),
    ("DENOISE", GaussianDenoise(ksize=3)),
    ("DEBLUR", LaplacianDeblur(alpha=1.2, ks=3)),
    ("CLUTTER",ClutterRemoval(history=50, var_threshold=25)),
]
# ---------------------------------------------------------------

def find_fourcc(cap, desired):
    for cc in ([desired] if desired else []) + ["MJPG", "YUYV", "H264"]:
        if cc and cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cc)):
            ok, _ = cap.read()
            if ok: return cc
    return None

def tile(images, cols=3):
    """이미지 리스트 → cols 열 그리드 numpy 배열 반환"""
    rows = (len(images) + cols - 1) // cols
    h, w, c = images[0].shape
    canvas = np.zeros((h*rows, w*cols, c), dtype=images[0].dtype)
    for idx, img in enumerate(images):
        r, c_ = divmod(idx, cols)
        canvas[r*h:(r+1)*h, c_*w:(c_+1)*w] = img
    return canvas

def main():
    ap = argparse.ArgumentParser(description="All-operators preview (2×3 grid)")
    ap.add_argument("--video", default="0",
                    help="Camera index or video file")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fourcc", default=None)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Display resize factor (0.5 = half)")
    args = ap.parse_args()

    # ---------- open source -------------------------------------------------
    try:
        src, is_cam = int(args.video), True
    except ValueError:
        src, is_cam = str(Path(args.video).expanduser()), False

    cap = cv2.VideoCapture(src, cv2.CAP_V4L2 if is_cam else 0)
    if not cap.isOpened():
        sys.exit(f"❌ cannot open {args.video}")

    if is_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        chosen = find_fourcc(cap, args.fourcc)
        print(f"[INFO] FOURCC={chosen or 'default'}")

    names = [n for n,_ in OPS]
    while True:
        ok, frame = cap.read()
        if not ok: break

        frames = [op(frame) for _, op in OPS]
        grid = tile(frames, cols=3)

        if args.scale != 1.0:
            grid = cv2.resize(grid, None, fx=args.scale, fy=args.scale,
                              interpolation=cv2.INTER_AREA)

        cv2.imshow("Original | CONTR | SHARP | DENOISE | DEBLUR | CLUTTER", grid)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
