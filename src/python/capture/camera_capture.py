# Thread‑1: Camera capture using OpenCV / V4L2.
import sys, os
sys.path.append(os.pardir)
import cv2, queue, threading, time
from utils.logger import get_logger

SUPPORT_FOURCC = ("MJPG", "YUYV", "H264")      # 순차 시도

def _find_working_fourcc(cap):
    for cc in SUPPORT_FOURCC:
        if cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cc)):
            # 읽기 테스트
            for _ in range(3):
                if cap.read()[0]:
                    return cc
    return None

class CameraCapture(threading.Thread):
    def __init__(self, cam_id: int, out_q: queue.Queue, cfg: dict):
        super().__init__(daemon=True)
        self.cam_id, self.q, self.cfg = cam_id, out_q, cfg
        self.log = get_logger(f"Camera{cam_id}")

        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {cam_id} open failed")

        # ── 1) 지원 해상도로 조정 ──────────────────────────
        w, h = cfg.get("width", 640), cfg.get("height", 480)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS,          cfg.get("fps", 30))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        # ── 2) 카메라가 먹는 FourCC 찾기 ───────────────────
        fourcc = _find_working_fourcc(self.cap)
        if not fourcc:
            raise RuntimeError("No working FourCC (MJPG/YUYV/H264) found")
        self.log.info("Using FourCC %s", fourcc)

        # 워밍업
        for _ in range(4):
            self.cap.read()

    def run(self):
        drop = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.log.warning("grab failed")
                time.sleep(0.05)
                continue
            try:
                self.q.put_nowait((time.time(), frame))
            except queue.Full:
                self.q.get_nowait()
                self.q.put_nowait((time.time(), frame))
                drop += 1
                if not drop % 100:
                    self.log.debug("dropped %d frames", drop)

    def stop(self):
        self.cap.release()


if __name__ == "__main__":
    """
    `$ python camera_capture.py --cam 0 --width 640 --height 480 --fps 30`
    로 실행하면 실시간 미预 뷰 창이 뜹니다.  
    * q / ESC : 종료
    """
    import argparse, sys

    parser = argparse.ArgumentParser(description="Quick camera-capture self-test")
    parser.add_argument("--cam",    type=int, default=0,   help="Camera index")
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps",    type=int, default=30)
    args = parser.parse_args()

    frame_q = queue.Queue(maxsize=8)
    cam_thr  = CameraCapture(
        cam_id=args.cam,
        out_q=frame_q,
        cfg={"width": args.width, "height": args.height, "fps": args.fps},
    )
    print('here')
    cam_thr.start()
    
    try:
        while True:
            ts, frame = frame_q.get()
            cv2.imshow("CameraCapture-TEST", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):   # ESC or q
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam_thr.stop()
        cv2.destroyAllWindows()
        sys.exit(0)