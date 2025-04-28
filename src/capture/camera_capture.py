# Thread‑1: Camera capture using OpenCV / V4L2.
import cv2, queue, threading, time
from utils.logger import get_logger

class CameraCapture(threading.Thread):
    def __init__(self, cam_id:int, out_q:queue.Queue, cfg:dict):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.q = out_q
        self.cfg = cfg
        self.log = get_logger(f"Camera{cam_id}")
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {cam_id} open failed")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.get("width", 640))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.get("height", 480))
        self.cap.set(cv2.CAP_PROP_FPS,          cfg.get("fps", 30))
        for _ in range(4): self.cap.read()

    def run(self):
        drop = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.log.warning("grab failed"); time.sleep(0.01); continue
            try:
                self.q.put_nowait((time.time(), frame))
            except queue.Full:
                _ = self.q.get_nowait(); self.q.put_nowait((time.time(), frame)); drop+=1
                if drop % 100 == 0:
                    self.log.debug("dropped %d frames", drop)

    def stop(self):
        if self.cap.isOpened():
            self.cap.release()