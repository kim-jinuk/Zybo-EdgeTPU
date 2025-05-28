# capture/camera.py
import cv2
import time
import queue
from .base import CaptureThread
from utils.logger import get_logger

class CameraCapture(CaptureThread):
    def __init__(self, cam_id, output_q, config):
        super().__init__(output_q, config)
        self.cam_id = cam_id
        self.log = get_logger(f"Camera{cam_id}")
        self.cap = cv2.VideoCapture(cam_id)
        self._configure_camera()

    def _configure_camera(self):
        width = self.cfg.get("camera", {}).get("width", 640)
        height = self.cfg.get("camera", {}).get("height", 480)
        fps = self.cfg.get("camera", {}).get("fps", 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {self.cam_id} could not be opened.")

    def run(self):
        while not self.stopped():
            ret, frame = self.cap.read()
            if not ret:
                self.log.warning("Frame grab failed.")
                time.sleep(0.01)
                continue
            ts = time.time()
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put((ts, frame))

        self.cap.release()