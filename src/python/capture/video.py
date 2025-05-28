# capture/video.py
import cv2
import time
from .base import CaptureThread
from utils.logger import get_logger

class VideoCapture(CaptureThread):
    def __init__(self, video_path, output_q, config):
        super().__init__(output_q, config)
        self.video_path = video_path
        self.log = get_logger("VideoFile")
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Video file {video_path} could not be opened.")

    def run(self):
        while not self.stopped():
            ret, frame = self.cap.read()
            if not ret:
                self.log.info("End of video.")
                break
            ts = time.time()
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put((ts, frame))
            time.sleep(1.0 / self.cap.get(cv2.CAP_PROP_FPS))

        self.cap.release()