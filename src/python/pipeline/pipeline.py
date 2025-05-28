# pipeline/pipeline.py
import threading
import time
import queue
from utils.logger import get_logger

class PipelineThread(threading.Thread):
    def __init__(self, in_q: queue.Queue, out_q: queue.Queue, preprocessor, detector, tracker):
        super().__init__(daemon=True)
        self.in_q = in_q
        self.out_q = out_q
        self.preprocessor = preprocessor
        self.detector = detector
        self.tracker = tracker
        self.log = get_logger("Pipeline")

    def run(self):
        while True:
            try:
                ts, frame = self.in_q.get(timeout=1)
            except queue.Empty:
                continue

            frame = self.preprocessor(frame)
            dets = self.detector.detect(frame)
            tracks = self.tracker.update(dets)

            if self.out_q.full():
                try:
                    self.out_q.get_nowait()
                except queue.Empty:
                    pass
            self.out_q.put((ts, frame, tracks))
