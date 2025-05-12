# Thread‑2: End‑to‑end pipeline glue.
import queue, threading
from processing.deblurring import Deblurrer
from processing.super_resolution import SuperResolver
from detection.tpu_detection import TPUDetector
from tracking.sort_tracker import Sort
from utils.logger import get_logger

class Pipeline(threading.Thread):
    def __init__(self, in_q:queue.Queue, out_q:queue.Queue, cfg:dict):
        super().__init__(daemon=True)
        self.in_q, self.out_q = in_q, out_q
        self.log = get_logger("Pipeline")
        #self.deblur = Deblurrer(cfg["deblur_model"])
        #self.sr     = SuperResolver(cfg["sr_model"])
        self.det    = TPUDetector(cfg["det_model"], cfg.get("det_thresh", 0.5))
        self.trk    = Sort()

    def run(self):
        while True:
            ts, frame = self.in_q.get()
            #frame = self.deblur(frame)
            #frame = self.sr(frame)
            dets  = self.det(frame)
            tracks = self.trk.update(dets)
            self.out_q.put((ts, frame, tracks))