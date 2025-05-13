# Thread‑2: End‑to‑end pipeline glue.
import queue, threading
from processing.enhancers import build_preprocessing      # ★ NEW
from processing.deblurring import Deblurrer
from processing.super_resolution import SuperResolver
from detection.tpu_detection import TPUDetector
from tracking.sort_tracker import Sort
from utils.logger import get_logger

# ------------------------------------------------------------
def _make_preprocessor(pcfg: dict):
    """Build Compose(...) from cfg['preprocessing'] with enable flag."""
    if not pcfg:                           # 블록이 없으면 그대로
        return lambda x: x
    active = {
        name: {k: v for k, v in params.items() if k != "enable"}
        for name, params in pcfg.items()
        if params.get("enable", True)
    }
    return build_preprocessing(active) if active else (lambda x: x)
# ------------------------------------------------------------


class Pipeline(threading.Thread):
    def __init__(self, in_q:queue.Queue, out_q:queue.Queue, cfg:dict):
        super().__init__(daemon=True)
        self.in_q, self.out_q = in_q, out_q
        self.log = get_logger("Pipeline")
        self.pre = _make_preprocessor(cfg.get("preprocessing", {}))
        #self.deblur = Deblurrer(cfg["deblur_model"])
        #self.sr     = SuperResolver(cfg["sr_model"])
        self.det    = TPUDetector(cfg["det_model"], cfg.get("det_thresh", 0.5))
        self.trk    = Sort()

    def run(self):
        while True:
            ts, frame = self.in_q.get()
            frame = self.pre(frame)
            #frame = self.deblur(frame)
            #frame = self.sr(frame)
            dets  = self.det(frame)
            tracks = self.trk.update(dets)
            self.out_q.put((ts, frame, tracks))


