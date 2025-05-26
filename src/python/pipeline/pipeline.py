# Thread‑2: End‑to‑end pipeline glue.
import queue, threading, time
from processing.enhancers import build_preprocessing      # ★ NEW
from detection.tpu_detection import TPUDetector
from tracking.sort_tracker import Sort
from utils.logger import get_logger
from typing import Optional
from tracking.factory import build_tracker

# ------------------------------------------------------------
def _make_preprocessor(pcfg: Optional[dict]):
    """
    YAML 섹션을 받아 Compose(...) 를 만든다.
    - preset 키가 있으면 그대로 build_preprocessing 로 전달
    - 개별 블록이면 enable: false 인 항목만 골라낸다
    - 비어 있으면 아이덴티티 λ
    """
    if not pcfg:                       # None, {}, etc.
        return lambda x: x

    # 1) preset 사용
    if "preset" in pcfg:
        return build_preprocessing(pcfg)

    # 2) 수동 블록 – enable 체크
    active = {}
    for name, params in pcfg.items():
        if not isinstance(params, dict):
            continue                   # 잘못된 타입 방지
        if not params.get("enable", True):
            continue                   # 비활성화
        active[name] = {k: v for k, v in params.items() if k != "enable"}

    return build_preprocessing(active) if active else (lambda x: x)
    
# ------------------------------------------------------------


class Pipeline(threading.Thread):
    def __init__(self, in_q: queue.Queue, out_q: queue.Queue, cfg: dict):
        super().__init__(daemon=True)
        self.in_q, self.out_q = in_q, out_q
        self.log = get_logger("Pipeline")

        self.pre = _make_preprocessor(cfg.get("preprocessing"))
        self.det = TPUDetector(cfg["det_model"], cfg.get("det_thresh", 0.5))
        # self.trk = Sort()
        self.trk = build_tracker(cfg)

    def run(self):
        while True:
            ts, frame = self.in_q.get()

            t0 = time.perf_counter()
            frame = self.pre(frame)
            t1 = time.perf_counter()

            dets = self.det(frame)
            t2 = time.perf_counter()

            tracks = self.trk.update(dets)
            t3 = time.perf_counter()

            self.out_q.put((ts, frame, tracks))
            t4 = time.perf_counter()

            print(f"Pre-processing:{(t1-t0)*1e3:5.1f}  \n"
                f"Detection:{(t2-t1)*1e3:5.1f}\n"
                f"Tracking:{(t3-t2)*1e3:5.1f}\n"
                f"Output:{(t4-t3)*1e3:5.1f}\n")

