# ByteTrack adapter – minimal wrapper around YOLOX implementation
# ===============================================================
from __future__ import annotations
import numpy as np

try:
    # pip install yolox ; or ensure yolox package in PYTHONPATH
    from yolox.tracker.byte_tracker import BYTETracker as _BT
except ImportError as e:
    raise ImportError("[ByteTrack] YOLOX 패키지가 필요합니다. → pip install yolox") from e

__all__ = ["ByteTracker"]

class ByteTracker:
    """ByteTrack wrapper exposing .update(detections)->tracks.

    • 입력  detections: (N,5) ndarray [x1,y1,x2,y2,score]
    • 출력  tracks    : (M,5) ndarray [x1,y1,x2,y2,track_id]
    """

    def __init__(self, **kwargs):
        # kwargs 그대로 YOLOX BYTETracker 에 전달 (fps, track_thresh 등)
        self.bt = _BT(kwargs if kwargs else {})

    # --------------------------------------------------------
    def update(self, detections: np.ndarray):
        if detections.size == 0:
            return np.empty((0,5), dtype=np.float32)

        # YOLOX BYTETracker expects tlbr+score; we append dummy class id 0
        dets = np.hstack([detections, np.zeros((detections.shape[0],1), np.float32)])
        online_targets = self.bt.update(dets, img_info=(None, None))  # img size not used

        res = []
        for t in online_targets:
            tlbr = t.tlbr  # [x1,y1,x2,y2]
            res.append([*tlbr, t.track_id])
        return np.asarray(res, dtype=np.float32)
