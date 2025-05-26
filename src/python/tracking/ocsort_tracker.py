# OC-SORT adapter – wrapper for ocsort package
# ============================================
from __future__ import annotations
import numpy as np

try:
    # pip install ocsort
    from ocsort.ocsort import OCSort as _OCSort
except ImportError as e:
    raise ImportError("[OC-SORT] ocsort 패키지가 필요합니다. → pip install ocsort") from e

__all__ = ["OCSort"]

class OCSort:
    """OC-SORT wrapper exposing .update(detections)->tracks.

    • 입력  detections: (N,5) ndarray [x1,y1,x2,y2,score]
    • 출력  tracks    : (M,5) ndarray [x1,y1,x2,y2,track_id]
    """

    def __init__(self, **kwargs):
        self.oc = _OCSort(**kwargs)

    # --------------------------------------------------------
    def update(self, detections: np.ndarray):
        if detections.size == 0:
            return np.empty((0,5), dtype=np.float32)

        outputs = self.oc.update(detections)  # (M,6) x1,y1,x2,y2,score,id
        if outputs.shape[0] == 0:
            return np.empty((0,5), dtype=np.float32)
        return outputs[:, [0,1,2,3,5]].astype(np.float32)
