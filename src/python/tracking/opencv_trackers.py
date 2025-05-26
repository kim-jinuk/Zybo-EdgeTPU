"""tracking/opencv_trackers.py
=================================
다중 객체용 경량 OpenCV 트래커 래퍼
---------------------------------
* **지원 엔진** : KCF · CSRT · MOSSE · MIL · MedianFlow (cv2 ≥ 4.5)
* **멀티 객체** : IoU‑기반 그리디 매칭으로 `N`개 객체를 유지·업데이트
* **의존**      : OpenCV 단일. (scipy 있으면 Hungarian 사용)

📐  Bounding‑box & API
----------------------
* 입력  : `detections` → (N,5) ndarray [x1,y1,x2,y2,score]
* 출력  : `tracks`     → (M,5) ndarray [x1,y1,x2,y2,track_id]

🚦  생명주기 파라미터
---------------------
* `max_age`   : miss 후 보류 프레임 수 (default 10)
* `min_iou`   : 매칭 임계값 (default 0.3)
"""
from __future__ import annotations
import cv2, numpy as np
from typing import List, Dict, Tuple

try:
    from scipy.optimize import linear_sum_assignment  # optional, for Hungarian
    _HUNGARIAN = True
except ImportError:
    _HUNGARIAN = False

# ------------------------------------------------------------
# 1. IoU utility
# ------------------------------------------------------------

def _iou(b1: np.ndarray, b2: np.ndarray) -> float:
    """Compute IoU between two bboxes [x1,y1,x2,y2]."""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)

# ------------------------------------------------------------
# 2. Single object wrapper
# ------------------------------------------------------------

_CTOR_MAP = {
    "kcf":        cv2.TrackerKCF_create,
    "csrt":       cv2.TrackerCSRT_create,
    "mosse":      cv2.TrackerMOSSE_create,
    "mil":        cv2.TrackerMIL_create,
    "medianflow": cv2.TrackerMedianFlow_create,
}

class _Track:
    def __init__(self, bbox: np.ndarray, tid: int, ctor):
        self.id = tid
        self.trk = ctor()
        x1,y1,x2,y2 = bbox
        self.trk.init(np.zeros((1,1,3), dtype=np.uint8), (x1,y1,x2-x1,y2-y1))
        self.bbox = bbox.copy()
        self.age  = 0     # total frames
        self.miss = 0     # consecutive misses

    def predict(self) -> Tuple[bool, np.ndarray]:
        ok, bb = self.trk.update(np.zeros((1,1,3), dtype=np.uint8))
        if ok:
            x,y,w,h = bb
            self.bbox = np.array([x,y,x+w,y+h], dtype=np.float32)
        return ok, self.bbox

# ------------------------------------------------------------
# 3. Multi‑object orchestrator
# ------------------------------------------------------------

class _MultiCVTracker:
    def __init__(self, engine: str, max_age:int=10, min_iou:float=0.3):
        if engine not in _CTOR_MAP:
            raise ValueError(f"Unknown engine {engine}")
        self._ctor = _CTOR_MAP[engine]
        self.max_age = max_age
        self.min_iou = min_iou
        self._next_id = 0
        self.tracks: List[_Track] = []

    # --------------------------------------------------------
    def update(self, detections: np.ndarray):
        M = len(self.tracks); N = detections.shape[0]
        if M == 0:
            for d in detections:
                self._add_track(d[:4])
            return self._collect()

        # 1) predict all
        preds = []
        for t in self.tracks:
            ok, bb = t.predict()
            preds.append(bb if ok else t.bbox)
        preds = np.stack(preds)

        # 2) build IoU matrix
        iou_mat = np.zeros((M,N), dtype=np.float32)
        for i in range(M):
            for j in range(N):
                iou_mat[i,j] = _iou(preds[i], detections[j,:4])

        # 3) match
        if _HUNGARIAN:
            row, col = linear_sum_assignment(-iou_mat)
            matches = [(r,c) for r,c in zip(row,col) if iou_mat[r,c]>=self.min_iou]
        else:
            matches=[]
            unmatched_d=list(range(N))
            for i in range(M):
                j = np.argmax(iou_mat[i])
                if iou_mat[i,j]>=self.min_iou and j in unmatched_d:
                    matches.append((i,j)); unmatched_d.remove(j)
            col = [j for _,j in matches]
            row = [i for i,_ in matches]
        matched_t = set(i for i,_ in matches)
        matched_d = set(j for _,j in matches)

        # 4) update matched tracks
        for t_idx, d_idx in matches:
            self.tracks[t_idx].trk.init(np.zeros((1,1,3), dtype=np.uint8),
                                        tuple(self._to_xywh(detections[d_idx,:4])))
            self.tracks[t_idx].bbox = detections[d_idx,:4].copy()
            self.tracks[t_idx].miss = 0

        # 5) unmatched det → new track
        for j in range(N):
            if j not in matched_d:
                self._add_track(detections[j,:4])

        # 6) unmatched track → age++ / remove if too old
        keep=[]
        for t in self.tracks:
            if t.id in matched_t:
                keep.append(t); continue
            t.miss +=1
            if t.miss <= self.max_age:
                keep.append(t)
        self.tracks = keep

        return self._collect()

    # --------------------------------------------------------
    def _add_track(self, bbox):
        self.tracks.append(_Track(bbox, self._next_id, self._ctor))
        self._next_id +=1

    @staticmethod
    def _to_xywh(b):
        x1,y1,x2,y2=b;return (x1,y1,x2-x1,y2-y1)

    def _collect(self):
        out=[]
        for t in self.tracks:
            out.append([*t.bbox, t.id])
        return np.asarray(out, dtype=np.float32)

# ------------------------------------------------------------
# 4. External classes (aliases for factory)
# ------------------------------------------------------------

class KCFTracker(_MultiCVTracker):
    def __init__(self, **kw): super().__init__("kcf", **kw)
class CSRTTracker(_MultiCVTracker):
    def __init__(self, **kw): super().__init__("csrt", **kw)
class MOSSETracker(_MultiCVTracker):
    def __init__(self, **kw): super().__init__("mosse", **kw)
class MILTracker(_MultiCVTracker):
    def __init__(self, **kw): super().__init__("mil", **kw)
class MedianFlowTracker(_MultiCVTracker):
    def __init__(self, **kw): super().__init__("medianflow", **kw)
