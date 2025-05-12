# Kalman + IoU‑Hungarian (SORT)
import numpy as np, time
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from utils.box_ops import iou_batch

class KalmanBoxTracker:
    """Represents the internal state of individual tracked objects."""
    count = 0
    def __init__(self, bbox):
        # x,y,s,r velocity + position (similar to original SORT)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            bbox = [0, 0, 1, 1]  # 최소 크기 보정
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # State: [cx, cy, s, r, vx, vy, vs]
        self.kf.F = np.eye(7)
        for i in range(4):
            self.kf.F[i, i+3] = 1
        self.kf.H = np.zeros((4,7))
        self.kf.H[:4, :4] = np.eye(4)
        self.kf.R *= 0.01
        self.kf.P *= 10.
        self.kf.Q *= 0.01
        cx = (bbox[0]+bbox[2])/2
        cy = (bbox[1]+bbox[3])/2
        s  = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        r  = (bbox[2]-bbox[0])/(bbox[3]-bbox[1]+1e-6)
        self.kf.x[:4] = np.array([cx,cy,s,r]).reshape(4,1)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def update(self, bbox):
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            bbox = [0, 0, 1, 1]  # 최소 크기 보정
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        cx = (bbox[0]+bbox[2])/2
        cy = (bbox[1]+bbox[3])/2
        s  = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        r  = (bbox[2]-bbox[0])/(bbox[3]-bbox[1]+1e-6)
        self.kf.update(np.array([cx,cy,s,r]).reshape(4,1))

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def get_state(self):
        cx, cy, s, r = self.kf.x[:4].reshape(-1)
        if not np.isfinite(s * r) or s <= 0 or r <= 0:
            w = h = 0.0
        else:
            w = np.sqrt(s * r)
            h = s / (w + 1e-6)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

class Sort:
    def __init__(self, max_age:int=10, min_hits:int=3, iou_thresh:float=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        """detections: ndarray Nx5 (x1,y1,x2,y2,score)"""
        self.frame_count += 1
        trks = np.zeros((len(self.trackers),4))
        to_del = []
        for t,tracker in enumerate(self.trackers):
            pos = tracker.predict()
            trks[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)
            trks = np.delete(trks, t, axis=0)

        matched, unmatched_dets, unmatched_trks = self._associate(detections, trks)

        # update matched trackers with assigned detections
        for t_idx, d_idx in matched:
            self.trackers[t_idx].update(detections[d_idx, :4])

        # add new trackers for unmatched detections
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[i,:4]))

        # remove dead trackers
        ret = []
        for trk in reversed(self.trackers):
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)
            elif trk.hits >= self.min_hits or self.frame_count <= self.min_hits:
                ret.append(np.concatenate((trk.get_state(), [trk.id])))
        return np.array(ret)

    def _associate(self, dets, trks):
        if len(trks) == 0 or len(dets) == 0:
            return [], np.arange(len(dets)), np.arange(len(trks))
        
        iou_mat = iou_batch(trks, dets[:, :4])
        
        # NaN 방지 처리
        iou_mat = np.nan_to_num(iou_mat, nan=-1e5)

        matched_indices = linear_sum_assignment(-iou_mat)
        matched_indices = np.asarray(matched_indices).T
        unmatched_dets = np.setdiff1d(np.arange(len(dets)), matched_indices[:, 1])
        unmatched_trks = np.setdiff1d(np.arange(len(trks)), matched_indices[:, 0])

        matches = []
        for t_idx, d_idx in matched_indices:
            if iou_mat[t_idx, d_idx] < self.iou_thresh:
                unmatched_dets = np.append(unmatched_dets, d_idx)
                unmatched_trks = np.append(unmatched_trks, t_idx)
            else:
                matches.append([t_idx, d_idx])
        
        return np.asarray(matches), unmatched_dets, unmatched_trks


