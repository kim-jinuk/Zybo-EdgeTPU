# sort/sort.py
import numpy as np
from .kalman_filter import KalmanBoxTracker
from .association import associate_detections_to_trackers

class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        # 추적된 객체가 detection 없이 유지될 수 있는 최대 프레임 수
        self.max_age = max_age
        # 새로운 객체에 ID를 부여하기 위해 필요한 최소 연속 매칭 횟수
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        # 현재 살아있는 tracker 리스트
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []

        # 다음 bbox 예측
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]         # Kalman filter로 다음 위치 예측
            trk[:4] = pos
            trk[4] = 0
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # IoU 기반 매칭
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold)

        # 매칭된 tracker 업데이트
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d[0], :])

        # 새 tracker 생성
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        ret = []
        for trk in self.trackers:
            d = trk.get_state()[0]
            # 최종 출력 필터링
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
