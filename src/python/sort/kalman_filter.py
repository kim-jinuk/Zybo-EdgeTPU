# sort/kalman_filter.py
import numpy as np
class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.time_since_update = 0
        self.hit_streak = 0
        self.history = []
        self.bbox = bbox[:4]

    def update(self, bbox):
        self.bbox = bbox[:4]
        self.time_since_update = 0
        self.hit_streak += 1

    def predict(self):
        self.time_since_update += 1
        return [self.bbox]

    def get_state(self):
        return [self.bbox]