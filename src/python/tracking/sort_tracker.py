# tracking/sort_tracker.py
import numpy as np
from sort.sort import Sort as SortImpl  # assuming sort/ is a dependency folder with sort.py

class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.tracker = SortImpl(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

    def update(self, detections: np.ndarray) -> np.ndarray:
        if len(detections) == 0:
            detections = np.empty((0, 5))
        return self.tracker.update(detections)