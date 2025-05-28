# sort/association.py
import numpy as np

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), []

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    matched_indices = np.stack(np.unravel_index(np.argsort(-iou_matrix, axis=None), iou_matrix.shape), axis=1)
    unmatched_dets = list(range(len(detections)))
    unmatched_trks = list(range(len(trackers)))
    matches = []
    for d, t in matched_indices:
        if d in unmatched_dets and t in unmatched_trks:
            if iou_matrix[d, t] < iou_threshold:
                continue
            matches.append([d, t])
            unmatched_dets.remove(d)
            unmatched_trks.remove(t)

    return np.array(matches), np.array(unmatched_dets), np.array(unmatched_trks)