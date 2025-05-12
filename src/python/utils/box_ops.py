# IoU 계산 및 bbox 헬퍼
import numpy as np

def iou_batch(bb_test, bb_gt):
    """Compute IoU between two arrays of boxes (N,4) & (M,4)."""
    bb_test = np.expand_dims(bb_test, 1)  # N,1,4
    bb_gt   = np.expand_dims(bb_gt, 0)    # 1,M,4

    xx1 = np.maximum(bb_test[...,0], bb_gt[...,0])
    yy1 = np.maximum(bb_test[...,1], bb_gt[...,1])
    xx2 = np.minimum(bb_test[...,2], bb_gt[...,2])
    yy2 = np.minimum(bb_test[...,3], bb_gt[...,3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h

    # 음수 면적 방지
    area_test = np.maximum(0., (bb_test[...,2]-bb_test[...,0])) * \
                np.maximum(0., (bb_test[...,3]-bb_test[...,1]))
    area_gt   = np.maximum(0., (bb_gt[...,2]-bb_gt[...,0])) * \
                np.maximum(0., (bb_gt[...,3]-bb_gt[...,1]))

    union = area_test + area_gt - inter + 1e-6
    iou = inter / union

    # NaN 처리
    return np.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)
