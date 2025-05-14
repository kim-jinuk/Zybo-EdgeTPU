# Edge‑TPU MobileNet‑SSD object detection.
import time
from pycoral.utils.edgetpu import make_interpreter, list_edge_tpus
from pycoral.adapters import common, detect
import numpy as np, cv2

class TPUDetector:
    def __init__(self, model_path:str, thresh:float=0.5):
        self.interp = make_interpreter(model_path, device=list_edge_tpus()[0]['path'])
        self.interp.allocate_tensors()
        self.thresh = thresh
        _, self.h, self.w, _ = self.interp.get_input_details()[0]['shape']

    def __call__(self, frame):
        print(self.interp._delegates)
        img = cv2.resize(frame, (self.w, self.h))
        common.set_input(self.interp, img)
        t0 = time.perf_counter()
        self.interp.invoke()
        print(f"invoke:{(time.perf_counter()-t0)*1e3:5.1f}")
        objs = detect.get_objects(self.interp, self.thresh)
        res = []
        sx, sy = frame.shape[1]/self.w, frame.shape[0]/self.h
        for o in objs:
            box = o.bbox
            res.append([box.xmin*sx, box.ymin*sy, box.xmax*sx, box.ymax*sy, o.score])
        return np.asarray(res)