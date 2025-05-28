# detection/tpu.py
import numpy as np
import cv2
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.adapters.common import set_input
from .base import Detector

class TPUDetector(Detector):
    def __init__(self, model_path, threshold=0.4):
        # tflite 모델 로딩
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.threshold = threshold
        self.size = input_size(self.interpreter)

    def detect(self, frame):
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, self.size)
        # numpy 배열 → Edge TPU 인터프리터에 넣음
        set_input(self.interpreter, resized)
        # 실제 추론 수행 (Edge TPU에서 실행됨)
        self.interpreter.invoke()
        objs = get_objects(self.interpreter, self.threshold)

        results = []
        for obj in objs:
            x0, y0, x1, y1 = obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax
            sx, sy = w / self.size[0], h / self.size[1]
            results.append([x0 * sx, y0 * sy, x1 * sx, y1 * sy, obj.score])

        return np.array(results)
