# DeblurGAN‑v2 Lite wrapper.
from tflite_runtime.interpreter import Interpreter
import cv2, numpy as np

class Deblurrer:
    def __init__(self, model_path:str, threads:int=2):
        self.interp = Interpreter(model_path=model_path, num_threads=threads)
        self.interp.allocate_tensors()
        self.in_det = self.interp.get_input_details()[0]
        self.out_det = self.interp.get_output_details()[0]
        _, self.in_h, self.in_w, _ = self.in_det['shape']

    def __call__(self, frame):
        inp = cv2.resize(frame, (self.in_w, self.in_h)).astype(np.uint8)[None]
        self.interp.set_tensor(self.in_det['index'], inp)
        self.interp.invoke()
        out = self.interp.get_tensor(self.out_det['index'])[0]
        return cv2.resize(out, (frame.shape[1], frame.shape[0]))