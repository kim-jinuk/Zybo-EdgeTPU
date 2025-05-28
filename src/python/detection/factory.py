# detection/factory.py
from .tpu import TPUDetector

def create_detector(cfg):
    model_path = cfg.get("model", "models/model.tflite")
    threshold = cfg.get("threshold", 0.4)
    return TPUDetector(model_path, threshold)