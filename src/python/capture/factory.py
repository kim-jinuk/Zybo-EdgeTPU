# capture/factory.py
from .camera import CameraCapture
from .video import VideoCapture

def create_capture_source(source, output_q, config):
    if isinstance(source, int):
        return CameraCapture(source, output_q, config)
    elif isinstance(source, str):
        return VideoCapture(source, output_q, config)
    else:
        raise ValueError("Unsupported source type")