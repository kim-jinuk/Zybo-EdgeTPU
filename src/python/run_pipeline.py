# Example: run_pipeline.py (simplified)
from utils.config import load_config
from utils.logger import get_logger
from capture.factory import create_capture_source
from processing.factory import build_preprocessing
from detection.factory import create_detector
from tracking.factory import build_tracker
from pipeline.pipeline import PipelineThread
from pipeline.output import OutputThread

import queue

if __name__ == '__main__':
    cfg = load_config("config/pipeline.yaml")
    cap_q = queue.Queue(maxsize=1)
    out_q = queue.Queue(maxsize=1)

    capture = create_capture_source(cfg["source"], cap_q, cfg)
    preprocessor = build_preprocessing(cfg.get("preprocessing"))
    detector = create_detector(cfg["detector"])
    tracker = build_tracker(cfg["tracker"])
    
    pipeline = PipelineThread(cap_q, out_q, preprocessor, detector, tracker)
    output = OutputThread(out_q, cfg)

    capture.start()
    pipeline.start()
    output.start()

    capture.join()
    pipeline.join()
    output.join()
