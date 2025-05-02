"""Entry point – wires threads and launches GUI."""
# python scripts/run_pipeline.py --cfg config/pipeline.yaml --source 0
import argparse, queue, yaml, signal, sys, os, threading
sys.path.append(os.pardir)
from pathlib import Path
from capture.camera_capture import CameraCapture
from pipeline.pipeline import Pipeline
from pipeline.output import Output
from utils.logger import get_logger


def load_cfg(path: str | os.PathLike):
    if not Path(path).exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Zybo EO real‑time pipeline runner")
    parser.add_argument("--cfg", default="config/pipeline.yaml", help="YAML config file")
    parser.add_argument("--source", default="0", help="Camera ID (int) or video file path")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    log = get_logger("Main")

    cap_q = queue.Queue(maxsize=cfg.get("queue", 4))
    out_q = queue.Queue(maxsize=cfg.get("queue", 4))

    # ----- Camera / File source select ---------------------------------
    try:
        cam_id = int(args.source)
        log.info(f"Opening camera {cam_id}")
        cam_th = CameraCapture(cam_id, cap_q, cfg["camera"])
    except ValueError:
        # treat as video file path
        from capture.camera_capture import cv2
        class VideoFileCapture(CameraCapture):
            def __init__(self, path, out_q, cfg):
                self.cfg = cfg; self.q = out_q
                self.log = get_logger("VideoFile")
                self.cap = cv2.VideoCapture(str(path))
                if not self.cap.isOpened():
                    raise RuntimeError(f"Cannot open video file {path}")
                super(threading.Thread, self).__init__(daemon=True)  # bypass CameraCapture init
        log.info(f"Opening video file {args.source}")
        cam_th = VideoFileCapture(args.source, cap_q, cfg["camera"])

    # ----- Launch threads ----------------------------------------------
    cam_th.start()
    Pipeline(cap_q, out_q, cfg).start()
    Output(out_q).start()

    # ----- Graceful shutdown -------------------------------------------
    def _sigint_handler(sig, frame):
        log.info("Ctrl‑C caught – shutting down.")
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigint_handler)
    signal.pause()


if __name__ == "__main__":
    main()