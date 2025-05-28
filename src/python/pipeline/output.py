# src/python/output.py  –  Thread‑3 : Display & Save (FPS HUD always, optional pseudo-IR display)
# ================================================================
import cv2, queue, threading, math, numpy as np, yaml, time
from utils.logger import get_logger

class Output(threading.Thread):
    """Displays frames and draws tracking boxes with an always-on FPS HUD.
    Optionally display pseudo-IR (thermal) if `display_gray` is True in config.
    Expects queue entries: (timestamp, frame, tracks).
    * timestamp: float – capture time.
    * frame: H×W×3 BGR image.
    * tracks: list[(x1,y1,x2,y2,id)].
    ESC closes the window."""
    def __init__(self, in_q: queue.Queue, config_path: str = "../../config/pipeline.yaml"):
        super().__init__(daemon=True)
        self.q = in_q
        self.log = get_logger("Output")
        self.last_ts = None
        self.fps_ema = 0.0
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
            self.display_gray = cfg.get('display_gray', False)
        except Exception as e:
            self.log.warning(f"Failed to load config '{config_path}': {e}")
            self.display_gray = False

    def _update_fps(self, curr_ts: float) -> float:
        if self.last_ts is None:
            self.last_ts = curr_ts
            return 0.0
        dt = curr_ts - self.last_ts
        self.last_ts = curr_ts
        if dt <= 0:
            return self.fps_ema
        inst_fps = 1.0 / dt
        self.fps_ema = 0.9 * self.fps_ema + 0.1 * inst_fps if self.fps_ema else inst_fps
        return self.fps_ema

    def run(self):
        while True:
            cap_ts, frame, tracks = self.q.get()

            # Optional pseudo-IR display
            if self.display_gray:
                # 1) BGR → Gray
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # 2) CLAHE 적용 (contrast enhancement)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                gray_eq = clahe.apply(gray)
                # 3) Gray → BGR (다른 코드와 통일시키기 위함)
                frame = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

            # Draw tracking boxes
            h, w = frame.shape[:2]
            for x1, y1, x2, y2, tid in tracks:
                if not all(map(math.isfinite, (x1, y1, x2, y2))):
                    continue
                xi1 = int(np.clip(x1, 0, w - 1)); yi1 = int(np.clip(y1, 0, h - 1))
                xi2 = int(np.clip(x2, 0, w - 1)); yi2 = int(np.clip(y2, 0, h - 1))
                if xi2 > xi1 and yi2 > yi1:
                    cv2.rectangle(frame, (xi1, yi1), (xi2, yi2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{int(tid)}", (xi1, yi1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # FPS HUD
            fps = self._update_fps(cap_ts)
            cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("EO", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()
