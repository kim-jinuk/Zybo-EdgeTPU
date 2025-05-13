# src/python/output.py  –  Thread‑3 : Display & Save (FPS HUD always)
# ================================================================
import cv2, queue, threading, math, numpy as np
from utils.logger import get_logger

class Output(threading.Thread):
    """Displays frames and draws tracking boxes with an always‑on FPS HUD.
    Expects queue entries: (timestamp, frame, tracks).
    * timestamp : float (seconds) – capture time of this frame.
    * frame     : np.ndarray      – BGR image.
    * tracks    : list[(x1,y1,x2,y2,id)] – tracker outputs.
    Esc key closes the window.
    """
    def __init__(self, in_q: queue.Queue):
        super().__init__(daemon=True)
        self.q        = in_q
        self.log      = get_logger("Output")
        self.last_ts  = None      # previous frame timestamp
        self.fps_ema  = 0.0       # exponential moving average FPS

    # ------------------------------------------------------------------
    def _update_fps(self, curr_ts: float) -> float:
        """Update EMA FPS from successive capture timestamps."""
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

    # ------------------------------------------------------------------
    def run(self):
        while True:
            cap_ts, frame, tracks = self.q.get()

            # -------- Draw tracking boxes --------------------------------
            for x1, y1, x2, y2, tid in tracks:
                if not all(map(math.isfinite, (x1, y1, x2, y2))):
                    continue
                h, w = frame.shape[:2]
                x1 = int(np.clip(x1, 0, w - 1)); y1 = int(np.clip(y1, 0, h - 1))
                x2 = int(np.clip(x2, 0, w - 1)); y2 = int(np.clip(y2, 0, h - 1))
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{int(tid)}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # -------- FPS HUD ---------------------------------------------
            fps_now = self._update_fps(cap_ts)
            cv2.putText(frame, f"FPS: {fps_now:5.1f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
                        cv2.LINE_AA)

            # -------- Display --------------------------------------------
            cv2.imshow("EO", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

        cv2.destroyAllWindows()
