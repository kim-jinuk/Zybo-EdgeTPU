# pipeline/output.py
import threading
import queue
import cv2
import time
import numpy as np
from utils.logger import get_logger

class OutputThread(threading.Thread):
    def __init__(self, in_q: queue.Queue, config: dict):
        super().__init__(daemon=True)
        self.q = in_q
        self.cfg = config
        self.display_gray = config.get("display_gray", False)
        self.log = get_logger("Output")
        self.last_ts = None
        self.fps_ema = 0.0

    def _update_fps(self, curr_ts):
        if self.last_ts is None:
            self.last_ts = curr_ts
            return 0.0
        delta = curr_ts - self.last_ts
        self.last_ts = curr_ts
        fps = 1.0 / delta if delta > 0 else 0.0
        alpha = 0.1
        self.fps_ema = (1 - alpha) * self.fps_ema + alpha * fps
        return self.fps_ema

    def run(self):
        while True:
            try:
                ts, frame, tracks = self.q.get(timeout=1)
            except queue.Empty:
                continue

            fps = self._update_fps(ts)

            # 유사 IR 열영상 변환 처리 옵션
            if self.display_gray:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # 객체 박스와 ID 출력
            for track in tracks:
                x1, y1, x2, y2, track_id = map(int, track[:5])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # FPS 표시
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Output", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()
