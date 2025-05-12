# Thread‑3 (display/save)
import cv2, queue, threading, time, math
import numpy as np
from utils.logger import get_logger

class Output(threading.Thread):
    def __init__(self, in_q:queue.Queue):
        super().__init__(daemon=True)
        self.q = in_q
        self.log = get_logger("Output")

    def run(self):
        while True:
            ts, frame, tracks = self.q.get()
            # draw tracks
            for x1,y1,x2,y2,tid in tracks:
                
                def _valid(v): return math.isfinite(v)

                if all(_valid(v) for v in (x1, y1, x2, y2)):
                    h, w = frame.shape[:2]
                    x1 = int(np.clip(x1, 0, w-1))
                    y1 = int(np.clip(y1, 0, h-1))
                    x2 = int(np.clip(x2, 0, w-1))
                    y2 = int(np.clip(y2, 0, h-1))
                    if x2 > x1 and y2 > y1:            # 실제 영역이 있을 때만
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                else:
                    # 좌표가 NaN/inf면 그 bbox는 건너뜀
                    continue
                cv2.putText(frame, f"ID:{int(tid)}", (int(x1),int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imshow("EO", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()