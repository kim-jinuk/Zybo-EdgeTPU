# Thread‑3 (display/save)
import cv2, queue, threading, time
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
                cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                cv2.putText(frame, f"ID:{int(tid)}", (int(x1),int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imshow("EO", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()