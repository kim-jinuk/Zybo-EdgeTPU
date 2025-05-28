# capture/base.py
import threading
from abc import ABC, abstractmethod
import queue

class CaptureThread(threading.Thread, ABC):
    def __init__(self, output_q: queue.Queue, config: dict):
        super().__init__(daemon=True)
        self.q = output_q
        self.cfg = config
        self._stop_flag = threading.Event()

    def stop(self):
        self._stop_flag.set()

    def stopped(self):
        return self._stop_flag.is_set()

    @abstractmethod
    def run(self):
        pass