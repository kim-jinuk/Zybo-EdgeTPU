import time, collections
class FPSMeter:
    def __init__(self,size:int=60):
        self.buf=collections.deque(maxlen=size)
        self.last=None
    def tick(self):
        now=time.time()
        if self.last is not None:
            self.buf.append(now-self.last)
        self.last=now
    def fps(self):
        return 0.0 if not self.buf else 1.0/(sum(self.buf)/len(self.buf))