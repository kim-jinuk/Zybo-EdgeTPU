"""Minimal colored console logger."""
import logging, sys
_FMT = "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s"
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(_FMT))
logging.basicConfig(level=logging.INFO, handlers=[_handler])

def get_logger(name:str):
    return logging.getLogger(name)