# test/test_logger.py
from utils.logger import get_logger

def test_logger_output(caplog):
    log = get_logger("test")
    with caplog.at_level("INFO"):
        log.info("hello")
    assert "hello" in caplog.text