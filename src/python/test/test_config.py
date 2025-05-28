# test/test_config.py
import tempfile
import yaml
from utils.config import load_config

def test_load_config():
    dummy_cfg = {"hello": "world"}
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.yaml') as tmp:
        yaml.dump(dummy_cfg, tmp)
        tmp.flush()
        loaded = load_config(tmp.name)
    assert loaded["hello"] == "world"