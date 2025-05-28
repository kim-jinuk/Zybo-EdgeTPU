# tracking/factory.py
from .sort_tracker import Sort

REGISTRY = {
    "sort": Sort,
    # "deepsort": DeepSort,
    # "bytetrack": ByteTrack,
    # ... other trackers can be added here
}

def build_tracker(cfg):
    name = cfg.get("name", "sort").lower()
    params = cfg.get("params", {})
    if name not in REGISTRY:
        raise ValueError(f"Unsupported tracker name: {name}")
    return REGISTRY[name](**params)
