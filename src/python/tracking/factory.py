# tracking/factory.py – 1‑stop factory to instantiate any tracker listed in config/pipeline.yaml
# -------------------------------------------------------------------------------------------
# ⚙️  사용법 (pipeline 내부)
# -------------------------------------------------------------------------------------------
#   from tracking.factory import build_tracker
#   tracker = build_tracker(cfg)  # cfg: dict loaded from pipeline.yaml
# -------------------------------------------------------------------------------------------
#   pipeline.yaml 예시
#   ------------------
#   tracker:
#     name: bytetrack          # sort | deepsort | bytetrack | ocsort ...
#     params:                  # (옵션) 각 트래커별 키워드 인자 → dict
#       track_thresh: 0.5
#       match_thresh: 0.8
# -------------------------------------------------------------------------------------------
#   신규 트래커 추가 시
#   --------------------
#   1) tracking/<foo>_tracker.py 에 클래스 구현 (BaseTracker duck‑type: update(ndarray)->ndarray)
#   2) _REGISTRY["foo"] = (import_path, class_name)
# -------------------------------------------------------------------------------------------
from __future__ import annotations
import importlib, sys, difflib
from typing import Dict, Tuple, Any
from functools import lru_cache

# ---------------------------------------------------------------------
# 1. Alias → (module_path, class_name)
# ---------------------------------------------------------------------
_REGISTRY: Dict[str, Tuple[str, str]] = {
    # Deep trackers ----------------------------------------------------
    "sort":      ("tracking.deep_trackers", "Sort"),
    "deepsort":  ("tracking.deep_trackers", "DeepSort"),
    "bytetrack": ("tracking.deep_trackers", "ByteTracker"),
    "ocsort":    ("tracking.deep_trackers", "OCSort"),

    # OpenCV trackers --------------------------------------------------
    "kcf":        ("tracking.opencv_trackers", "KCFTracker"),
    "csrt":       ("tracking.opencv_trackers", "CSRTTracker"),
    "mosse":      ("tracking.opencv_trackers", "MOSSETracker"),
    "mil":        ("tracking.opencv_trackers", "MILTracker"),
    "medianflow": ("tracking.opencv_trackers", "MedianFlowTracker"),
}

# ---------------------------------------------------------------------
# 2. Import helper with LRU‑cache
# ---------------------------------------------------------------------
@lru_cache(maxsize=None)
def _import_class(module_path: str, class_name: str):
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

# ---------------------------------------------------------------------
# 3. Public factory API
# ---------------------------------------------------------------------

def build_tracker(cfg: Dict[str, Any]):
    t_cfg   = cfg.get("tracker", {})
    alias   = str(t_cfg.get("name", "sort")).lower()
    params  = t_cfg.get("params", {})

    if alias not in _REGISTRY:
        hint = difflib.get_close_matches(alias, _REGISTRY.keys(), n=1)
        raise ValueError(f"[TrackerFactory] Unknown tracker '{alias}'. Did you mean {hint}?" if hint else
                         f"[TrackerFactory] Unknown tracker '{alias}'.")

    module_path, class_name = _REGISTRY[alias]
    try:
        TrackerCls = _import_class(module_path, class_name)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"[TrackerFactory] '{alias}' import 실패 → {e}\n"
            "필요한 pip 패키지를 설치했는지 확인하세요.") from e

    return TrackerCls(**params)

# ---------------------------------------------------------------------
# 4. CLI helper
# ---------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        print("\n[TrackerFactory] Registry & import status:\n")
        for k,(m,c) in _REGISTRY.items():
            try:
                _import_class(m,c)
                status = "✓"
            except Exception as e:
                status = f"✗  ({type(e).__name__})"
            print(f"  {k:<11} → {m}.{c:<18} {status}")
    else:
        print("usage: python -m tracking.factory list")
