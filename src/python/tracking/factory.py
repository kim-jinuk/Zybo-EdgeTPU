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
import importlib
from typing import Dict, Tuple, Callable, Any

# (1) registry:  alias → (module_path, class_name)
_REGISTRY: Dict[str, Tuple[str, str]] = {
    "sort":      ("tracking.sort_tracker",      "Sort"),
    "deepsort":  ("tracking.deepsort_tracker",  "DeepSort"),
    "bytetrack": ("tracking.bytetrack_tracker", "ByteTracker"),
    "ocsort":    ("tracking.ocsort_tracker",    "OCSort"),
}


def _dynamic_import(module_path: str, class_name: str):
    """늦은 import로 불필요한 의존 로딩 최소화."""
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_tracker(cfg: Dict[str, Any]):
    """Factory 함수

    Args:
        cfg (dict): 전체 pipeline.yaml 파싱 결과.

    Returns:
        tracker instance (duck‑typed): .update(ndarray Nx5) → ndarray Mx5  (x1,y1,x2,y2,id)
    """
    t_cfg = cfg.get("tracker", {})
    name  = str(t_cfg.get("name", "sort")).lower()
    params = t_cfg.get("params", {})

    if name not in _REGISTRY:
        raise ValueError(f"[TrackerFactory] 지원하지 않는 tracker '{name}'. Registry = {list(_REGISTRY)}")

    module_path, class_name = _REGISTRY[name]
    try:
        TrackerCls = _dynamic_import(module_path, class_name)
    except ModuleNotFoundError as e:
        raise ImxcdwsewwrvportError(f"[TrackerFactory] '{name}' import 실패: {e}. 모듈 설치 여부 확인!")

    # 트래커마다 __init__ 시그니처가 다르므로 **params 전달
    return TrackerCls(**params)
