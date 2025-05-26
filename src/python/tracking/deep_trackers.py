# -*- coding: utf-8 -*-
"""tracking/deep_trackers.py
=================================
Unified *re‑export* module that groups all **deep‑learning‑based MOT trackers**
into a single import path. This keeps the public factory registry tidy while
avoiding code duplication.

Available classes
-----------------
* **Sort**        – Kalman + IoU (lightweight, no DL dependency)
* **DeepSort**    – Sort + CNN Re‑ID embedding
* **ByteTracker** – YOLOX ByteTrack implementation
* **OCSort**      – Occlusion‑aware IoU tracker

Only the class that is actually imported by user code will trigger the heavy
underlying library import, so unused dependencies do **not** slow startup.

Example
-------
```python
from tracking.deep_trackers import ByteTracker
tracker = ByteTracker(track_thresh=0.5)
```
"""
from __future__ import annotations

# ---------------------------------------------------------------------
# Lazy re‑export: simply import the classes from their original modules
# so external code can do `from tracking.deep_trackers import Foo`.
# ---------------------------------------------------------------------
from tracking.sort_tracker import Sort  # type: ignore
from tracking.deepsort_tracker import DeepSort  # type: ignore
from tracking.bytetrack_tracker import ByteTracker  # type: ignore
from tracking.ocsort_tracker import OCSort  # type: ignore

__all__ = [
    "Sort",
    "DeepSort",
    "ByteTracker",
    "OCSort",
]
