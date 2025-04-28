"""Demonstration entry‑point.

Usage (from project root)::

    python -m python.main --cfg config/pipeline.yaml --source 0

It simply forwards all CLI arguments to ``scripts/run_pipeline.py`` so
there is a single place that implements the full threaded pipeline.
"""
from pathlib import Path
import sys

# Ensure project root is on sys.path when executed as a module
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_pipeline import main as _runner


def main() -> None:  # noqa: D401
    """Delegate to :pyfunc:`scripts.run_pipeline.main`."""
    _runner()


if __name__ == "__main__":
    main()