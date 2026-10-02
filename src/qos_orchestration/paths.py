"""Repository paths. Set QOS_RESULTS to measure and analyse into another results directory (a new run)."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
RESULTS = Path(os.environ.get("QOS_RESULTS", ROOT / "results")).resolve()
RAW = RESULTS / "raw"
FIGURES = Path(os.environ.get("QOS_FIGURES", ROOT / "figures")).resolve()
NETEM = ROOT / "netem"
