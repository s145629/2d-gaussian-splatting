"""Versioned adapter that swaps only the SAR loss in native train_v2."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_native = importlib.import_module("train_v2")
from sar_geom_loss_v2_v2 import SarGeometryLoss

_native.SarGeometryLoss = SarGeometryLoss
for _name, _value in vars(_native).items():
    if _name not in {
        "__name__",
        "__package__",
        "__loader__",
        "__spec__",
        "__file__",
        "__cached__",
        "SarGeometryLoss",
    }:
        globals()[_name] = _value
globals()["SarGeometryLoss"] = SarGeometryLoss

VERSION = "train_v2_t4_sar_v2"
