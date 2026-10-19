#!/usr/bin/env python3
"""Lightweight offline consistency checks for published summary artifacts."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pilot = ROOT / "results" / "pilot"
external = ROOT / "results" / "external"
selections = json.loads((pilot / "selection.json").read_text(encoding="utf-8"))
for method in ("probe_only", "patch_only", "combined"):
    assert len(selections[method]) == 4
    assert len(set(selections[method])) == 4
    assert all(isinstance(layer, int) and 0 <= layer < 36 for layer in selections[method])

with (pilot / "aggregate.csv").open(newline="", encoding="utf-8") as f:
    pilot_rows = list(csv.DictReader(f))
with (external / "external_results.csv").open(newline="", encoding="utf-8") as f:
    external_rows = list(csv.DictReader(f))
methods = {"combined", "full_stack", "patch_only", "probe_only", "random_four"}
assert {row["method"] for row in pilot_rows} == methods
assert {row["method"] for row in external_rows} == methods | {"base"}
assert len(external_rows) == 26  # 5 methods x 5 seeds, plus base
assert all(int(row["n"]) == 1000 for row in external_rows)
assert all(0 <= float(row["accuracy"]) <= 1 for row in external_rows)
assert all(0 <= float(row["accuracy_mean"]) <= 1 for row in pilot_rows)
print("Published selection and summary artifacts passed structural checks.")
print("This does not reproduce or validate GPU training results.")
