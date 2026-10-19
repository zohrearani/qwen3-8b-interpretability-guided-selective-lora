# External generalization evaluation for Qwen3_Controlled_Ablations_RTX3060.ipynb
#
# Run as a NEW CELL at the end of your completed notebook, where these already exist:
#   OUT, SEEDS, train_df, val_df, test_df, norm, prefix_ids, load_base, evaluate, cleanup
# Saved LoRA adapter folders from the original training must exist.
#
# Evaluation only: DOES NOT re-train adapters or select layers using external test labels.
# Requires network access once to download ParsiAI/snappfood-sentiment-analysis.
#
# NOTE: Source dataset's label_id is reversed relative to the original experiment:
#   HAPPY => 1 (positive); SAD => 0 (negative).
# Do NOT copy its numeric label_id directly.

from pathlib import Path
from datasets import load_dataset
from peft import PeftModel
import pandas as pd
import numpy as np
import torch
import json

EXTERNAL_DATASET_ID = "ParsiAI/snappfood-sentiment-analysis"
EXTERNAL_SPLIT = "test"
PER_CLASS = 500       # 500 positive + 500 negative; change BEFORE final experiment
SAMPLE_SEED = 20261013
GEN_OUT = Path(OUT) / "external_generalization_snappfood"
GEN_OUT.mkdir(parents=True, exist_ok=True)

raw = load_dataset(EXTERNAL_DATASET_ID, split=EXTERNAL_SPLIT)
frame = pd.DataFrame({
    "text": raw["comment"],
    "label": [{"HAPPY": 1, "SAD": 0}[str(x).upper().strip()] for x in raw["label"]]
})
frame = frame.dropna(subset=["text"]).copy()
frame["text"] = frame["text"].astype(str).str.strip()
frame = frame[frame["text"].str.len() > 0].copy()
frame["normalized"] = frame["text"].map(norm)

# Exclude exact normalized repeats both within external dataset and against all
# original train/validation/test records; near duplicates should also be audited.
original_texts = set().union(
    *[set(part["text"].map(norm)) for part in (train_df, val_df, test_df)]
)
frame = frame.drop_duplicates(subset=["normalized"]).copy()
frame = frame[~frame["normalized"].isin(original_texts)].copy()
counts = frame["label"].value_counts()
assert set(counts.index) == {0, 1}, f"Missing a class: {counts}"
assert int(counts.min()) >= PER_CLASS, (
    f"Need at least {PER_CLASS} unique items/class; available: {counts.to_dict()}"
)
# Fixed, class-balanced, deterministic evaluation subset.
ext = (frame.groupby("label", group_keys=False)
       .sample(n=PER_CLASS, random_state=SAMPLE_SEED)
       .sample(frac=1, random_state=SAMPLE_SEED)
       .reset_index(drop=True)[["text", "label"]])
ext.to_csv(GEN_OUT / "external_test_manifest.csv", index=False, encoding="utf-8-sig")

# Report how much of the review is truncated by the original fixed context length.
# The original notebook's prefix_ids() truncates the review at the token budget.
manifest = {
    "dataset": EXTERNAL_DATASET_ID, "split": EXTERNAL_SPLIT,
    "sample_seed": SAMPLE_SEED, "n_total": len(ext),
    "n_positive": int((ext.label == 1).sum()),
    "n_negative": int((ext.label == 0).sum()),
    "label_mapping": {"HAPPY": 1, "SAD": 0},
    "task": "Binary overall sentiment; electronics -> food-delivery domain shift",
    "note": "No external examples were used for layer selection or adapter training.",
}
(GEN_OUT / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)

methods = ["full_stack", "random_four", "probe_only", "patch_only", "combined"]
runs = []

def record_run(method, seed, model):
    metrics, predictions = evaluate(model, ext)
    predictions.to_csv(
        GEN_OUT / f"predictions_{method}_{seed}.csv", index=False, encoding="utf-8-sig"
    )
    entry = {"method": method, "seed": seed, **metrics}
    runs.append(entry)
    pd.DataFrame(runs).drop(columns=["confusion_matrix"], errors="ignore").to_csv(
        GEN_OUT / "external_results.csv", index=False
    )
    print(method, seed, "accuracy=", round(metrics["accuracy"], 4),
          "macro_f1=", round(metrics["macro_f1"], 4),
          "MCC=", round(metrics["mcc"], 4))

# Evaluate base once; deterministic setup.
model = load_base(training=False)
record_run("base", "none", model)
del model
cleanup()

for method in methods:
    for seed in SEEDS:
        adapter_path = Path(OUT) / "runs" / method / str(seed) / "adapter"
        if not (adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(f"Missing trained adapter: {adapter_path}")
        model = load_base(training=False)
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        model.eval()
        record_run(method, seed, model)
        del model
        cleanup()

summary = (pd.DataFrame(runs)
           .groupby("method")[["accuracy", "macro_f1", "mcc"]]
           .agg(["mean", "std", "min", "max"]))
summary.to_csv(GEN_OUT / "external_summary.csv")
print(summary)
print("Saved to", GEN_OUT.resolve())

# For the final manuscript:
# - Report the entire predeclared set (or a fixed, independently sampled subset),
#   NOT the best-looking subset after inspecting scores.
# - Add paired bootstrap intervals over examples and per-example paired tests.
# - For random_four, report that layer placements vary across seeds.
# - This is cross-domain evaluation, NOT proof of mechanism-guided transfer.
