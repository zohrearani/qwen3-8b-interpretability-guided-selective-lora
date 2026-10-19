# Research data and provenance

## Pilot data

`pilot/train.csv`, `pilot/validation.csv`, and `pilot/test.csv` are verbatim exports of the 57/12/13 example pilot splits embedded in the published research notebook. The generated `example_id` indexes each original split (starting at 0). The `label` values are 0 = negative and 1 = positive. The notebook has previously used/inspected these data; **the 13-item test set is not an untouched confirmatory test**.

`pilot/activation_patching_pairs.csv` exports the same 40 synthetic clean/corrupt examples as the research notebook. The exported text and targets are unchanged. These are synthetic intervention prompts, not independent natural test samples.

These input examples already existed in the notebook. The CSV files are a reproducibility convenience, not additional training data or new runs. Confirm redistribution rights for any underlying third-party review texts before making the repository public.

## External evaluation

`../results/external/per_example_predictions.csv` contains the **original 1,000-example, 26-configuration evaluation** in wide format: `example_id`, `y_true`, and `pred_<method>_<seed>` / `margin_<method>_<seed>` columns. The `base_none` configuration has no seed. Every method was checked for identical IDs and gold labels.

For reuse of actual review texts, refer to `ParsiAI/snappfood-sentiment-analysis` and the evaluation sampling procedure in the repository's Python script. **Raw SnappFood review text is omitted here** to avoid republishing a third-party dataset without confirming its licensing and privacy terms. The row IDs follow the evaluation's deterministic, fixed-seed sampled ordering. The code uses `HAPPY=1` (positive) and `SAD=0` (negative).

`../results/external/manifest.json` is the saved run manifest from the original evaluation and includes sampling provenance. Values in the per-example file were copied from the author's original prediction artifacts, not recomputed.

## Code and metrics

Other aggregate reported results live under `../results/pilot/` and `../results/external/`.

No model weights, API credentials, personal system paths, or private access tokens are included in this data package.
