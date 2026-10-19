# Interpretability-Guided Selective LoRA for Qwen3-8B

Research code and reported summary results for the paper on **layer-wise QLoRA allocation using linear probing and activation patching**, evaluated on **low-resource Persian sentiment classification**.

## Research question

Do transformer blocks highlighted by linear probes or activation patching make the best locations for parameter-efficient adaptation? We compare full-stack LoRA, a random four-block baseline, probe-selected four-block allocation, patch-selected four-block allocation, and a combined allocation. The base model is also evaluated.

## Repository contents

| Path | Description |
| --- | --- |
| `notebooks/Qwen3_Controlled_Ablations_RTX3060.ipynb` | Controlled pilot: input splits, training, probing, activation patching, five-seed training and evaluations. |
| `scripts/Qwen3_External_Generalization_Cell.py` | Additional external SnappFood test on saved adapters (no retraining). Run **inside** the completed notebook namespace. |
| `scripts/verify_results.py` | Offline structural consistency checks on published summary artifacts. |
| `results/pilot/` | Pilot aggregate, layer selection, seed metrics, paired comparisons. |
| `results/external/` | External 1,000-example evaluation metrics and summary. |

**No model weights, fine-tuned adapters, credentials or downloaded datasets are included.** The main notebook embeds the *small pilot text split* and the *synthetic intervention pairs* used for the recorded study. Reusers should check underlying dataset redistribution conditions before republishing or deriving distributions.

## Environment and setup

- Research setup: NVIDIA RTX 3060, 12 GB VRAM; Qwen3-8B available **locally**.
- Python, CUDA-enabled PyTorch, `transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`, `numpy`, `pandas`, `scipy`, `scikit-learn`.
- A compatible CUDA stack and adequate available GPU memory are required. Exact original package-version pins are not supplied in the available artifacts; **do not present `requirements.txt` as a verified lockfile**.

Example from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
# Install CUDA-enabled PyTorch matching your system first if necessary.
pip install -r requirements.txt
export QWEN3_MODEL_PATH=/absolute/path/to/your/qwen3-8b
jupyter notebook notebooks/Qwen3_Controlled_Ablations_RTX3060.ipynb
```

The notebook also requires `jupyter` or another compatible Jupyter environment. Start in the repository root so that `controlled_ablation_v1/` is created there. **Run cells top to bottom**. Training runs are expensive and consume GPU memory. The full-stack memory preflight must succeed before the controlled comparisons. Do not change the layer count, LoRA rank, or training parameters selectively, since doing so invalidates the comparison.

To run the external generalization stage **after the notebook has finished and adapters are saved**, add and execute a notebook cell:

```python
from pathlib import Path
exec(Path('scripts/Qwen3_External_Generalization_Cell.py').read_text(encoding='utf-8'))
```

This downloads `ParsiAI/snappfood-sentiment-analysis` (on first run), filters normalized duplicates, samples a fixed balanced subset of 1,000 examples, and evaluates existing saved configurations. Ensure that local dataset licensing and network access permit the download.

## Reported results (not independently rerun for this repository package)

| Method | Pilot accuracy (13-example reused test) | External accuracy (1,000 examples) |
| --- | ---: | ---: |
| Base, unadapted | — | 53.50% |
| Full-stack LoRA | 92.31% | 82.54% |
| Random four-block LoRA | 86.15% | 81.10% |
| Probe-selected four blocks | 76.92% | 69.68% |
| Patch-selected four blocks | 76.92% | 70.90% |
| Combined four-block selection | 76.92% | 71.34% |

Figures shown are means across five training seeds for adapted methods; the unadapted model is evaluated once. The pilot test contains only **13 examples** and had been inspected previously, so its metrics should not be interpreted as a fresh confirmatory test. The external results show that the selected layer placements **did not outperform** either full-stack LoRA or the random four-block baseline in this experiment. The full-stack method uses a substantially larger trainable parameter budget.

```bash
python scripts/verify_results.py
```

This check validates only the structure of published result files; it does not retrain or re-evaluate the model.

## Reproducibility and limitations

- Pilot dataset: 57 train / 12 dev / 13 test examples; five training seeds (`42,52,62,72,82`).
- The probe selection is based on training-only five-fold cross-validation on frozen, unadapted representations.
- Patching uses 40 synthetic clean–corrupt pairs at the final token position. That intervention is limited and is **not** a proof of unique causal localization.
- Methods use the same basic adapter configuration (rank 32, alpha 64, q/v/o targets) but different trainable block allocations.
- Full-stack trainable parameters: ~24.77M; four-block allocation: ~2.75M.
- The provided outputs are from prior user-run experiments; this repository packaging process included no new GPU experiment.
- Review the original text-split provenance, synthetic pair validity, variance across layer allocations, and data licenses before making stronger research claims.

## Citation and code availability

After creating a **public** repository, replace `<YOUR_GITHUB_USERNAME>` in your manuscript with your username:

> The source code and aggregate experimental results are available at `https://github.com/<YOUR_GITHUB_USERNAME>/qwen3-8b-interpretability-guided-selective-lora`.

For a long-lived paper link, optionally archive a versioned GitHub release with Zenodo to obtain a DOI. The repository is not publicly published merely because this archive exists.

## Licensing

No open-source license is assigned by this package. Before offering third-party reuse rights, the author should select a license and confirm dataset redistribution permissions. The model itself has separate distribution terms.
