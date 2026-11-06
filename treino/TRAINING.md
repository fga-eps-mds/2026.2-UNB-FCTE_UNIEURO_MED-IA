# Reproducing the paper: training guide

Reproduction of [An Explainable Self-Attention Deep Neural Network for Detecting Mild
Cognitive Impairment Using Multi-input Digital Drawing Tasks](https://www.biorxiv.org/content/10.1101/2021.12.15.472738v1.full.pdf)
(bioRxiv 2021.12.15.472738v1) — specifically the proposed **multi-input Conv-Att model
with soft label** (Table 1, last row; Figure 2).

## 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .                 # installs the local `mci` package (src/mci)

unzip -q images.zip -d data      # -> data/images/<patient_id>/{clock,copy,trail}.png
```

The CUDA wheel index (`cu124`) matches the RTX 2060 SUPER (8GB) this was built and timed
on, driver CUDA 13.1 (backward compatible). On a different GPU/driver, pick the matching
index from https://pytorch.org/get-started/locally/ — everything else is unaffected.

Sanity check before any real training:

```bash
python scripts/smoke_test.py --config configs/original.yaml
```

This (1) times a handful of real-size training steps and reports estimated total GPU-time
and peak VRAM, and (2) runs 2 full epochs end-to-end on a 60-subject slice to catch
integration bugs early. On the reference RTX 2060 SUPER (8GB) it reports ~2.2 GPU-hours
for all 5 seeds x 100 epochs, ~3.7GB peak VRAM.

## 2. What's implemented, and how faithfully

| Paper detail | Where | Notes |
|---|---|---|
| VGG16 (ImageNet) backbone per task, no FC head | `model.py::VGGBackbone` | `torchvision.models.vgg16(weights=IMAGENET1K_V1).features` |
| 1x1 conv projection to hidden dim D, flatten, prepend [CLS] | `model.py::SelfAttentionTower` | D=128 |
| 3 self-attention (transformer) layers, 1 head, ff_dim=512 | `model.py::TransformerEncoderBlock` | post-norm (Attn→Add&Norm→FF→Add&Norm), matches Fig. 2 |
| Concatenate 3 tasks' [CLS] reprs -> 2-node FC + softmax | `model.py::MultiInputConvAtt` | |
| Soft label y = 1 - sigmoid(m - 24.5) | `dataset.py::soft_label_from_moca` | |
| Hard label MCI iff MoCA < 25 | `dataset.py::hard_label_from_moca` | |
| BCE on soft labels | `losses.py::soft_cross_entropy` | equivalent, generalized cross-entropy form |
| Stratified 70/15/15 split, 5 repetitions | `dataset.py::make_splits` | reproduces the paper's exact reported test-set composition (138: 98 healthy / 40 MCI) for every seed — see §4 |
| Pad 256->280, random-crop back to 256 | `dataset.py::build_transforms` | |
| Adam, lr=1e-5, β=(0.9,0.99), ε=1e-7 | `run.py::_build_optimizer` | |
| 100 epochs, batch size 64 | `configs/original.yaml` | see §3 for the micro-batch/grad-accum note |
| Attention rollout for heat maps | `attention_rollout.py` | Abnar & Zuidema (2020), as described in Methods |

Two implementation details the paper doesn't specify, filled in with standard practice
(not methodological changes, just necessary choices to make the described architecture
runnable): ImageNet mean/std input normalization (required for a pretrained backbone),
and dropout=0.1 / GELU activation inside the transformer blocks (BERT-style, matching the
paper's own reference to BERT's self-attention). Both are called out in
`configs/original.yaml`'s comments.

**Not implemented** (by scope decision — see below): the single-input and multi-input
plain-VGG16 baselines, the single-input Conv-Att ablation, and the hard-label ablation
(Table 1's other 4 rows), plus the human-expert Grad-CAM interpretability comparison
(Table 2, requires real clinicians). `model.py::MultiInputVGGBaseline` and
`model.py::build_model`'s `num_inputs`/`tasks` plumbing already support training any of
those (e.g. set `data.tasks: [clock]` for single-input, `model.type: vgg_baseline` for
the plain-VGG16 baseline, `label_mode: hard` for hard labels) if you want to extend to
the full ablation table later — nothing needs to be rewritten, just new config files and
`python scripts/run_experiment.py --config configs/<new>.yaml` runs.

## 3. Hardware note: batch size 64 on an 8GB GPU

Batch size 64 across 3 simultaneous VGG16 towers at 256x256 does not fit in 8GB — it
silently spills into slow shared/system memory (observed ~14s/step instead of ~0.4s/step,
a 35x slowdown, with no error or warning). `training.micro_batch_size: 16` +
`training.batch_size: 64` in the configs make `run_single_split` use gradient
accumulation (4 micro-steps of 16 per optimizer step) — mathematically the same gradient
as a true batch of 64, just computed in smaller chunks. If you have more VRAM, raising
`micro_batch_size` (up to `batch_size`) removes the accumulation overhead; if you have
less, lower it further. `training.amp: true` (mixed precision) is on by default in both
configs for the same VRAM/speed reason and does not change the optimizer, architecture,
or data — safe to turn off (`amp: false`) if you want to rule it out as a variable.

## 4. Running

Single split (one of the paper's "5 different random training-validation-test data
splitting" repetitions):

```bash
python scripts/train.py --config configs/original.yaml --seed 0
```

All 5 repetitions + Table-1-style mean±std summary:

```bash
python scripts/run_experiment.py --config configs/original.yaml
```

Both write per-seed checkpoints/history/metrics to `outputs/original/seed_<n>/` and an
aggregate `outputs/original/summary.json`. Swap in `configs/improved.yaml` for the
version with the additions documented in [IMPROVEMENTS.md](IMPROVEMENTS.md) — same
commands, `--config configs/improved.yaml`.

`scripts/run_all.sh` runs both configs back-to-back (used for the long unattended run).

## 5. Visual explanations (Figure 1 reproduction)

```bash
python scripts/visualize.py --config configs/original.yaml --seed 0 --pick mci
```

Loads that split's trained checkpoint, picks a random MCI test subject (or pass
`--patient_id <id>` for a specific one, or `--pick healthy`), computes the attention
rollout heat map for each of the 3 drawings, and saves a side-by-side original/heat-map
figure (`outputs/original/seed_0/viz_<patient_id>.png`) in the same spirit as the paper's
Figure 1.

## 6. Interpreting results

- **Accuracy/F1/AUC**: computed exactly as the paper describes — threshold the predicted
  P(MCI) at 0.5 for accuracy/F1, use the raw probability for AUC — reported as mean ± std
  over the 5 seeds in `summary.json`, directly comparable to Table 1's
  "Multi-input Conv-Att with soft label" row (Accuracy 0.8116 ± 0.0103, F1 0.6539 ±
  0.0097, AUC 0.8375 ± 0.0116). Exact equality isn't expected — the paper's specific
  random splits, and unspecified details like dropout/init, aren't recoverable — but
  results should land in the same neighborhood; a large gap points to a bug, not
  irreducible randomness.
- **Per-epoch history** (`outputs/.../seed_<n>/history.json`): train loss + validation
  accuracy/F1/AUC/loss per epoch, useful for spotting overfitting (relevant to
  [IMPROVEMENTS.md](IMPROVEMENTS.md) #2).
