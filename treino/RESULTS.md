# Results: original vs improved vs paper

Full run: 5 seeds x 100 epochs, both configs, RTX 2060 SUPER (8GB). Total wall time
~4h20min (`original` ~2h43min, `improved` ~2h17min). All 1000 expected epoch-lines
logged, no NaN/Inf/errors/tracebacks anywhere in the run, and train/val/test patient-ID
sets independently verified disjoint (no leakage) — the gap below is real model/training
behavior, not a crash or a data bug.

## Headline numbers

| Metric | Paper (Table 1, proposed) | `original` (ours) | `improved` (ours) |
|---|---|---|---|
| Accuracy | 0.8116 ± 0.0103 | 0.7536 ± 0.0210 | 0.7362 ± 0.0227 |
| F1 | 0.6539 ± 0.0097 | 0.5241 ± 0.0570 | 0.4498 ± 0.1020 |
| AUC | 0.8375 ± 0.0116 | 0.7650 ± 0.0373 | 0.7544 ± 0.0452 |

Two honest conclusions, neither of which is what we expected going in:

1. **Neither config reaches the paper's numbers.** The gap (accuracy -0.058,
   F1 -0.130, AUC -0.073 for `original`) is bigger than the paper's own std would
   suggest is normal split-to-split variance — this points to a genuine
   methodological difference from the paper, not just unlucky seeds (see Diagnosis).
2. **`improved` did not beat `original` — it's slightly worse on every metric, with
   ~2x the F1 variance (0.102 vs 0.057).** The documented rationale in
   IMPROVEMENTS.md was reasonable a priori, but the data say the bundled changes,
   as configured, did not help on this dataset. Worth stating plainly rather than
   spinning it.

Per-seed numbers: `outputs/original/summary.json`, `outputs/improved/summary.json`,
and each `outputs/<config>/seed_<n>/result.json` / `history.json`.

## Diagnosis: every one of the 10 runs overfits within the first ~20 epochs

Looking at when each run's best validation AUC actually occurred, across all 10
runs (5 seeds x 2 configs), out of 100 epochs each:

| | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| `original` best-val-AUC epoch | 7 | 17 | 15 | 8 | 8 |
| `improved` best-val-AUC epoch | 5 | 8 | 5 | 11 | 3 |

Every single run peaks on validation within the first 17 epochs (most within the
first 10), then spends the remaining 80-95 epochs with validation AUC flat-to-declining
while train_loss keeps decreasing the whole time (e.g. `original` seed 0: train_loss
0.63 -> 0.31 over 100 epochs, val_auc 0.77 -> peak 0.86 at epoch 7 -> oscillating
0.81-0.86 with no further trend). Textbook overfitting signature, and it is
universal across every split and both configs — not one unlucky run.

**Most likely cause:** all three VGG16 backbones are fine-tuned end-to-end
(`train_backbone: true`, ~46M trainable parameters total) against only 642 training
images per split. That is a lot of capacity relative to data, and it fits fast. The
paper doesn't state whether its VGG16 backbones were frozen or fine-tuned — we chose
"fine-tune everything" because lr=1e-5 reads naturally as a fine-tuning rate, but a
frozen-backbone (train only the ~1-2M projection/attention/classifier parameters)
reading of the same architecture is equally consistent with what's written, would
need far fewer effective parameters for 642 images, and is a common choice at this
dataset size in the transfer-learning literature this paper draws on.

**This also explains why `improved` came out worse, not better.** `best_val_auc`
model selection assumes the validation curve has a real, findable peak worth
picking out. With only 138 validation subjects, per-epoch val AUC is itself a noisy
estimate; scanning 100 noisy checkpoints and keeping the single highest value is a
multiple-comparisons setup that tends to pick a lucky noise spike rather than a
truly-better model — and an early one at that (epoch 3-11 for `improved`), before
the randomly-initialized self-attention layers have had much training. Direct
evidence: for seed 0, switching from `original`'s last-epoch checkpoint to
`improved`'s best-val-AUC checkpoint (epoch 5) made test performance *worse*
(F1 0.486 -> 0.327, AUC 0.722 -> 0.687) — the opposite of what IMPROVEMENTS.md #2
predicted. The idea wasn't unreasonable, but this data doesn't support it as
implemented, stacked with the rest of `improved`'s changes.

## Recommended next step (not yet run)

Test `model.train_backbone: false` (freeze the 3 VGG16 backbones, train only the
1x1 projections + self-attention + classifier) on top of `configs/original.yaml`'s
otherwise-unchanged, paper-faithful settings. This directly targets the
capacity-vs-data-size mismatch above and is a single, isolated change — cheap to
try (same code path, one config flag) before combining it with anything else.
