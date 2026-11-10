# Improvements beyond the paper's pipeline

`configs/original.yaml` reproduces the paper's "multi-input Conv-Att with soft label"
model exactly as specified in the Materials & Methods section. `configs/improved.yaml`
is the same architecture family with a set of additions that are **not** in the paper,
chosen because they address specific, visible weaknesses of the original recipe on this
dataset. None of this is guaranteed to beat Table 1 — treat it as a documented,
reasonable set of bets to A/B against `original` on the same 5 splits, not as a proven
result. Run both and compare `outputs/original/summary.json` vs
`outputs/improved/summary.json`.

Numbers below match the `# IMPROVEMENTS.md #N` comments in `configs/improved.yaml`.

## 1. Class-balanced sampling (`training.class_balanced_sampling: true`)

**Problem:** the dataset is imbalanced, 267 MCI vs. 651 healthy (~29%/71%). The paper's
own results show it: F1-score is consistently the weakest metric in Table 1 (as low as
0.36 for the single-input baselines), well below accuracy/AUC — a classic imbalance
symptom, since accuracy and even AUC can look fine while the minority (MCI) class is
under-served.

**What it does:** a `WeightedRandomSampler` over the training set, weighting each class
inversely to its frequency, so MCI and healthy subjects are seen equally often per epoch
(the total number of training samples per epoch is unchanged, only their distribution).

**Why it should help:** directly targets recall on the MCI class, which is the clinically
important one (missing MCI is worse than a false alarm). **Trade-off:** can trade a bit
of accuracy/precision for F1/recall — check both.

## 2. Best-validation-AUC checkpoint instead of the last epoch (`model_selection: best_val_auc`)

**Problem:** the paper trains for a fixed 100 epochs at lr=1e-5 with no mentioned early
stopping or checkpoint selection, then (implicitly) evaluates the final-epoch model. With
~640 training subjects and a 46M-parameter network (3 fine-tuned VGG16 towers), late
epochs are exactly where fine-tuned CNNs on small medical-imaging datasets start to
overfit the training split.

**What it does:** tracks validation AUC every epoch and keeps the best checkpoint, used
for the final test evaluation instead of the epoch-100 weights.

**Why it should help:** removes variance/luck from "which epoch happened to be the last
one" and typically improves test AUC/F1 on small datasets. **Trade-off:** slightly more
disk/RAM (one extra copy of the weights kept in memory during training); makes the run
less literally "the last epoch", which is exactly why this is opt-in and not in
`original.yaml`.

## 3. AdamW + weight decay + cosine LR schedule with warmup

**Problem:** plain Adam with a flat lr=1e-5 for 100 epochs has no regularization pressure
on the weights (Adam's own weight decay is well known to interact badly with its adaptive
learning rate — see Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, 2019)
and no annealing, so the tail of training is 1e-5 forever, which can under-move on the
randomly-initialized attention layers early and over-move relative to what a decayed rate
would do late.

**What it does:** `AdamW` (decoupled weight decay, `weight_decay: 0.01`), 5 epochs of
linear warmup, then cosine decay to ~0 by epoch 100. Base lr raised slightly to `3e-5`
since the warmup+decay schedule tolerates a higher peak than a flat schedule would.

**Why it should help:** weight decay regularizes the large pretrained backbones (less
overfitting → pairs well with #2); warmup stabilizes the randomly-initialized
self-attention/CLS-token parameters, which otherwise receive large, noisy gradients
through the pretrained VGG16 features from step 1.

## 4. Mild augmentation: small rotation/translation/scale (`augmentation.extra: mild_affine`)

**Problem:** the paper's only augmentation is a translation jitter (zero-pad to 280,
random-crop back to 256). Hand-drawn clock/cube/trail images naturally vary in rotation
and scale from subject to subject (tablet tilt, hand size, drawing area used); the model
never sees that variation synthetically.

**What it does:** adds `RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.97, 1.03))`
on top of the paper's pad+crop, applied independently per task image. Small angles/scale
are deliberate — large rotations would distort clock-hand angles, which is exactly the
diagnostic signal for the CDT.

**Explicit non-improvement — do not add horizontal/vertical flips.** They are actively
harmful here, unlike in typical vision tasks: a horizontally-flipped clock face still
looks like a clock but silently reverses the hands' apparent direction/time; a flipped
Necker cube changes which faces look "in front" (the depth-cue error the model is
supposed to detect); a flipped trail-making test reverses the 1-A-2-B-... left-to-right
reading order the subject was instructed to follow. Any of these would let the model
train on drawings that no longer correspond to their label.

## 5. Learned positional embeddings on the patch tokens (`model.use_pos_embed: true`)

**Problem:** the paper's self-attention operates on a flattened 8x8 VGG16 feature grid
plus a [CLS] token, with no positional encoding — as described, the only way an attention
layer can tell two patch tokens apart positionally is indirectly, through whatever
location-dependent content convolution happened to bake into each patch's features (edge
effects, receptive field content). Self-attention itself is permutation-invariant.

**What it does:** adds a learned `(65, 128)` positional embedding, added to the tokens
before the first self-attention layer (standard ViT-style absolute position embedding,
initialized `~N(0, 0.02)`).

**Why it should help:** explicit spatial grounding tends to sharpen both classification
and — more importantly for this paper's stated goal — the attention-rollout heat maps,
since the model can now separate "this is the top-right of the drawing" from "this
content happens to look like that." This is the change most likely to affect
interpretability quality rather than just accuracy, so also inspect
`scripts/visualize.py` outputs side-by-side with `original`, not just the metrics table.

## 6. More attention heads (`model.num_heads: 4`, was 1 in the paper)

**Problem:** a single attention head must average all relational patterns (e.g.,
"clock-hand region" vs. "digit region" vs. "contour region") into one softmax; with
hidden_dim=128 there is ample room for more heads without changing the model's overall
capacity much.

**What it does:** 4 heads of dim 32 instead of 1 head of dim 128 (same `hidden_dim`
total, so parameter count is essentially unchanged — only the `qkv`/`proj` projections'
internal split changes). Attention-rollout code already averages over heads before
building the rollout matrix, so `scripts/visualize.py` works unchanged.

**Trade-off:** slightly less "one clean interpretable map" per layer (4 heads → rollout
averages them) — if the qualitative heat maps get *less* legible with 4 heads on your
data, this is the first change to revert.

---

## Ideas considered but not turned on by default in `improved.yaml`

These are larger or riskier changes — worth trying as *separate* ablations (one change at
a time against `original`, not stacked with everything above) rather than folded silently
into "the" improved config:

- **Different backbone** (EfficientNet-B0 / ConvNeXt-Tiny instead of VGG16). Modern
  backbones usually transfer better than VGG16 with fewer parameters and less
  overfitting risk on ~640 training images, but this changes the feature-map spatial
  resolution feeding the self-attention layers and is a big enough change to isolate on
  its own. `src/mci/model.py::VGGBackbone` is the one place to swap.
- **Focal loss** instead of (soft) cross-entropy, to further emphasize hard/minority
  examples on top of #1.
- **5-fold (or repeated stratified k-fold) cross-validation** instead of one fixed
  train/val/test split per seed — more robust given only 918 subjects, at ~5x the
  compute.
- **Ensembling the 5 seeds' trained models** (average predicted probabilities) as a
  final predictor — nearly free given the 5 runs are already produced by
  `run_experiment.py`, typically a reliable small accuracy/AUC bump.
- **Test-time augmentation** (average predictions over a few random crops per test
  image) — same idea as ensembling, applied at inference time.
- **Discriminative learning rates / gradual unfreezing** (lower LR for the pretrained
  VGG16 layers, higher for the randomly-initialized attention head; optionally freeze
  the backbone for the first few epochs) instead of one flat LR for the whole network
  from step 1 — addresses the same "randomly-initialized attention on top of pretrained
  features" issue as the warmup in #3, more surgically.
- **Exponential moving average (EMA)** of the weights for a more stable final model,
  complementary to #2.

## Blocked by data, not by method: pen dynamics (pressure, velocity, timing)

Raised while discussing the tablet-based test administration: could pen pressure (or
stroke velocity/timing) be added as a feature? The paper's own Discussion flags exactly
this as future work — switching MoCA to a digital tablet "allows us to store the drawing
trajectory... which contains both temporal and spatial information," while their model
(and the released dataset) "only uses the final drawing... discard[ing] the
information-rich temporal information provided by the drawing trajectory."

**Checked directly against `images.zip`:** `label.csv` + `images.zip` contain only the
final rasterized 256x256 PNGs, nothing else. A distance-transform pass over
`clock.png` samples shows stroke width is uniform (~2px, median exactly 2.0) with no
smooth thickening/thinning along a single stroke — only at line junctions/crossings,
which is a rendering artifact, not pressure modulation. Confirmed visually too (zoomed
crop of a clock-hand crossing: constant line weight throughout). **There is no pressure
signal recoverable from the released images, not even as a lossy proxy** — building a
"thickness = pressure" feature on this data would not be scientifically defensible.

Implementing this for real needs the raw per-stroke capture stream (x, y, pressure,
timestamp per sample point), which was never published here. Two ways to actually get
it: (1) ask the paper's authors / King Chulalongkorn Memorial Hospital whether the raw
tablet logs still exist and can be shared beyond what's already public (no guarantee,
possible IRB/privacy constraints beyond the released de-identified images); or (2) a new
data collection with a pressure-capable stylus/app that logs the full stroke stream —
i.e., a new study, not a re-analysis of this dataset. If such trajectory data ever
becomes available, it would plug in as a 4th (sequence-shaped) input stream alongside
the 3 existing image towers — architecturally additive, not a rework of what's here.

## How to compare `original` vs `improved`

```bash
python scripts/run_experiment.py --config configs/original.yaml
python scripts/run_experiment.py --config configs/improved.yaml
```

Both write a `summary.json` (mean ± std accuracy/F1/AUC over the 5 seeds, same format as
the paper's Table 1) to their `output_dir`. Because both configs use the same
`seed_list`, the 5 train/val/test splits are identical between the two runs (the split
only depends on the seed and `label.csv`), so the comparison is apples-to-apples.
