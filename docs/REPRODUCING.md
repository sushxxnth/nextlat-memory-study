# Methods and reproduction

Run commands from the repository root. Setup and the result table are in the
[top-level README](../README.md).

## Data and supervised targets

The published Path-Star benchmark uses a fixed 200,000-example training corpus
and 20,000-example held-out test set. The official prompt mask supervises the
path and EOS. Data generation and sampled autoregressive full-path evaluation
come from the pinned upstream repository.

Our memory extension generates sequences on the fly:

```text
BOS  CUE_1  CUE_2  CUE_3  CUE_4  TAG  DISTRACTORS...  QUERY_INDEX  ANSWER  EOS
```

The cues are binary (red/blue); any of the four can be queried. The nuisance tag
has four values. The query index arrives only after the distractors. A matched
family has eight counterfactual examples varying the queried bit and tag while
holding the other bits, query and distractors fixed. Hash-based split assignment
keeps noise families separate between train, probe fitting, validation and test.
Repeated presentations are not counted as distinct examples.

Sparse training delays are 4,8,16; dense training covers every integer 4–16.
Response supervision applies the official NTP/KL prompt mask to ANSWER and EOS;
context Smooth L1 remains active. Full supervision predicts all eligible tokens.
We cross those factors with NTP/NextLat and three seeds (1237–1239).

## Published baseline

Small CPU pipeline check:

```sh
python run_pathstar.py --algorithm gpt --out runs/pathstar_gpt_smoke
python run_pathstar.py --algorithm nextlat --out runs/pathstar_nextlat_smoke
```

These defaults are small pilot settings. They do not reproduce the published
scale or the recorded 18.625%/98.630% result. For the recorded scale, allocate a
compatible CUDA GPU and run:

```sh
python prepare_paper.py
python launch_paper.py --algorithm gpt --seed 1234
python launch_paper.py --algorithm nextlat --seed 1234
```

The launcher uses `configs/paper_g5_5.json`, bf16 and gradient accumulation to
reach an effective batch of 512. Our recorded validation is one seed and one
topology. Execution differences from the multi-GPU release are disclosed in the
archived report. Cluster credentials and institution-specific configuration are
not part of this repository.

## Four-cue controls

For a short pipeline check use `configs/memory_four_cue_smoke.json`. To reproduce
the original full factorial procedure:

```sh
python -m memory.controls --out reproduction/controls_2000 --workers 2
python -m memory.extend_controls --source reproduction/controls_2000 --out reproduction/controls_6000 --steps 6000 --workers 2
```

The second command requires completed first-stage runs and continues every cell
uniformly, keeping original checkpoints untouched. Fresh output directories
avoid mixing changed plans or source versions. Checkpoints include model,
optimizer and generator state. The exact-continuation test compares continued
and uninterrupted training parameters and training-stream digests.

The original training extension was motivated by validation failures after initial
test results were visible. Replicating the procedure does not remove that original
exploratory status. A future confirmatory study needs fresh held-out data.

## Frozen probes and causal check

```sh
python -m memory.published_probes --run reproduction/controls_6000 --out reproduction/probes
python -m memory.probe_report --out reproduction/probes
python -m memory.probe_causality_audit --run reproduction/controls_6000 --out reproduction/probes
```

Probe fitting uses 1,024 examples at each of delays 4,8,16 (3,072 total), on
families separate from backbone training and evaluation. Tests have 1,024
examples per delay, in 128 families. We inspect the single state before the query
index, never an answer/EOS input. All four cue bits are decoded independently,
without future-query side information.

- Affine ordinary least squares follows the linear-regression methodology of
  Shai et al., adapted to cue marginals and a four-way nuisance tag.
- Ridge uses fixed alpha=1 as a sensitivity check. Both methods standardize with
  probe-training statistics only; binary outputs are thresholded at 0.5.
- Official NextLat `Probe.linear_probes` modules supply four bias-free binary
  readouts. Targets, fixed L2 and LBFGS settings are adaptations. No backbone
  parameters are trained. Some 200-iteration fits do not reach the requested
  gradient tolerance; those diagnostics remain in the results.
- Twenty fixed shuffled-training-label fits per affine method/site/model permute
  families and rows within each family. True test labels remain unchanged.
- Both residual blocks, their concatenation, token embedding and final norm are
  reported. Concatenation doubles feature dimension; no layer is selected using
  test performance.

Low linear accuracy cannot establish information erasure. The query-perturbation
audit checks that changing a future query cannot change the earlier inspected
state. Native generation retains full causal attention; attention-free NextLat
rollout is separately reported and is not an equal-compute comparison.

## Evidence versus recomputation

The `results/` directory contains original recorded metrics, histories and source
snapshots. Large Transformer checkpoints, probe weights and feature arrays remain
in the original local workspace, rather than Git. To recompute hidden states and
probes on another machine, reproduce training first or obtain those artifacts.
Historical absolute paths in audit JSON are preserved provenance and should not
be used as current runtime paths. Do not overwrite the archived evidence with a
new experiment.
