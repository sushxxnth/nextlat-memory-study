# Midterm published-probe supplement

Aditi & Sushanth — November 14, 2026

This supplement adds frozen-model diagnostics to the existing delayed-query memory study. It contains no newly trained Transformer models and makes no claim that harmful compression has been established.

## Main result

Dense-delay, response-supervised models at retention delay 64, three training seeds:

| Four-bit decoding probe | NTP (%) | NextLat (%) |
|---|---:|---:|
| Shai-style affine OLS | 56.3 +/- 4.0 | 83.4 +/- 11.3 |
| Fixed ridge alpha=1 | 55.5 +/- 6.0 | 86.4 +/- 11.0 |
| Official bias-free linear CE readouts | 57.0 +/- 6.0 | 82.7 +/- 7.1 |

Values are mean +/- sample SD across training seeds, not confidence intervals. These are frozen before-query probe scores, not native generation accuracy. Shuffled-training-label affine controls average 50.3% and 50.8%, respectively. This reused-test analysis is exploratory. See the report for all conditions, failures and limitations.

## Contents

- `protocol.json`: settings frozen before fitting, source and input-plan hashes.
- `all_results.json`, `aggregate.json`, `per_seed.csv`: every condition, seed, layer, method and evaluated delay.
- `audit.json`: input hashes, exact feature reproduction, optimizer diagnostics and limitations.
- `causal_query_audit.json`: perturbation of the future query while holding the inspected prefix fixed.
- Individual run folders: fitted affine/ridge models including shuffled controls, official readout weights, and metrics.
- `published_probes.py`: exact frozen analysis source used for the run.
- `MIDTERM_RESULTS.md` and figures: editable findings and plots.

## Source methods

Shai et al., https://arxiv.org/abs/2405.15943 : affine belief-state regression and shuffle controls, adapted here to four binary cue marginals. We do not reproduce the HMM belief-simplex experiment or its full resampling protocol.

Teoh et al., https://arxiv.org/abs/2511.05963 : official `Probe.linear_probes` heads from commit `3770be6009cea2b3c455a9ce7f2ca88b504bb955`. Binary cue targets, fixed regularization and LBFGS optimization are adaptations; this is not a replication of their future-token probe experiment.

## Reproduce in the existing project

Use the project's `work/nextlat-venv/bin/python` and set `PYTHONPATH` to `outputs/nextlat-published`, or run from that directory in an activated environment. Limit Torch/BLAS threads to two. Commands:

```sh
python -m unittest memory.test_published_probes -v
python -m memory.published_probes --run memory/runs/controls_v3_6000 --out memory/probes_midterm_reproduction
python -m memory.probe_report --out memory/probes_midterm_reproduction
python -m memory.probe_causality_audit --run memory/runs/controls_v3_6000 --out memory/probes_midterm_reproduction
```

The PDF build script uses ReportLab and the Georgia/Arial fonts available on the original Mac. It defaults to the original result directory; change DATA to render a different analysis folder.

The analysis archive is a supplement, not a standalone training environment. Original Transformer checkpoints and feature caches are retained in `memory/runs/controls_v3_6000` in the workspace and are not duplicated in this archive. Training code and prior raw training summaries are available in the existing `outputs/nextlat-memory-controls.zip`. The official repository is pinned and unchanged. No test data or test labels are used to fit probes, normalize features or select layers. All evaluated layers and conditions are retained.
