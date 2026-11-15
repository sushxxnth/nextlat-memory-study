# When Does Compression Forget Too Much?

## Midterm check-in: published probes reveal accessible memory in NextLat

**Aditi & Sushanth | CSCI-GA.3033 The Science of Deep Learning | November 14, 2026**

This is an exploratory analysis of 24 existing, fixed-final-budget models, not 24 new training runs. All uncertainty below is sample standard deviation across three independent training seeds. Counterfactual examples and 20 shuffled fits are not treated as independent training seeds.

### Primary result at delay 64

Dense training uses every delay 4-16; response supervision scores ANSWER/EOS for NTP/KL while retaining context latent Smooth L1. Probes see a single state before the query index arrives and decode all four stored bits independently. Binary chance is 50%.

| Probe | NTP (%) | NextLat (%) |
|---|---:|---:|
| Shai-style affine OLS | 56.3 +/- 4.0 | 83.4 +/- 11.3 |
| Fixed ridge sensitivity check | 55.5 +/- 6.0 | 86.4 +/- 11.0 |
| Official NextLat linear CE heads | 57.0 +/- 6.0 | 82.7 +/- 7.1 |

### Every experimental condition at delay 64

| Condition | Native recall (%) | Affine bit probe (%) | Official CE bit probe (%) | Training gate |
|---|---:|---:|---:|---:|
| dense_full_gpt | 18.2 +/- 19.7 | 50.9 +/- 0.9 | 51.3 +/- 0.1 | 2/3 |
| dense_full_nextlat | 40.6 +/- 14.2 | 53.6 +/- 7.9 | 59.1 +/- 5.5 | 2/3 |
| dense_response_gpt | 60.3 +/- 12.8 | 56.3 +/- 4.0 | 57.0 +/- 6.0 | 3/3 |
| dense_response_nextlat | 79.9 +/- 4.5 | 83.4 +/- 11.3 | 82.7 +/- 7.1 | 3/3 |
| sparse_full_gpt | 22.3 +/- 20.9 | 51.0 +/- 2.7 | 52.3 +/- 1.0 | 3/3 |
| sparse_full_nextlat | 9.2 +/- 8.1 | 50.2 +/- 2.1 | 61.6 +/- 4.8 | 3/3 |
| sparse_response_gpt | 61.6 +/- 9.4 | 58.5 +/- 2.3 | 63.7 +/- 1.8 | 3/3 |
| sparse_response_nextlat | 77.7 +/- 8.9 | 80.6 +/- 16.2 | 82.4 +/- 12.6 | 3/3 |

### Methods and provenance

- Shai et al.: affine least-squares decoding, adapted from belief-state regression to known binary cue marginals; not a replication of HMM belief-simplex geometry.

- NextLat: reuse the unchanged released Probe.linear_probes heads (four bias-free binary readouts). Targets, L2 penalty and optimizer are adaptations; this is not the TinyStories future-token experiment.

- Training-only standardization for affine/ridge; CE uses raw final-normalized states. All probes train on separate families at delays 4,8,16 (3,072 examples). No test tuning or checkpoint selection.

- Shuffled-label controls permute both families and rows within families; 20 fixed fits per affine method/site/model. Test labels remain true. This is an adaptation of shuffle controls, not the original paper's full resampling protocol.

- Embedding, both residual blocks, concatenated blocks and final norm are all reported. The final norm reproduces saved features bitwise across 264 run/split/delay checks.

### Interpretation and limits

The consistent result is stronger linear accessibility of relevant historical cues in response-supervised NextLat at long delay. In the dense condition, the second block carries markedly more decodable cue information than the first. At a trained delay, irrelevant nuisance-tag decoding is lower for NextLat, consistent with selective representation, but failed linear decoding does not prove erasure.

Native recall at delay 64 is 60.3 +/- 12.8% for NTP and 79.9 +/- 4.5% for NextLat. NTP performs better at delays 24,32,48, so this is not a universal ranking. Probe accuracy is separate from unrestricted native generation.

There are only three seeds. Test families were previously examined. All models were uniformly extended from 2,000 to 6,000 updates after validation learning failures, with early test results already visible; intermediate results remain preserved. Dense/full conditions do not all pass the >=95% learning gate at every training delay. Linear CE optimization uses a fixed 200-iteration cap and some fits do not attain the requested gradient tolerance. OLS and ridge independently support the main pattern.

An ideal belief state does not discard predictive information. These experiments do not establish harmful compression, a hard state-capacity bottleneck, or a causal link between the auxiliary predictor and memory. Full causal attention remains available in normal Transformer inference. Attention-free latent rollout is separate evidence of retained information, not an equal-compute baseline.

### Next steps

1. Before submission: verify the figures/tables against the raw files, include both names, and submit an exploratory midterm report with the tested question and current limitations.

2. After check-in: lock a fresh held-out set and endpoint, add training seeds, and test a response-supervised lambda sweep (0,.1,1,10) with stable in-distribution learning.

3. Vary retention delay and cue capacity while holding prefix length and vocabulary fixed. Predictor-MLP width is an auxiliary-function capacity variable, not a hard Transformer state bottleneck.

4. Add nonlinear probes and causal cue interventions to separate linear inaccessibility from actual information loss; retain all seeds and report learning gates.

### References

- Shai et al., Transformers Represent Belief State Geometry in their Residual Stream. https://arxiv.org/abs/2405.15943

- Teoh et al., Next-Latent Prediction Transformers Learn Compact World Models, v4. https://arxiv.org/abs/2511.05963

- Official probe code, pinned commit 3770be6: https://github.com/JaydenTeoh/NextLat/blob/3770be6009cea2b3c455a9ce7f2ca88b504bb955/models/model_probe.py

### Reproduce

From outputs/nextlat-published with the project Python environment:

```sh
python -m unittest memory.test_published_probes -v
python -m memory.published_probes --run memory/runs/controls_v3_6000 --out memory/probes_midterm_reproduction
python -m memory.probe_report --out memory/probes_midterm_reproduction
```

Full settings, source hashes, per-seed results, all layers and delays, fitted probe weights and optimizer diagnostics are retained alongside this report. The fitting script rejects a changed protocol when resuming.
