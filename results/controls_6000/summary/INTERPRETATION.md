# What the supervision and delay controls show

**Aditi & Sushanth · November 13, 2026**

We completed 24 matched four-cue training runs at 2,000 updates, then continued every run uniformly to 6,000 updates after several validation learning checks failed. The final results below use 6,000 updates and three fresh seeds (1237–1239). Original intermediate results and checkpoints are preserved. These are 24 continued models, not 48 independent seeds.

## The strongest finding: the supervised targets matter

At the longest evaluated delay (64 distractors), scoring only ANSWER/EOS targets during training improves recall substantially. Evaluation still permits every vocabulary token. This is not forced-choice evaluation; the response-only condition uses the official prompt-mask setting for NTP and KL, while keeping context Smooth L1 active.

| Training delays | Token supervision | NTP recall at 64 | NextLat recall at 64 |
|---|---|---:|---:|
| 4, 8, 16 | full | 22.3% ± 20.9% | 9.2% ± 8.1% |
| 4, 8, 16 | response | 61.6% ± 9.4% | 77.7% ± 8.9% |
| Every integer 4–16 | full * | 18.2% ± 19.7% | 40.6% ± 14.2% |
| Every integer 4–16 | response | 60.3% ± 12.8% | 79.9% ± 4.5% |

Values are mean ± sample SD across three seeds, not confidence intervals. * Dense/full comparisons pass the matched learning gate in only one of three seeds, so that cell is confounded by incomplete in-distribution learning. All response-only comparisons pass in all three seeds.

With sparse delays, response masking changes NextLat’s non-answer rate at delay 64 from 88.0% to 0.0%; NTP changes from 61.9% to 3.1%. Paired recall gains are +68.5 ± 4.9 percentage points for NextLat and +39.3 ± 13.3 points for NTP. These large differences show why the original unrestricted-output collapse could not be interpreted directly as cue erasure. Masking also changes token-loss normalization and representation training, so it is not a head-only intervention.

## Broader delay coverage helps interpolation, without a uniform extrapolation gain

Dense training includes delay 12; near-perfect recall there is therefore an in-distribution result, not evidence of extrapolation. With response-only supervision, broadening coverage changes delay-64 recall by −1.3 ± 11.6 points for NTP and +2.2 ± 10.8 points for NextLat. Three seeds do not establish a reliable additional benefit at this endpoint.

Performance is not monotonic with delay. Under dense response training, NTP is better at delays 24, 32 and 48; NextLat is better at 64. We therefore do not claim a uniform NextLat advantage or a simple memory-versus-delay threshold.

## Internal-state and attention-free memory evidence

At delay 64 under dense response training, the average frozen pre-query stored-bit probe scores 83.3% ± 7.5% for NextLat and 54.5% ± 6.7% for NTP (binary chance is 50%). These probes decode each stored bit without knowing the future query index.

NextLat can also start from a single state after the first distractor and process the remaining observed distractors/query through its latent predictor, without further Transformer attention or any answer input. The same output head reads the final latent state:

| Delay | NextLat normal Transformer recall | NextLat attention-free latent rollout recall |
|---:|---:|---:|
| 24 | 72.4% ± 12.4% | 91.7% ± 9.8% |
| 32 | 74.9% ± 13.0% | 84.9% ± 15.4% |
| 48 | 75.7% ± 12.5% | 79.5% ± 14.0% |
| 64 | 79.9% ± 4.5% | 74.7% ± 13.4% |

The latent rollout is evidence of useful retained memory in this setting. It is an additional recurrent inference path with different computation, not an equal-FLOP comparison against NTP. High probe accuracy is positive evidence of accessibility; a failed frozen probe would not prove all information was erased.

## Training budget changes the interpretation

With sparse response training at 2,000 updates, NTP recall at 64 was 69.9% and NextLat was 62.6%; at 6,000 updates the ordering reverses. The uniform extension was motivated by validation learning failures in other cells, after initial test results had already been visible. Both budgets are retained, and no best-test checkpoint was selected. These remain exploratory small-model results, not a robust model ranking.

## Implication for the project

Our current evidence supports task-relevant memory preservation under appropriately supervised NextLat, while revealing important readout, training-budget and length-generalization confounds. It does not establish a compression-induced information-loss boundary. Predictor width still controls the auxiliary MLP rather than imposing a hard bottleneck on the full Transformer.

The next capacity study should hold prefix length and vocabulary fixed while varying the set of cues that can be queried, use the response-only conditional objective as the main task, and include enough independent seeds and stable in-distribution learning before interpreting retention differences.

## Verification

The plan/source hashes and matching training/evaluation/probe families are audited. All 27 correctness checks pass, including an exact continued-versus-uninterrupted training test. Final checkpoints were reloaded and Transformer predictions and query hidden states were reproduced at delays 8, 12 and 64 for all 24 runs (72 run/delay checks). Saved-state/output-head checks also verify the recorded normal and latent-rollout recalls. Full seed values, learning gates and factorial contrasts are retained.

## Midterm-ready description

After validating the published NextLat implementation, we designed controlled delayed-query memory experiments using matched counterfactual histories, frozen-state probes and attention-free latent rollouts. A factorial control study shows that response-target supervision removes many wrong-token failures that could otherwise look like forgetting. Under that supervision, NextLat retains useful single-state memory and can answer after long attention-free latent rollouts. Results depend on retention length and training budget, so our preliminary findings support a careful separation of output failures, representation accessibility and genuine memory loss.
