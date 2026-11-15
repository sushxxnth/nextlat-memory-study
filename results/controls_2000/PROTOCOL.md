# Four-cue factorial controls

**Aditi & Sushanth · November 13, 2026**

The prior study exposed full-vocabulary output failures and sensitivity to
unseen lengths, including an intermediate delay. This follow-up targets those
mechanisms. It uses fresh seeds 1237, 1238 and 1239, and trains every model from
scratch. The source and plan were frozen before training. This is a follow-up
chosen after inspecting earlier results, rather than a public preregistration.

## Factors

| Factor | First level | Second level |
|---|---|---|
| Training delays | Uniform over 4, 8, 16 | Uniform over every integer 4–16 |
| Token supervision | Full sequence | Only ANSWER and EOS |
| Method | Official NTP/GPT | Official NextLat |

All factors are crossed: 2 × 2 × 2 × 3 seeds = 24 training runs. NextLat uses
auxiliary weight 1 for both Smooth L1 and KL, horizon 3 and predictor width
128. Each model uses the same 2-layer, 128-wide, 4-head trunk and a 2,000-step,
batch-32 budget. The optimizer settings match the prior memory study.

Response masking uses the authors' existing context_length mechanism, set to
the query position for each equal-length training batch and restored afterward.
It masks prompt targets from both NTP and KL. It retains Smooth L1 on context
transitions. The operation is not a head-only intervention: it changes the
target normalization and gradient balance of token losses against latent loss.
No official source file is edited.

## Matching and evaluation

Within each seed and delay schedule, all masks and methods consume the exact
same training examples, checked by cumulative stream hashes. All cells use
identical test families for each delay. Probe-training families and budgets
are identical across cells: 1,024 examples at each of delays 4, 8, 16. This
avoids giving dense-delay models extra labeled probe data.

Validation curves use the common trained delays 4, 8, 16. Every model is tested
at the final fixed budget; no best-test checkpoint is selected. Evaluations
cover every integer delay 4–16, plus 24, 32, 48 and 64, with 1,024 examples per
delay. Delay 12 is trained in the dense cells and unseen in the sparse cells.
Improvements at 12 under dense training are not claims of OOD extrapolation.
The primary extrapolation comparisons are at 24, 32, 48 and 64.

We report unrestricted recall, restricted two-answer recall, the rate of
predicting a non-answer token, pre-query stored-bit probes, post-query answer
probes, geometry and NextLat's one-step/long attention-free rollouts. Answer
and EOS are excluded from evaluation inputs. Probe and test families remain
disjoint by the existing family hash split.

The full learning gate requires both methods in a matched cell to reach at
least 95% recall at every delay trained in that cell. A separate common gate
checks only delays 4, 8, 16. All runs are reported, even if gates fail. Paired
differences and factorial interactions are reported with sample SD and all
three seed values, not p-values or confidence intervals.

## Limits

Steps and examples are matched; expected training-token counts differ slightly
between sparse and dense schedules. Response supervision changes the overall
training objective's normalization. Three seeds provide preliminary evidence.
Frozen-probe failures can reflect changed encodings; poor latent rollouts can
reflect accumulated transition error. Predictor capacity and length shift are
not hard information bottlenecks. These controls do not isolate a complete
theory of memory erasure.
