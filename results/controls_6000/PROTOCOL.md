# Uniform 6,000-step continuation of the factorial controls

**Aditi & Sushanth · November 13, 2026**

This phase continues all 24 factorial runs from 2,000 to 6,000 updates. The
factors, four-cue task, three seeds, held-out families, probes, architecture,
optimizer and loss definitions are unchanged. See `../controls_v3/PROTOCOL.md`
for the original design. Every condition receives the same final budget.

The common-delay validation minimum remained below 95% in several dense-delay
conditions at 2,000 steps. Those validation values and the source-plan hash
are recorded in this phase's immutable `plan.json`. The purpose of the longer
budget is to remove inadequate in-distribution task learning as a confound.
The intermediate test results had also been visible, so this is an exploratory
follow-up rather than an untouched confirmatory experiment. We do not choose
the highest-test checkpoint; all initial and final results are retained.

Original checkpoints/results remain untouched. Copies retain model tensors,
optimizer state, data-generator state, training logs and cumulative stream
digests. Only the planned step-budget metadata changes in the copies. Each
copy has a `continuation.json` identifying its original checkpoint hash. A
regression test confirms exact equality between uninterrupted training and
the copied-checkpoint continuation, including every model parameter and the
training-stream digest.

These are 24 continued models with two evaluated budgets, not 48 independent
training seeds. The analysis retains the same matched learning gates, all
seed values, sample SD and factorial contrasts. Any remaining failures of
the 95% gate are flagged rather than reinterpreted as memory erasure.
