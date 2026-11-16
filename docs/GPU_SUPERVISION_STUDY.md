# GPU study: output failures versus cue accessibility

This follow-up tests whether apparent forgetting comes from unrestricted token
readout, inaccessible cue information, incomplete learning, or a combination.
It adds a frozen-head intervention and fixed training progression to our earlier
exploratory controls. No outcome is guaranteed or selected for presentation.

## Fixed design

- Four cues, fixed vocabulary and prefix structure, dense training delays 4–16.
- NTP, NextLat lambda 0.1, 1 and 10. Lambda changes both Smooth L1 and KL weights.
- Full-sequence and response supervision, with all other settings matched.
- Five new training seeds, 1240–1244: 40 models in total.
- Prespecified budgets of 500, 2,000 and 6,000 updates. Complete each budget
  across all cells before proceeding to the next. Preserve each stage.
- Same two-layer/128-width backbone, batch 32, float32 with TF32 disabled. CUDA
  arithmetic need not reproduce the earlier CPU weights bitwise.
- Evaluate every trained delay plus 24,32,48,64. The final delay-64 endpoint is
  primary; report all delays and budgets. Do not select the best test checkpoint.
- Require >=95% native recall at every trained delay before interpreting a model
  as fully learned. Report failed seeds rather than discarding them.

New training seeds do not imply wholly new evaluation noise families. The original
hash partitions are unchanged and can overlap previously viewed evaluation
families, particularly at short delays. This is an exploratory follow-up, not an
independent held-out confirmation.

## Three separate measurements

1. **Native unrestricted answers:** recall, non-answer-token rate, and restricted
   two-answer scores as an explicitly labeled diagnostic.
2. **Before-query probes:** decode each of the four stored bits from one state
   before the query index arrives. This measures linear accessibility of memory.
3. **Frozen output-head refit:** keep every Transformer parameter and cached
   at-query state fixed. Refit only the existing bias-free 24-class output matrix
   on separate probe-fitting families at delays 4,8,16. Initialize from native
   head weights. Use mean CE plus sum((W-W_native)^2)/(2*N), fixed LBFGS settings
   (300 iterations maximum), and never use test labels for fitting or selection.
   Evaluation permits the entire vocabulary; no output token is masked.

For the head-only comparison, both native and refitted heads decode identical
cached features on CPU in float64. Original CUDA recall is separately retained,
so tiny arithmetic changes are not attributed to the intervention. At-query
states have already used full causal attention: a successful refit demonstrates
accessible answer information there, not necessarily memory in the pre-query
state. Low scores cannot establish information erasure. Masking changes loss
normalization and representation training as well as output behavior.

## Launch and inspect

Use an existing allocated CUDA GPU, activate the project Python environment,
and initialize the pinned upstream submodule. A short end-to-end execution check:

```sh
python -m memory.supervision_study --device cuda --seeds 1240 --stages 10 20 --examples 128 --out reproduction/gpu_pipeline_smoke
```

This 20-update check is an execution test, not research evidence. After it passes:

```sh
bash scripts/run_gpu_supervision.sh reproduction/gpu_supervision_v1
```

The wrapper checks CUDA and finite gradients, runs the intervention/design unit
checks, then launches the fixed study. The default plan uses 40 models and three
budgets. Measure execution time from the smoke check before estimating total
resource requirements. One allocated GPU runs models sequentially; no parallel
GPU jobs are submitted by this script.

The run writes `plan.json`, source snapshots, an environment record, resumable
model/optimizer/data-RNG checkpoints, stage-specific raw metrics and fitted
output heads. Re-run the identical command and output path to resume. A changed
plan or source is rejected. Existing completed stages remain untouched.

While training, inspect `status.json` or summarize completed cells:

```sh
python -m memory.supervision_report --run reproduction/gpu_supervision_v1
```

The report includes incomplete seed counts, learning gates, matched training-data
digest checks, native and refitted readouts, and before-query cue probes. Only
completed run+head analyses enter the tables. Retain all conditions and optimizer
diagnostics; a favorable head-refit result would support a specific readout
explanation, not a general claim that compression cannot hurt.
