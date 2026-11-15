# Recorded experiment evidence

- `pathstar/`: one paired published G(5,5) run, seed 1234, 20,000 updates;
  sampled and greedy full-path accuracy, configs and training histories.
- `controls_2000/`: all 24 original factorial models at 2,000 updates.
- `controls_6000/`: the same 24 models after uniform continuation to 6,000
  updates, with intermediate results and source snapshots retained.
- `probes_midterm/`: frozen-model affine, ridge and official linear-head probe
  analysis; every seed, condition, inspected site and delay, figures and audits.
- `ARTIFACT_SHA256.json`: SHA-256 hashes for the original copied evidence.

The current main findings use the final 6,000-update models and three independent
training seeds. Two budgets are not 48 independent models. Standard deviations
are across training seeds, not shuffled probe fits or counterfactual family rows.

These files are immutable records. Some historical reports describe what had
finished on their creation date; the current project status and limitations are
in the root README and `docs/MIDTERM_RESULTS.md`. Historical source paths point
to the original workspace. Code outside this directory is the runnable version.

The archive includes small result files and raw training histories. Checkpoints,
feature arrays, generated datasets and fitted probe objects are excluded. Their
hashes, where recorded, document original artifacts rather than making them
available in this Git clone. `scripts/verify_recorded_results.py` checks included
files and recomputes the main probe aggregates without claiming an inference run.
