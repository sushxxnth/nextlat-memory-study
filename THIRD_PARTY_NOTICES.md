# Third-party source and attribution

This project uses JaydenTeoh/NextLat as the `upstream` Git submodule:

- https://github.com/JaydenTeoh/NextLat
- Commit: `3770be6009cea2b3c455a9ce7f2ca88b504bb955`
- License: MIT, copyright Microsoft Corporation.
- Full license text: `upstream/LICENSE`.

The official GPT/NextLat models, Path-Star generator/tokenizer/evaluator, and
linear probe heads are used without source changes. Project-specific runners,
delayed-query data, controls, probes and reports are maintained outside that
submodule. `PROVENANCE.json` records selected upstream source hashes.

The affine regression and shuffle-control methodology is adapted from Shai et al.,
*Transformers Represent Belief State Geometry in their Residual Stream*,
https://arxiv.org/abs/2405.15943. We do not reproduce their HMM belief-simplex
experiment. The cue-target CE probes reuse official NextLat linear heads with
adapted targets and optimizer settings; they are not a reproduction of the
published future-token probe task.
