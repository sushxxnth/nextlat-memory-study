#!/usr/bin/env bash
# Run inside an existing GPU allocation with the project environment active.
set -euo pipefail
cd "$(dirname "$0")/.."
STUDY_OUT=${1:-reproduction/gpu_supervision_v1}
STUDY_PYTHON=${STUDY_PYTHON:-python}
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$PWD/.matplotlib"
"$STUDY_PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available(), 'No allocated CUDA GPU; nothing was launched.'
x=torch.randn(32,128,device='cuda',requires_grad=True)
x.square().mean().backward()
assert torch.isfinite(x.grad).all()
print('GPU_PREFLIGHT_OK',torch.cuda.get_device_name(),torch.__version__,flush=True)
PY
"$STUDY_PYTHON" -m unittest memory.test_supervision_study -v
"$STUDY_PYTHON" -m memory.supervision_study --device cuda --out "$STUDY_OUT"
"$STUDY_PYTHON" -m memory.supervision_report --run "$STUDY_OUT"
