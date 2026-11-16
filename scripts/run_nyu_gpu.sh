#!/usr/bin/env bash
# Reuse the project's existing NYU CUDA overlay inside an allocated GPU session.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SCRATCH:?Run this in an allocated NYU Jupyter GPU terminal}"
export NEXTLAT_STUDY_ROOT=$PWD
export NEXTLAT_STUDY_OUT=${1:-reproduction/gpu_supervision_v1}
NEXTLAT_STUDY_IMAGE=${NEXTLAT_STUDY_IMAGE:-/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif}
NEXTLAT_STUDY_OVERLAY=${NEXTLAT_STUDY_OVERLAY:-$SCRATCH/nextlat-containers/nextlat.ext3}
test -f "$NEXTLAT_STUDY_IMAGE"
test -f "$NEXTLAT_STUDY_OVERLAY"
mkdir -p logs
singularity exec --nv --overlay "$NEXTLAT_STUDY_OVERLAY:ro" "$NEXTLAT_STUDY_IMAGE" /bin/bash -s <<'CONTAINER'
set -euo pipefail
cd "$NEXTLAT_STUDY_ROOT"
export STUDY_PYTHON=/ext3/nextlat/venv/bin/python
export NEXTLAT_EXTRA_LIBS="$SCRATCH/nextlat-supervision-extras-v1"
export UV_CACHE_DIR="$SCRATCH/nextlat-supervision-uv-cache"
# Add missing scientific packages in scratch, preserving the read-only overlay.
/ext3/nextlat/uv/uv pip install --python "$STUDY_PYTHON" --target "$NEXTLAT_EXTRA_LIBS" --no-deps \
  'numpy==2.4.4' 'scipy==1.17.1' 'scikit-learn==1.8.0' 'joblib==1.4.2' 'threadpoolctl==3.5.0'
export PYTHONPATH="$NEXTLAT_STUDY_ROOT:$NEXTLAT_EXTRA_LIBS${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="$NEXTLAT_STUDY_ROOT/.matplotlib"
export PYTHONUNBUFFERED=1
"$STUDY_PYTHON" - <<'PY'
import json, torch
from importlib.metadata import version
from pathlib import Path
assert torch.cuda.is_available(), 'No allocated CUDA GPU; stopped.'
x=torch.randn(32,128,device='cuda',requires_grad=True)
x.square().mean().backward()
assert torch.isfinite(x.grad).all()
record={'gpu':torch.cuda.get_device_name(),'packages':{name:version(name) for name in
 ['torch','lightning','numpy','scipy','scikit-learn','joblib','threadpoolctl']}}
Path('logs/gpu_environment.json').write_text(json.dumps(record,indent=2)+'\n')
print('GPU_PREFLIGHT_OK',json.dumps(record),flush=True)
PY
"$STUDY_PYTHON" -m unittest memory.test_supervision_study -v
"$STUDY_PYTHON" -m memory.supervision_study --device cuda --seeds 1240 --stages 10 20 --examples 128 --out reproduction/gpu_pipeline_smoke
"$STUDY_PYTHON" -m memory.supervision_report --run reproduction/gpu_pipeline_smoke
if [[ ${NEXTLAT_SMOKE_ONLY:-0} == 1 ]]; then exit 0; fi
bash scripts/run_gpu_supervision.sh "$NEXTLAT_STUDY_OUT"
CONTAINER
