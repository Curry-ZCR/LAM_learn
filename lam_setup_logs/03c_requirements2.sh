#!/usr/bin/env bash
# Step 3c: resume the requirements.txt install (chumpy handled in 03b, the 4
# git+ extensions are built from source in step 4).
# numpy 1.23.0 / scipy 1.10.1 are passed explicitly so the resolver cannot
# silently promote numpy to 2.x (tensorflow 2.12 requires numpy<1.24).
set -x
PIP=/usr/local/miniconda3/envs/lam/bin/pip
PY=/usr/local/miniconda3/envs/lam/bin/python
cd /workspace/LAM

$PIP install -r /workspace/lam_setup_logs/requirements_local2.txt "numpy==1.23.0" "scipy==1.10.1" 2>&1 | tail -60
echo "EXIT_REQ=${PIPESTATUS[0]}"
$PY -c "import numpy,scipy;print('numpy',numpy.__version__,'scipy',scipy.__version__)"
