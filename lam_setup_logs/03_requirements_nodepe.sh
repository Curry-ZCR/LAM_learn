#!/usr/bin/env bash
# Step 3: install the remaining Python dependencies from requirements.txt
# Strategy: install everything EXCEPT the 4 git/binary packages, which are handled
# separately in step 4 (they need compilation and version pinning).
set -x
PIP=/usr/local/miniconda3/envs/lam/bin/pip
cd /workspace/LAM

# tensorflow 2.12 needs numpy<1.24 ; requirements.txt pins numpy==1.23.0 so it is consistent.
# Install the pinned scientific stack first so pip does not resolve to numpy 2.x.
$PIP install "numpy==1.23.0" "Cython<3.0" 2>&1
echo "EXIT_NUMPY=$?"

$PIP install -r requirements.txt \
     --no-deps 2>&1 | tail -40
echo "EXIT_REQ_NODEPS=$?"
