#!/usr/bin/env bash
# Step 5: compile the FaceBoxesV2 Cython extension.
# install_cu121.sh does: cd external/landmark_detection/FaceBoxesV2/utils && sh make.sh
# make.sh calls `python3 build.py build_ext --inplace`.
#
# NOTE: `python3` on this machine resolves to the SYSTEM python (/usr/bin/python3,
# no numpy/Cython). It MUST be run with the conda env's python, otherwise:
#   ModuleNotFoundError: No module named 'Cython'
# So we invoke build.py with the env interpreter directly instead of `sh make.sh`.
set -x
PY=/usr/local/miniconda3/envs/lam/bin/python
cd /workspace/LAM/external/landmark_detection/FaceBoxesV2/utils

$PY build.py build_ext --inplace 2>&1 | tail -30
echo "EXIT_FACEBOXES=${PIPESTATUS[0]}"

ls -la nms/
