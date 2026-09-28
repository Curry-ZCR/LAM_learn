#!/usr/bin/env bash
# Step 3b: fix the two problems that step 3 exposed.
#
# Problem A: `pip install chumpy==0.70` fails because chumpy's setup.py does
#   `from pip._internal.req import parse_requirements` while pip builds it in an
#   isolated env that has NO pip inside it  ->  ModuleNotFoundError: No module named 'pip'.
#   Fix: build it with --no-build-isolation so it sees the environment's pip.
#
# Problem B: chumpy declares `scipy>=0.13.0` with no upper bound, so pip picks
#   scipy 1.15.3, which drags in numpy 2.2.6 and blows away the numpy==1.23.0 that
#   both tensorflow 2.12 and the whole LAM stack need. numpy 1.x must be restored
#   and scipy pinned to a release that still supports numpy 1.23.0.
#   Fix: --no-deps for chumpy, then pin numpy/scipy explicitly.
set -x
PIP=/usr/local/miniconda3/envs/lam/bin/pip
PY=/usr/local/miniconda3/envs/lam/bin/python
cd /workspace/LAM

echo "############ FIX A: build chumpy without build isolation ############"
# --no-deps: don't let chumpy re-resolve numpy/scipy. Its only runtime needs are
# scipy.sparse / scipy.optimize, and FLAME's .pkl files merely unpickle
# chumpy.array objects, so no compiled chumpy extension is required.
$PIP install "chumpy==0.70" --no-build-isolation --no-deps 2>&1 | tail -12
echo "EXIT_CHUMPY=${PIPESTATUS[0]}"

echo "############ FIX B: pin numpy 1.x + a matching scipy ############"
# scipy 1.10.1 requires numpy>=1.19.5,<1.27.0 -> compatible with numpy 1.23.0
$PIP install "numpy==1.23.0" "scipy==1.10.1" "six" 2>&1 | tail -12
echo "EXIT_PINS=${PIPESTATUS[0]}"

$PY -c "import numpy,scipy,chumpy;print('numpy',numpy.__version__);print('scipy',scipy.__version__);print('chumpy ok')"
