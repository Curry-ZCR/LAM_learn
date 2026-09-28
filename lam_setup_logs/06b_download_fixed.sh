#!/usr/bin/env bash
# Step 6 (fixed): download LAM assets + weights.
#
# Problem: `python -m modelscope download ...` (as written in the README) fails on
# modelscope >= 1.40 with:
#   No module named modelscope.__main__; 'modelscope' is a package and cannot be
#   directly executed
# modelscope now ships a console-script CLI instead of a __main__ entry point.
# Fix: call the `modelscope` executable directly.
set -x
MSC=/usr/local/miniconda3/envs/lam/bin/modelscope
cd /workspace/LAM

mkdir -p ./tmp
$MSC download --repo-type model Damo_XR_Lab/LAM-assets --local-dir ./tmp/
echo "EXIT_ASSETS=$?"
ls -la ./tmp/

tar -xf ./tmp/LAM_assets.tar && echo "TAR1_OK"
tar -xf ./tmp/thirdparty_models.tar && echo "TAR2_OK"

$MSC download --repo-type model Damo_XR_Lab/LAM-20K \
    --local-dir ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/
echo "EXIT_WEIGHTS=$?"

rm -rf ./tmp/
echo "===== RESULT ====="
du -sh /workspace/LAM/model_zoo /workspace/LAM/assets 2>/dev/null
find /workspace/LAM/model_zoo -maxdepth 3 2>/dev/null | head -20
