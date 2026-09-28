#!/usr/bin/env bash
# Step 6: download the LAM assets + model weights.
# README offers HuggingFace or ModelScope. ModelScope measured ~2x faster from
# this host (15.7 MB/s vs 7.4 MB/s), and the ModelScope demo is the officially
# recommended one, so ModelScope is used here.
set -x
PY=/usr/local/miniconda3/envs/lam/bin/python
PIP=/usr/local/miniconda3/envs/lam/bin/pip
cd /workspace/LAM

$PIP install modelscope 2>&1 | tail -5
echo "EXIT_MS=$?"

# --- assets: thirdparty models, sample inputs, sample motions, pretrain models ---
$PY -m modelscope download --model "Damo_XR_Lab/LAM-assets" --local_dir "./tmp/"
echo "EXIT_ASSETS=$?"
ls -la ./tmp/

cd /workspace/LAM && tar -xf ./tmp/LAM_assets.tar
echo "EXIT_TAR1=$?"
tar -xf ./tmp/thirdparty_models.tar
echo "EXIT_TAR2=$?"

# --- LAM-20K weights ---
$PY -m modelscope download "Damo_XR_Lab/LAM-20K" \
    --local_dir "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/"
echo "EXIT_WEIGHTS=$?"

rm -rf ./tmp/
du -sh /workspace/LAM/model_zoo /workspace/LAM/assets /workspace/LAM/thirdparties 2>/dev/null
