#!/usr/bin/env bash
# Step 0: create the conda environment for LAM
set -x
export PATH=/usr/local/miniconda3/bin:$PATH
source /usr/local/miniconda3/etc/profile.d/conda.sh

conda create -y -n lam python=3.10 2>&1
echo "EXIT_CODE=$?"
conda run -n lam python -V
