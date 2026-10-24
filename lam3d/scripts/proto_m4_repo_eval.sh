#!/bin/bash
set -x
cd /workspace/lam3d
/usr/local/miniconda3/envs/lam/bin/python scripts/eval_cov_offline.py 2>&1 | tail -60
