#!/bin/bash
cd /workspace/lam3d
/usr/local/miniconda3/envs/lam/bin/python scripts/proto_m4_repocmp2.py 2>&1 | tail -40
