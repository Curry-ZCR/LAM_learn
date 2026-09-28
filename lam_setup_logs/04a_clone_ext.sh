#!/usr/bin/env bash
# Step 4a: clone the 4 source-only CUDA extension dependencies
set -x
cd /workspace/LAM
mkdir -p thirdparty_src && cd thirdparty_src

clone() {
  name=$1; url=$2; branch=$3
  rm -rf "$name"
  if [ -n "$branch" ]; then
    git clone --recursive --branch "$branch" --depth 1 "$url" "$name"
  else
    git clone --recursive --depth 1 "$url" "$name"
  fi
  echo "CLONE_${name}=$?"
}

clone diff-gaussian-rasterization https://github.com/ashawkey/diff-gaussian-rasterization/ ""
clone simple-knn                   https://github.com/camenduru/simple-knn/                  ""
clone nvdiffrast                   https://github.com/ShenhanQian/nvdiffrast                 "backface-culling"
clone pytorch3d                    https://github.com/facebookresearch/pytorch3d.git         "v0.7.6"

ls -la
