#!/usr/bin/env bash
# Step 7a: fetch the DINOv2 ViT-L/14-reg4 backbone that LAM's encoder loads.
#
# Problem: torch.hub's default URL
#   https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth
# downloads at ~300-350 KB/s from this host (~1 hour for 1.13 GB), and torch.hub
# writes two competing *.partial files when the model is built twice.
# Fix: pre-seed torch's hub cache from a HuggingFace mirror (measured ~6.5 MB/s),
# using the exact filename torch.hub expects, so load_state_dict_dict_from_url
# finds it already cached and skips the slow download.
set -x
DEST=/root/.cache/torch/hub/checkpoints
FILE=dinov2_vitl14_reg4_pretrain.pth
URL=https://huggingface.co/showstarpro/dinov2_vitl14_reg4_pretrain/resolve/main/dinov2_vitl14_reg4_pretrain.pth

mkdir -p $DEST
rm -f $DEST/*.partial
curl -L --retry 3 -C - -o $DEST/$FILE.tmp "$URL"
echo "CURL_EXIT=$?"
mv $DEST/$FILE.tmp $DEST/$FILE
ls -la $DEST/
echo "SIZE=$(stat -c%s $DEST/$FILE)"
