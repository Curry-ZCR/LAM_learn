# install torch 2.3.0
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu118

# install dependencies
pip install -r requirements.txt

# === If you fail to install some modules due to network connection, you can also try the following: ===
# git clone https://github.com/facebookresearch/pytorch3d.git
# pip install ./pytorch3d
# git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
# pip install ./diff-gaussian-rasterization
# git clone https://github.com/camenduru/simple-knn.git
# pip install ./simple-knn

cd external/landmark_detection/FaceBoxesV2/utils/
sh make.sh
cd ../../../../
