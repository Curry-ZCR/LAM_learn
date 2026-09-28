#!/usr/bin/env bash
# 最终回归：确认 Web 栈修复没有破坏命令行推理，且 app_lam.py 仍可启动。
export PATH=/usr/local/miniconda3/envs/lam/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH=/workspace/LAM
export TORCH_CUDA_ARCH_LIST="8.9"
ulimit -n 65535
PY=/usr/local/miniconda3/envs/lam/bin/python
cd /workspace/LAM

echo "########## A. 依赖版本核对 ##########"
$PY -c "
import torch, numpy, scipy, fastapi, starlette, pydantic, gradio, pytorch3d, nvdiffrast
import diff_gaussian_rasterization, simple_knn._C
print('torch        ', torch.__version__, '| cuda', torch.cuda.is_available())
print('numpy/scipy  ', numpy.__version__, scipy.__version__)
print('gradio       ', gradio.__version__)
print('web 栈       ', 'fastapi', fastapi.__version__, '| starlette', starlette.__version__, '| pydantic', pydantic.VERSION)
print('CUDA 扩展     OK')
assert starlette.__version__.startswith('0.27'), 'starlette 未降级'
assert pydantic.VERSION.startswith('2.8'), 'pydantic 未降级'
print('>>> A PASS')
" 2>&1 | grep -vE "tensorflow/|TF-TRT|cpu_feature|AVX2"

echo
echo "########## B. 命令行推理回归 ##########"
rm -f exps/videos/lam/lam_20k/status_audio.mp4
bash run_infer.sh assets/sample_input/status.png Look_In_My_Eyes > /tmp/reg2.log 2>&1
echo "Traceback 数: $(grep -c Traceback /tmp/reg2.log)"
ls -la exps/videos/lam/lam_20k/status_audio.mp4
[ -f exps/videos/lam/lam_20k/status_audio.mp4 ] && echo ">>> B PASS"

echo
echo "########## C. Web 界面启动 ##########"
setsid nohup $PY app_lam.py > /tmp/applam_final.log 2>&1 < /dev/null &
for i in $(seq 1 40); do
    sleep 2
    CODE=$($PY -c "
import requests
try: print(requests.get('http://127.0.0.1:7860/', timeout=3).status_code)
except Exception: print(0)
" 2>/dev/null)
    if [ "$CODE" = "200" ]; then
        echo "HTTP 200 —— Web 界面已就绪 (等待 ${i}0 秒)"
        $PY -c "
import requests
r=requests.get('http://127.0.0.1:7860/', timeout=5)
print('首页大小: %d bytes' % len(r.text))
print('>>> C PASS')
"
        break
    fi
done
pkill -f app_lam.py 2>/dev/null
echo "已停止 Web 服务"
