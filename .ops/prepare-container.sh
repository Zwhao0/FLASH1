#!/bin/bash
set -euo pipefail
cd /home/campus/zhangweihao/FLASH1
./docker-gh create --name greyhound-seed --entrypoint bash tianyuanwu/greyhound:ae >/dev/null
if [[ ! -f Greyhound/detector/build/libncclprobe.so ]]; then
  ./docker-gh cp greyhound-seed:/workspace/Greyhound/detector/build Greyhound/detector/
fi
if [[ ! -f Greyhound/detector/dist/control_plane-1.0-py3-none-any.whl ]]; then
  ./docker-gh cp greyhound-seed:/workspace/Greyhound/detector/dist Greyhound/detector/
fi
sudo -n chown -R test:test Greyhound/detector/build Greyhound/detector/dist
./docker-gh cp greyhound-seed:/workspace/Greyhound/Megatron-LM/megatron/training.py artifacts/image-training.py
if [[ ! -f Greyhound/Megatron-LM/megatron/training.py.upstream ]]; then
  cp Greyhound/Megatron-LM/megatron/training.py Greyhound/Megatron-LM/megatron/training.py.upstream
fi
diff -u Greyhound/Megatron-LM/megatron/training.py.upstream artifacts/image-training.py > results/control-hook-compatibility.patch || test "$?" = 1
./docker-gh rm greyhound-seed
./docker-gh run --rm --runtime nvidia --gpus '"device=0,1,2,3"' --network host \
 -e CUDA_VISIBLE_DEVICES=0,1,2,3 -v "$PWD/Greyhound:/workspace/Greyhound" \
 --entrypoint bash tianyuanwu/greyhound:ae -c 'python -c "import torch,redis,cvxpy,Rbeast; print(torch.__version__,torch.version.cuda); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0)); print(cvxpy.installed_solvers())"; ldd /workspace/Greyhound/detector/build/libncclprobe.so' > results/environment-check.txt 2>&1
cat results/environment-check.txt
./docker-gh info > results/docker-info.txt
