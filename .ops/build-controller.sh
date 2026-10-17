#!/bin/bash
set -euo pipefail
cd /home/campus/zhangweihao/FLASH1
./docker-gh run --rm --network none -v "$PWD/Greyhound:/workspace/Greyhound" \
  -w /workspace/Greyhound/detector --entrypoint python tianyuanwu/greyhound:ae \
  setup.py bdist_wheel > results/controller-wheel-build.log 2>&1
sudo -n chown -R test:test Greyhound/detector/dist
sha256sum Greyhound/detector/dist/control_plane-1.0-py3-none-any.whl
