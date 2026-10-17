#!/bin/bash
set -euo pipefail
cd /home/campus/zhangweihao/FLASH1
cat > runtime/containerd.toml <<'TOML'
version = 2
disabled_plugins = ["io.containerd.grpc.v1.cri"]
root = "/home/campus/zhangweihao/FLASH1/runtime/containerd-data"
state = "/run/greyhound-containerd"
[grpc]
  address = "/run/greyhound-containerd/containerd.sock"
TOML
sed -i 's|ExecStart=/opt/greyhound-docker/containerd .*|ExecStart=/opt/greyhound-docker/containerd --config /home/campus/zhangweihao/FLASH1/runtime/containerd.toml|' runtime/greyhound-containerd.service
sudo -n install -m 644 runtime/greyhound-containerd.service /etc/systemd/system/
sudo -n systemctl kill -s KILL greyhound-docker || true
sudo -n systemctl daemon-reload
sudo -n systemctl reset-failed greyhound-containerd greyhound-docker
sudo -n systemctl restart greyhound-containerd
sudo -n systemctl start greyhound-docker
sleep 2
./docker-gh version
