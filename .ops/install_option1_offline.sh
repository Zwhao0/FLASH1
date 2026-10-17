#!/bin/bash
# Offline Docker setup. Run as test on 10.10.4.1 after uploading the artifacts.
set -euo pipefail
ROOT=/home/campus/zhangweihao/FLASH1
cd "$ROOT"
mkdir -p runtime results artifacts dataset
if [[ ! -d Greyhound ]]; then tar --warning=no-unknown-keyword -xzf greyhound-source-d641b39.tar.gz; fi
if [[ ! -x /opt/greyhound-docker/docker ]]; then
  sudo -n mkdir -p /opt/greyhound-docker
  sudo -n tar -xzf docker-28.5.2.tgz -C /opt/greyhound-docker --strip-components=1
fi
cat > runtime/containerd.toml <<'TOML'
version = 2
disabled_plugins = ["io.containerd.grpc.v1.cri"]
root = "/home/campus/zhangweihao/FLASH1/runtime/containerd-data"
state = "/run/greyhound-containerd"
[grpc]
  address = "/run/greyhound-containerd/containerd.sock"
TOML
cat > runtime/daemon.json <<'JSON'
{
  "data-root": "/home/campus/zhangweihao/FLASH1/runtime/docker-data",
  "exec-root": "/run/greyhound-docker",
  "pidfile": "/run/greyhound-docker.pid",
  "hosts": ["unix:///run/greyhound-docker.sock"],
  "containerd": "/run/greyhound-containerd/containerd.sock",
  "containerd-namespace": "greyhound",
  "containerd-plugins-namespace": "greyhound-plugins",
  "bridge": "none", "iptables": false, "ip6tables": false,
  "ip-forward": false, "ip-masq": false, "userland-proxy": false,
  "storage-driver": "overlay2",
  "runtimes": {"nvidia": {"path": "/usr/bin/nvidia-container-runtime", "runtimeArgs": []}},
  "default-runtime": "nvidia"
}
JSON
cat > runtime/greyhound-containerd.service <<'UNIT'
[Unit]
Description=Dedicated containerd for Greyhound Docker
[Service]
Environment=PATH=/opt/greyhound-docker:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/opt/greyhound-docker/containerd --config /home/campus/zhangweihao/FLASH1/runtime/containerd.toml
Restart=on-failure
Delegate=yes
KillMode=process
TasksMax=infinity
[Install]
WantedBy=multi-user.target
UNIT
cat > runtime/greyhound-docker.service <<'UNIT'
[Unit]
Description=Isolated Docker daemon for Greyhound reproduction
After=network.target greyhound-containerd.service
Requires=greyhound-containerd.service
[Service]
Environment=PATH=/opt/greyhound-docker:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/opt/greyhound-docker/dockerd --config-file=/home/campus/zhangweihao/FLASH1/runtime/daemon.json
Restart=on-failure
Delegate=yes
KillMode=process
TasksMax=infinity
LimitNOFILE=1048576
[Install]
WantedBy=multi-user.target
UNIT
sudo -n install -m 644 runtime/greyhound-{containerd,docker}.service /etc/systemd/system/
sudo -n systemctl daemon-reload
sudo -n systemctl start greyhound-containerd greyhound-docker
cat > docker-gh <<'WRAPPER'
#!/bin/bash
exec sudo -n /opt/greyhound-docker/docker -H unix:///run/greyhound-docker.sock "$@"
WRAPPER
chmod +x docker-gh
for i in {1..30}; do ./docker-gh info > results/docker-info.txt 2>/dev/null && break; sleep 1; done
./docker-gh info >/dev/null
if ! ./docker-gh image inspect tianyuanwu/greyhound:ae >/dev/null 2>&1; then
  python3 verify-image.py > results/image-verification.log
  ./docker-gh load -i /home/test/greyhound-ae-linux-amd64.tar > results/docker-load.log 2>&1
fi
./docker-gh image inspect tianyuanwu/greyhound:ae --format '{{.Id}}'
