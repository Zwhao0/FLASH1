#!/bin/bash
set -euo pipefail
ROOT=/home/campus/zhangweihao/FLASH1
cd "$ROOT"
mkdir -p runtime artifacts results dataset
COPYFILE_DISABLE=1 tar -xzf greyhound-source-d641b39.tar.gz
cp -n /home/test/.codex-trash/greyhound-tokenizer-20260923/gpt2-* dataset/
sudo -n mkdir -p /opt/greyhound-docker
sudo -n tar -xzf docker-28.5.2.tgz -C /opt/greyhound-docker --strip-components=1
cat > runtime/daemon.json <<'JSON'
{
  "data-root": "/home/campus/zhangweihao/FLASH1/runtime/docker-data",
  "exec-root": "/run/greyhound-docker",
  "pidfile": "/run/greyhound-docker.pid",
  "hosts": ["unix:///run/greyhound-docker.sock"],
  "bridge": "none",
  "iptables": false,
  "ip6tables": false,
  "ip-forward": false,
  "ip-masq": false,
  "userland-proxy": false,
  "storage-driver": "overlay2",
  "runtimes": {"nvidia": {"path": "/usr/bin/nvidia-container-runtime", "runtimeArgs": []}},
  "default-runtime": "nvidia"
}
JSON
cat > runtime/greyhound-docker.service <<'UNIT'
[Unit]
Description=Isolated Docker daemon for Greyhound reproduction
After=network.target
[Service]
Type=simple
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
sudo -n install -m 644 runtime/greyhound-docker.service /etc/systemd/system/greyhound-docker.service
sudo -n systemctl daemon-reload
sudo -n systemctl start greyhound-docker
cat > docker-gh <<'WRAPPER'
#!/bin/bash
exec sudo -n /opt/greyhound-docker/docker -H unix:///run/greyhound-docker.sock "$@"
WRAPPER
chmod +x docker-gh
sleep 2
./docker-gh version
./docker-gh info > results/docker-info.txt
