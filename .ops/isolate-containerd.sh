#!/bin/bash
set -euo pipefail
cd /home/campus/zhangweihao/FLASH1
sudo -n systemctl stop greyhound-docker
cat > runtime/greyhound-containerd.service <<'UNIT'
[Unit]
Description=Dedicated containerd for Greyhound Docker
[Service]
Environment=PATH=/opt/greyhound-docker:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/opt/greyhound-docker/containerd --root /home/campus/zhangweihao/FLASH1/runtime/containerd-data --state /run/greyhound-containerd --address /run/greyhound-containerd/containerd.sock
Restart=on-failure
Delegate=yes
KillMode=process
TasksMax=infinity
[Install]
WantedBy=multi-user.target
UNIT
python3 - <<'PY'
import json
from pathlib import Path
p=Path('runtime/daemon.json')
c=json.loads(p.read_text()); c['containerd']='/run/greyhound-containerd/containerd.sock'; c['containerd-namespace']='greyhound'; c['containerd-plugins-namespace']='greyhound-plugins'; p.write_text(json.dumps(c,indent=2))
p=Path('runtime/greyhound-docker.service'); s=p.read_text().replace('After=network.target','After=network.target greyhound-containerd.service\nRequires=greyhound-containerd.service'); p.write_text(s)
PY
sudo -n install -m 644 runtime/greyhound-containerd.service runtime/greyhound-docker.service /etc/systemd/system/
sudo -n systemctl daemon-reload
sudo -n systemctl start greyhound-containerd
sudo -n systemctl start greyhound-docker
sleep 2
./docker-gh version
./docker-gh info > results/docker-info.txt
