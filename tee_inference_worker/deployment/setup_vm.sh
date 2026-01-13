#!/bin/bash
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root" >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y docker.io jq
systemctl enable --now docker

SERVICE_FILE="${1:-}"
if [[ -n "$SERVICE_FILE" && -f "$SERVICE_FILE" ]]; then
    cp "$SERVICE_FILE" /etc/systemd/system/inference-worker.service
    chmod 644 /etc/systemd/system/inference-worker.service
    systemctl daemon-reload
    systemctl enable inference-worker
    systemctl start inference-worker
fi
