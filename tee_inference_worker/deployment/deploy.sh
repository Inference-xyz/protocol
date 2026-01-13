#!/usr/bin/env bash
set -euo pipefail

GCP_PROJECT="${GCP_PROJECT:?set GCP_PROJECT}"
GCP_ZONE="${GCP_ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-tee-inference-worker}"
MACHINE_TYPE="${MACHINE_TYPE:-n2d-standard-4}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
MODEL_PATH="${MODEL_PATH:-models/model.onnx}"
RECREATE_VM="${RECREATE_VM:-false}"
USE_CLOUD_BUILD="${USE_CLOUD_BUILD:-false}"

if ! command -v gcloud >/dev/null 2>&1; then
    echo "gcloud not found" >&2
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found" >&2
    exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "model not found: $MODEL_PATH" >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v sha256sum >/dev/null 2>&1; then
    MODEL_HASH="sha256:$(sha256sum "$MODEL_PATH" | awk '{print $1}')"
else
    MODEL_HASH="sha256:$(shasum -a 256 "$MODEL_PATH" | awk '{print $1}')"
fi

IMAGE_REF="gcr.io/$GCP_PROJECT/tee-inference:$IMAGE_TAG"
IMAGE_REPO="gcr.io/$GCP_PROJECT/tee-inference"

gcloud config set project "$GCP_PROJECT" >/dev/null
gcloud config set compute/zone "$GCP_ZONE" >/dev/null

gcloud services enable artifactregistry.googleapis.com containerregistry.googleapis.com --project "$GCP_PROJECT" >/dev/null
gcloud auth configure-docker gcr.io --quiet >/dev/null

docker build -t "$IMAGE_REF" -f docker/Dockerfile .

if [[ "$USE_CLOUD_BUILD" == "true" ]]; then
    gcloud services enable cloudbuild.googleapis.com --project "$GCP_PROJECT" >/dev/null
    CLOUD_BUILD_CFG=$(mktemp)
    cat > "$CLOUD_BUILD_CFG" <<EOF
steps:
- name: gcr.io/cloud-builders/docker
  args: ['build','-t','$IMAGE_REF','-f','docker/Dockerfile','.']
images: ['$IMAGE_REF']
EOF
    if ! gcloud builds submit --project "$GCP_PROJECT" --config "$CLOUD_BUILD_CFG" . >/dev/null; then
        rm -f "$CLOUD_BUILD_CFG"
        exit 1
    fi
    rm -f "$CLOUD_BUILD_CFG"
    DIGEST=$(gcloud container images list-tags "$IMAGE_REPO" --filter="tags:$IMAGE_TAG" --format='get(digest)' --limit=1)
    if [[ -z "$DIGEST" ]]; then
        echo "failed to get image digest" >&2
        exit 1
    fi
    IMAGE_DIGEST="$IMAGE_REPO@$DIGEST"
else
    PUSH_OK=0
    for i in {1..5}; do
        if docker push "$IMAGE_REF"; then
            PUSH_OK=1
            break
        fi
        sleep $((i * 2))
    done
    if [[ "$PUSH_OK" -eq 0 ]]; then
        echo "failed to push image" >&2
        exit 1
    fi
    IMAGE_DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' "$IMAGE_REF" 2>/dev/null || true)
    if [[ -z "$IMAGE_DIGEST" ]]; then
        DIGEST=$(gcloud container images list-tags "$IMAGE_REPO" --filter="tags:$IMAGE_TAG" --format='get(digest)' --limit=1)
        if [[ -z "$DIGEST" ]]; then
            echo "failed to get image digest" >&2
            exit 1
        fi
        IMAGE_DIGEST="$IMAGE_REPO@$DIGEST"
    fi
fi

if [[ -z "$IMAGE_DIGEST" ]]; then
    echo "failed to get image digest" >&2
    exit 1
fi

echo "$MODEL_HASH" > .model-hash
echo "$IMAGE_DIGEST" > .image-digest

SERVICE_FILE=$(mktemp)
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=TEE Inference Worker
After=docker.service
Requires=docker.service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Restart=always
RestartSec=10

Environment="IMAGE_DIGEST=$IMAGE_DIGEST"
Environment="MODEL_HASH=$MODEL_HASH"
Environment="MODEL_PATH=/app/models/model.onnx"
Environment="ENFORCE_HASH=true"

ExecStartPre=-/usr/bin/docker pull \${IMAGE_DIGEST}
ExecStartPre=-/usr/bin/docker stop inference-worker
ExecStartPre=-/usr/bin/docker rm inference-worker

ExecStart=/usr/bin/docker run \
    --rm \
    --read-only \
    --tmpfs /tmp:rw,noexec,nosuid,size=100m \
    --no-new-privileges \
    --security-opt=no-new-privileges \
    --cap-drop=ALL \
    --network=host \
    --name=inference-worker \
    -e MODEL_PATH=\${MODEL_PATH} \
    -e MODEL_HASH=\${MODEL_HASH} \
    -e ENFORCE_HASH=\${ENFORCE_HASH} \
    \${IMAGE_DIGEST}

ExecStop=/usr/bin/docker stop -t 10 inference-worker

[Install]
WantedBy=multi-user.target
EOF

STARTUP_SCRIPT=$(mktemp)
cat > "$STARTUP_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y docker.io
systemctl enable --now docker

cat > /etc/systemd/system/inference-worker.service <<'SERVICE_EOF'
$(cat "$SERVICE_FILE")
SERVICE_EOF

systemctl daemon-reload
systemctl enable inference-worker
systemctl start inference-worker
EOF

chmod +x "$STARTUP_SCRIPT"

gcloud compute firewall-rules describe tee-inference-allow-8000 >/dev/null 2>&1 || \
  gcloud compute firewall-rules create tee-inference-allow-8000 \
    --direction=INGRESS \
    --priority=1000 \
    --network=default \
    --action=ALLOW \
    --rules=tcp:8000 \
    --source-ranges=0.0.0.0/0 \
    --target-tags=tee-inference-worker >/dev/null

if gcloud compute instances describe "$VM_NAME" --zone="$GCP_ZONE" >/dev/null 2>&1; then
    if [[ "$RECREATE_VM" == "true" ]]; then
        gcloud compute instances delete "$VM_NAME" --zone="$GCP_ZONE" --quiet >/dev/null
    else
        echo "vm exists: $VM_NAME" >&2
        echo "set RECREATE_VM=true to delete and recreate" >&2
        exit 1
    fi
fi

gcloud compute instances create "$VM_NAME" \
  --zone="$GCP_ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --confidential-compute \
  --maintenance-policy=TERMINATE \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-standard \
  --network=default \
  --tags=tee-inference-worker,http-server \
  --scopes=cloud-platform \
  --metadata-from-file=startup-script="$STARTUP_SCRIPT" >/dev/null

rm -f "$STARTUP_SCRIPT" "$SERVICE_FILE"

VM_IP=$(gcloud compute instances describe "$VM_NAME" \
  --zone="$GCP_ZONE" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo "VM_IP=$VM_IP"
echo "IMAGE_DIGEST=$IMAGE_DIGEST"
echo "MODEL_HASH=$MODEL_HASH"
