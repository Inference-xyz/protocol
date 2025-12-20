# Inference Protocol - Quickstart Guide

Welcome to the Inference Protocol, a verifiable, trust-minimized inference system that brings cryptographic guarantees to AI inference execution.

## What is Inference Protocol?

Inference Protocol (IP) is **not** OpenRouter or another GPU aggregator. IP provides something fundamentally different:

### Two Key Differentiators

**1. Verifiable Inference Execution**: TEE attestations and cryptographic signatures let you prove how and where inference was executed, not just receive an output.

**2. Native Web3 Composability**: Results, payments, rewards, and incentives live on-chain. Smart contracts can trust and act on inference results directly.

---

## Use Cases

**Verifiable Research**: Prove specific model, weights, and inputs were used for reproducible experiments and benchmarks.

**Trust-Minimized On-Chain AI**: DeFi protocols, DAOs, and on-chain agents can use verifiable AI outputs for governance, trading, and risk assessment.

**High-Stakes Auditable Inference**: Financial compliance, enterprise workflows, and regulated use cases requiring independent verification of execution.

---

## Architecture Overview

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Solver    │────────▶│ TEE Worker   │────────▶│  Blockchain │
│             │ Deploy  │ (Container   │ Submit  │  (Contest   │
│             │ Image   │  runs once)  │ Result  │   Contract) │
└─────────────┘         └──────────────┘         └─────────────┘
                              │
                              │ Attestation (startup)
                              ▼
                       ┌──────────────┐
                       │ GCP Metadata │
                       │   Server     │
                       └──────────────┘
```

**Execution Model:**

Container runs inference once at startup: build image → deploy to TEE → fetch attestation → derive key → run inference → sign result → exit.

**Security:**

- Immutable container (digest-pinned), fixed entrypoint, read-only filesystem
- No SSH, no network access (except metadata for attestation)
- TEE proves what image ran; signature proves result came from that TEE
- ❌ No interactive inference, runtime inputs, or API endpoints

**Verification:**

Check attestation JWT (Google-signed), image digest, model hash, input hash, and signature validity.

---

## Getting Started

### Prerequisites

- GCP account with billing enabled
- `gcloud` CLI installed and authenticated
- `docker` installed
- Python 3.8+ (for verification)

### Quick Start

**1. Set up GCP project**

```bash
# Authenticate with GCP
gcloud auth login
gcloud auth application-default login

# Set your project
export GCP_PROJECT="your-project-id"
export GCP_ZONE="us-central1-a"  # AMD zones required for SEV-SNP
```

**2. Prepare model and inputs**

```bash
cd tee_inference_worker

# Option A: Use existing model and inputs
# Copy your ONNX model to models/model.onnx
# Place input data in a predefined location (e.g., inputs/input.json)

# Option B: Create test model
python -c "from src.model_loader import create_simple_test_model; create_simple_test_model('models/model.onnx')"

# Create test input data (baked into container)
mkdir -p inputs
python -c "import json; json.dump([float(i % 100) / 10.0 for i in range(512)], open('inputs/input.json', 'w'))"
```

**3. Build Docker image**

```bash
# Build the container image with model and inputs baked in
docker build -t gcr.io/$GCP_PROJECT/tee-inference:v1 -f docker/Dockerfile .

# Compute and record the image digest (CRITICAL for verification)
IMAGE_DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' gcr.io/$GCP_PROJECT/tee-inference:v1)
echo "Image Digest: $IMAGE_DIGEST"
echo "$IMAGE_DIGEST" > .image-digest

# Compute model hash
MODEL_HASH="sha256:$(sha256sum models/model.onnx | awk '{print $1}')"
echo "Model Hash: $MODEL_HASH"
echo "$MODEL_HASH" > .model-hash
```

**4. Push image to GCR**

```bash
# Configure Docker for GCR
gcloud auth configure-docker

# Push image
docker push gcr.io/$GCP_PROJECT/tee-inference:v1
```

**5. Deploy to GCP Confidential VM**

```bash
# Deploy with real TEE attestation
# This creates a Confidential VM that runs the container once
./deployment/gcp_deploy.sh
```

The script creates a Confidential VM, starts the container (runs inference once), and outputs the signed result.

**6. Retrieve and verify result**

```bash
# Get result from VM logs
VM_NAME="tee-inference-worker"
gcloud compute instances get-serial-port-output $VM_NAME \
    --zone=$GCP_ZONE | grep -A 100 "Inference Result"

# Verify attestation and signature
python verifier/verify_attestation.py \
    --attestation-report <attestation-jwt> \
    --expected-image-digest "$(cat .image-digest)" \
    --expected-weights-hash "$(cat .model-hash)" \
    --input-hash <input-hash> \
    --output-hash <output-hash> \
    --signature <signature> \
    --public-key <public-key>
```

### Customization

You can customize the deployment by setting environment variables:

```bash
export GCP_PROJECT="your-project-id"
export GCP_ZONE="us-central1-a"
export VM_NAME="my-tee-worker"
export MACHINE_TYPE="n2d-standard-4"  # AMD required
export MODEL_PATH="models/model.onnx"
export MODEL_HASH="sha256:..."  # Auto-computed if not set

./deployment/gcp_deploy.sh
```

---

## Smart Contracts

Solver workflow:
1. Build container with model/inputs → deploy to TEE
2. Retrieve signed result (outputHash + signature)
3. Submit to contest contract

```solidity
teeContest.submitResult(contestId, outputHash, verifierSignature);
```

See `protocol/contracts/tee_contest.sol` for contract interface.

