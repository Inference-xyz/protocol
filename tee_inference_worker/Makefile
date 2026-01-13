# Makefile for TEE Inference Worker
# Provides convenient commands for development and deployment

.PHONY: help install test build push deploy clean

# Configuration
PROJECT ?= your-project-id
IMAGE_NAME = gcr.io/$(PROJECT)/tee-inference
IMAGE_TAG = v1
MODEL_PATH = models/model.onnx

help:
	@echo "TEE Inference Worker - Available Commands:"
	@echo ""
	@echo "  make install        - Install Python dependencies"
	@echo "  make test          - Run tests"
	@echo "  make build         - Build Docker image"
	@echo "  make push          - Push image to GCR"
	@echo "  make deploy        - Deploy to GCP Confidential VM"
	@echo "  make verify        - Verify attestation"
	@echo "  make clean         - Clean up artifacts"
	@echo "  make test-model    - Create and test simple model"
	@echo ""
	@echo "Configuration:"
	@echo "  PROJECT=$(PROJECT)"
	@echo "  IMAGE_NAME=$(IMAGE_NAME)"
	@echo "  IMAGE_TAG=$(IMAGE_TAG)"
	@echo ""

install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	cd verifier && pip install -r requirements.txt
	@echo "✓ Dependencies installed"

test-model:
	@echo "Creating test model..."
	python3 -c "from src.model_loader import create_simple_test_model; create_simple_test_model('$(MODEL_PATH)')"
	@echo "✓ Test model created at $(MODEL_PATH)"

test:
	@echo "Running tests..."
	python3 src/crypto_utils.py
	python3 src/model_loader.py
	@echo "✓ Tests passed"

build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) -f docker/Dockerfile .
	@echo "✓ Image built: $(IMAGE_NAME):$(IMAGE_TAG)"

push: build
	@echo "Pushing image to GCR..."
	docker push $(IMAGE_NAME):$(IMAGE_TAG)
	@docker inspect --format='{{index .RepoDigests 0}}' $(IMAGE_NAME):$(IMAGE_TAG) > .image-digest
	@echo "✓ Image pushed"
	@echo "Image digest: $$(cat .image-digest)"

compute-hash:
	@echo "Computing model hash..."
	@sha256sum $(MODEL_PATH) | awk '{print "sha256:" $$1}' > .model-hash
	@echo "✓ Model hash: $$(cat .model-hash)"

deploy: push compute-hash
	@echo "Deploying to GCP..."
	export GCP_PROJECT=$(PROJECT) && \
	export IMAGE_DIGEST=$$(cat .image-digest) && \
	export MODEL_HASH=$$(cat .model-hash) && \
	./deployment/gcp_deploy.sh

verify:
	@if [ ! -f .deployment-info ]; then \
		echo "Error: .deployment-info not found. Run 'make deploy' first."; \
		exit 1; \
	fi
	@. ./.deployment-info && \
	cd verifier && \
	python verify_attestation.py \
		--vm-ip $$VM_IP \
		--expected-weights-hash $$MODEL_HASH \
		--test-inference

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ src/__pycache__
	rm -f .image-digest .model-hash .deployment-info
	docker system prune -f
	@echo "✓ Cleaned up"

run-local:
	@echo "Running server locally..."
	export MODEL_PATH=$(MODEL_PATH) && \
	export ENFORCE_HASH=false && \
	python3 src/inference_server.py

docker-run:
	@echo "Running Docker container locally..."
	docker run --rm -it \
		-p 8000:8000 \
		-e MODEL_PATH=/app/$(MODEL_PATH) \
		-e ENFORCE_HASH=false \
		$(IMAGE_NAME):$(IMAGE_TAG)

status:
	@if [ -f .deployment-info ]; then \
		. ./.deployment-info && \
		echo "Deployment Status:" && \
		echo "  VM Name: $$VM_NAME" && \
		echo "  VM IP: $$VM_IP" && \
		echo "  Zone: $$GCP_ZONE" && \
		echo "  Image: $$IMAGE_DIGEST" && \
		echo "  Model: $$MODEL_HASH" && \
		echo "  Deployed: $$DEPLOYED_AT" && \
		echo "" && \
		echo "Testing health endpoint..." && \
		curl -s http://$$VM_IP:8000/health | python3 -m json.tool || echo "Failed to reach VM"; \
	else \
		echo "No deployment found. Run 'make deploy' first."; \
	fi

logs:
	@if [ -f .deployment-info ]; then \
		. ./.deployment-info && \
		gcloud compute ssh $$VM_NAME --zone=$$GCP_ZONE -- journalctl -u inference-worker -f; \
	else \
		echo "No deployment found. Run 'make deploy' first."; \
	fi

# Development shortcuts
dev-setup: install test-model
	@echo "✓ Development environment ready"

dev-test: test-model run-local

# Complete workflow
all: install test-model build push deploy verify
	@echo "✓ Complete deployment finished successfully"

