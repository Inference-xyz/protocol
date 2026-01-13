# Protocol Development Guide

A simple guide for developers working on the Inference Protocol.

## Essential Components

### Smart Contracts (`contracts/`)

**Core Contracts:**
- `Contest.sol` - Manages individual AI inference competitions
- `ContestManager.sol` - Factory contract that creates and manages Contest instances using minimal proxy pattern
- `InfToken.sol` - ERC20 token used for staking, rewards, and governance

**Purpose:** On-chain infrastructure for decentralized AI inference competitions. Participants run inference in TEE (Trusted Execution Environment), submit results with cryptographic attestation proofs, and winners are selected based on verified TEE execution.

### TEE Inference Worker (`tee_inference_worker/`)

**Purpose:** Trusted Execution Environment (TEE) worker that runs AI inference in a verifiable, attestable environment. Proves that specific models and inputs were used to generate outputs.

**Key Files:**
- `src/inference_server.py` - Main inference server
- `src/attestation.py` - TEE attestation handling
- `deployment/gcp_deploy.sh` - GCP Confidential VM deployment script
- `Makefile` - Convenient commands for building and deploying

### Tests (`test/`)

- `Contest.t.sol` - Tests for Contest contract
- `ContestManager.t.sol` - Tests for ContestManager

## Setup

### Prerequisites

```bash
# Install Foundry
curl -L https://foundry.paradigm.xyz | bash
foundryup

# Install Node.js dependencies (for deployment scripts)
npm install
```

### Build

```bash
# Install Foundry dependencies
forge install

# Build contracts
forge build

# Run tests
forge test
```

## Launch

### Deploy Smart Contracts

1. **Set up environment variables:**
   ```bash
   # Create .env file
   export SEPOLIA_RPC_URL="your-rpc-url"
   export SEPOLIA_PRIVATE_KEY="your-private-key"
   export ETHERSCAN_API_KEY="your-etherscan-key"
   ```

2. **Deploy:**
   ```bash
   # Using deploy script
   ./deploy.sh
   
   # Or manually
   forge script Deploy.s.sol:DeployScript \
     --rpc-url $SEPOLIA_RPC_URL \
     --private-key $SEPOLIA_PRIVATE_KEY \
     --broadcast \
     --verify
   ```

### Deploy TEE Worker

1. **Set up GCP:**
   ```bash
   export GCP_PROJECT="your-project-id"
   export GCP_ZONE="us-central1-a"  # AMD zones required
   gcloud auth login
   gcloud auth application-default login
   ```

2. **Build and deploy:**
   ```bash
   cd tee_inference_worker
   
   # Install dependencies
   make install
   
   # Create test model (optional)
   make test-model
   
   # Build, push, and deploy
   make deploy
   
   # Verify deployment
   make verify
   ```

## Development Workflow

### Smart Contracts

```bash
# Run specific test
forge test --match-contract ContestTest

# Run with gas reporting
forge test --gas-report

# Format code
forge fmt

# Lint
forge fmt --check
```

### TEE Worker

```bash
cd tee_inference_worker

# Local development
make dev-setup      # Install deps and create test model
make run-local      # Run server locally

# Full deployment cycle
make all            # Install → build → push → deploy → verify
```

## Project Structure

```
protocol/
├── contracts/          # Solidity smart contracts
│   ├── Contest.sol
│   ├── ContestManager.sol
│   └── InfToken.sol
├── test/              # Foundry tests
├── tee_inference_worker/  # TEE worker implementation
│   ├── src/           # Python inference server
│   ├── deployment/    # Deployment scripts
│   └── verifier/      # Attestation verification
├── Deploy.s.sol       # Deployment script
└── foundry.toml       # Foundry configuration
```

## Key Concepts

**Contest Lifecycle:**
1. Contest owner publishes container image (with model + inputs) to IPFS
2. Owner creates contest on-chain with image hash and input hash
3. Solvers download image and run it in TEE (Confidential VM)
4. TEE produces attestation report proving correct execution
5. Solvers submit results with TEE attestation signatures
6. Contract verifies attestation signatures on-chain
7. Winner selected based on first valid submission or best result

**TEE Worker:**
- Runs inference in isolated, attestable environment (GCP Confidential VM or Azure SEV-SNP)
- Generates cryptographic attestation proving:
  - Correct container image was loaded
  - Environment is trusted (SEV-SNP)
  - No runtime modifications
- Cloud attestation service validates and signs the proof
- On-chain contract verifies the signature to ensure integrity


