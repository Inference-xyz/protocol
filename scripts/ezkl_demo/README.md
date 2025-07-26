# Demo v1

This demo demonstrates how to generate cryptographic hashes for machine learning models, inputs, and outputs using the EZKL (Efficient Zero-Knowledge Learning) framework. This is useful for creating verifiable ML computations on blockchain.

## What it does

The demo performs the following steps:

1. **Model Generation**: Creates a simple neural network and exports it to ONNX format
2. **Hash Generation**: Computes Keccak256 hashes for:
   - The ONNX model file
   - Input tensor data
   - Model output tensor
3. **File Output**: Saves all hashes to text files for use in smart contracts or verification systems

## Prerequisites

Install the required Python packages:

```bash
pip install torch onnx onnxruntime pysha3 numpy
```

## How to run

1. **Generate the model and input**:
   ```bash
   python setup.py
   ```
   This creates:
   - `model.onnx`: The neural network model in ONNX format
   - `input.pt`: A random input tensor for testing

2. **Run inference and generate hashes**:
   ```bash
   python run_and_hash.py
   ```
   This creates:
   - `model_hash.txt`: Keccak256 hash of the model file
   - `input_hash.txt`: Keccak256 hash of the input tensor
   - `output_hash.txt`: Keccak256 hash of the model output

## Output files

- `model.onnx`: Neural network model in ONNX format
- `input.pt`: PyTorch tensor file containing test input
- `model_hash.txt`: Cryptographic hash of the model file
- `input_hash.txt`: Cryptographic hash of the input data
- `output_hash.txt`: Cryptographic hash of the model output

## On-chain Workflow

The demo also demonstrates a complete on-chain workflow for verifiable ML computations:

### 1. Model Registration
- **Model Owner**: Registers a model hash on-chain using `registerModelHash()`
- The model hash serves as a unique identifier for the ML model
- Only registered models can be used for jobs

### 2. Job Posting
- **Client**: Posts a job with:
  - Model hash (must be registered)
  - Input hashes (Keccak256 hashes of input data)
  - Payment amount and token
- The client deposits payment tokens to the smart contract
- Job is created and available for claiming

### 3. Job Execution
- **Compute Provider**: Claims the job using `claimJob()`
- Provider receives the job details (model hash, input hashes)
- **Input Delivery**: The provider must obtain the actual input data off-chain
- Provider runs inference using the model and inputs
- Provider generates ZK proofs proving correct computation
- Provider encrypts the output for privacy

### 4. Job Completion
- **Compute Provider**: Calls `completeJob()` with:
  - Encrypted output blob
  - ZK proof of correct computation
  - Public inputs for verification
- Smart contract verifies the ZK proof using the input hashes
- If verification passes, payment is released to the provider

### 5. Verification
- ZK proofs are verified on-chain using the registered model hash and input hashes
- The proof demonstrates that the output was computed correctly from the given inputs
- No need to trust the compute provider - the proof guarantees correctness

## Use case

These hashes can be used in smart contracts to verify that:
- A specific model was used for computation
- Specific input data was processed
- The expected output was produced

This enables trustless verification of ML computations on blockchain networks. 