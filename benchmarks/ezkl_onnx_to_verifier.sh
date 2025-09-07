#!/bin/bash

# Simple EZKL ONNX to Verifier Contract Generator
# Usage: ./ezkl_onnx_to_verifier.sh <onnx_file> <input_file> [output_dir]

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <onnx_file> <input_file> [output_dir]"
    echo "Example: $0 model.onnx inputs/model_input.json"
    echo "Example: $0 model.onnx inputs/model_input.json verifiers"
    exit 1
fi

ONNX_FILE="$1"
INPUT_FILE="$2"
OUTPUT_DIR="${3:-verifiers}"
MODEL_NAME=$(basename "$ONNX_FILE" .onnx)

echo "Processing: $ONNX_FILE"
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate settings
echo "Generating settings..."
ezkl gen-settings --model "$ONNX_FILE" --settings-path "$OUTPUT_DIR/settings.json"

# Compile circuit
echo "Compiling circuit..."
ezkl compile-circuit --model "$ONNX_FILE" --compiled-circuit "$OUTPUT_DIR/$MODEL_NAME.ezkl" --settings-path "$OUTPUT_DIR/settings.json"

# Generate witness
echo "Generating witness..."
ezkl gen-witness --compiled-circuit "$OUTPUT_DIR/$MODEL_NAME.ezkl" --data "$INPUT_FILE" --output "$OUTPUT_DIR/witness.json"

# Setup proving system
echo "Setting up proving system..."
ezkl setup --model "$OUTPUT_DIR/$MODEL_NAME.ezkl" --settings-path "$OUTPUT_DIR/settings.json" --output "$OUTPUT_DIR/kzg.srs"

# Generate proof
echo "Generating proof..."
ezkl prove --model "$OUTPUT_DIR/$MODEL_NAME.ezkl" --settings-path "$OUTPUT_DIR/settings.json" --witness "$OUTPUT_DIR/witness.json" --output "$OUTPUT_DIR/proof.json" --kzg "$OUTPUT_DIR/kzg.srs"

# Verify proof
echo "Verifying proof..."
ezkl verify --model "$OUTPUT_DIR/$MODEL_NAME.ezkl" --settings-path "$OUTPUT_DIR/settings.json" --proof "$OUTPUT_DIR/proof.json" --kzg "$OUTPUT_DIR/kzg.srs"

# Generate verifier contract
echo "Generating verifier contract..."
ezkl gen-sol-verifier --model "$OUTPUT_DIR/$MODEL_NAME.ezkl" --settings-path "$OUTPUT_DIR/settings.json" --output "$OUTPUT_DIR/verifier.sol"

echo "Done! Generated verifier.sol"
