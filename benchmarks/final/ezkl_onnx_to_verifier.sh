#!/bin/bash

# Simple EZKL ONNX to Verifier Contract Generator
# Usage: ./ezkl_onnx_to_verifier.sh <onnx_file> [output_dir]

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <onnx_file> [output_dir]"
    echo "Example: $0 model.onnx"
    echo "Example: $0 model.onnx verifiers"
    exit 1
fi

ONNX_FILE="$1"
OUTPUT_DIR="${2:-verifiers}"
MODEL_NAME=$(basename "$ONNX_FILE" .onnx)

echo "Processing: $ONNX_FILE"
echo "Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# Generate settings
echo "Generating settings..."
ezkl gen-settings --model "../$ONNX_FILE" --settings-path $OUTPUT_DIR/settings.json

# Compile circuit
echo "Compiling circuit..."
ezkl compile-circuit --model "../$ONNX_FILE" --compiled-circuit "$MODEL_NAME.ezkl" --settings-path $OUTPUT_DIR/settings.json

# Generate sample input
echo "Generating input..."
python3 -c "
import json
input_data = {'input_data': [[0.0] * 4]}
with open('input.json', 'w') as f:
    json.dump(input_data, f)
"

# Generate witness
echo "Generating witness..."
ezkl gen-witness --compiled-circuit "$MODEL_NAME.ezkl" --data $OUTPUT_DIR/input.json --output $OUTPUT_DIR/witness.json

# Setup proving system
echo "Setting up proving system..."
ezkl setup --model "$MODEL_NAME.ezkl" --settings-path $OUTPUT_DIR/settings.json --output $OUTPUT_DIR/kzg.srs

# Generate proof
echo "Generating proof..."
ezkl prove --model "$MODEL_NAME.ezkl" --settings-path $OUTPUT_DIR/settings.json --witness $OUTPUT_DIR/witness.json --output $OUTPUT_DIR/proof.json --kzg $OUTPUT_DIR/kzg.srs

# Verify proof
echo "Verifying proof..."
ezkl verify --model "$MODEL_NAME.ezkl" --settings-path $OUTPUT_DIR/settings.json --proof $OUTPUT_DIR/proof.json --kzg $OUTPUT_DIR/kzg.srs

# Generate verifier contract
echo "Generating verifier contract..."
ezkl gen-sol-verifier --model "$MODEL_NAME.ezkl" --settings-path $OUTPUT_DIR/settings.json --output $OUTPUT_DIR/verifier.sol

echo "Done! Generated verifier.sol"
