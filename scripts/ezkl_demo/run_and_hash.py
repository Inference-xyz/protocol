import onnxruntime as ort
import numpy as np
import sha3
import torch
import os

MODEL_PATH = "model.onnx"
INPUT_PATH = "input.pt"
OUTPUT_HASH_PATH = "output_hash.txt"
MODEL_HASH_PATH = "model_hash.txt"
INPUT_HASH_PATH = "input_hash.txt"

def hash_file(file_path):
    """Generate Keccak256 hash of a file"""
    k = sha3.keccak_256()
    with open(file_path, 'rb') as f:
        k.update(f.read())
    return k.hexdigest()

def hash_tensor(tensor):
    """Generate Keccak256 hash of a tensor"""
    k = sha3.keccak_256()
    k.update(tensor.numpy().tobytes())
    return k.hexdigest()

# Generate model hash from the actual model file
print("Generating model hash...")
model_hash = hash_file(MODEL_PATH)
print("Model hash (Keccak256):", model_hash)
with open(MODEL_HASH_PATH, "w") as f:
    f.write(model_hash)

# Generate input hash from the actual input data
print("Generating input hash...")
input_tensor = torch.load(INPUT_PATH)
input_hash = hash_tensor(input_tensor)
print("Input hash (Keccak256):", input_hash)
with open(INPUT_HASH_PATH, "w") as f:
    f.write(input_hash)

# Run model inference
print("Running model inference...")
session = ort.InferenceSession(MODEL_PATH)
input_data = input_tensor.numpy()

input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_data})
output = outputs[0]

# Generate output hash
print("Generating output hash...")
output_bytes = output.tobytes()
k = sha3.keccak_256()
k.update(output_bytes)
output_hash = k.hexdigest()

print("Output hash (Keccak256):", output_hash)
with open(OUTPUT_HASH_PATH, "w") as f:
    f.write(output_hash)

print("✅ All hashes generated successfully!")
print(f"Model hash saved to: {MODEL_HASH_PATH}")
print(f"Input hash saved to: {INPUT_HASH_PATH}")
print(f"Output hash saved to: {OUTPUT_HASH_PATH}") 