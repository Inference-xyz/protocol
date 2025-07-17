#!/usr/bin/env python3
"""
EZKL Setup Script for Compute Marketplace Demo
This script creates a simple neural network model and generates the necessary 
proving and verification keys for EZKL integration.
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import ezkl

class SimpleModel(nn.Module):
    """Simple neural network for demonstration"""
    
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(2, 4)
        self.linear2 = nn.Linear(4, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

def setup_ezkl_demo():
    """Set up EZKL demo with model, keys, and utilities"""
    
    # Create directories
    os.makedirs('ezkl_demo', exist_ok=True)
    os.chdir('ezkl_demo')
    
    print("🔧 Setting up EZKL demo...")
    
    # 1. Create and export the model
    model = SimpleModel()
    
    # Set some weights for deterministic behavior
    with torch.no_grad():
        model.linear1.weight.data = torch.tensor([
            [0.5, 0.3],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.7, 0.4]
        ])
        model.linear1.bias.data = torch.tensor([0.1, 0.2, 0.3, 0.4])
        model.linear2.weight.data = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        model.linear2.bias.data = torch.tensor([0.0])
    
    # Export model
    dummy_input = torch.randn(1, 2)
    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print("✅ Model exported to model.onnx")
    
    # 2. Generate sample input data
    sample_input = torch.tensor([[1.0, 2.0]])
    torch.save(sample_input, "input.pt")
    print("✅ Sample input saved")
    
    # 3. Generate EZKL settings
    ezkl.gen_settings("model.onnx", "settings.json")
    print("✅ EZKL settings generated")
    
    # 4. Compile the model
    ezkl.compile_circuit("model.onnx", "network.ezkl", "settings.json")
    print("✅ Circuit compiled")
    
    # 5. Generate proving and verification keys
    ezkl.setup(
        "network.ezkl",
        "vk.key",
        "pk.key",
        "settings.json"
    )
    print("✅ Keys generated")
    
    # 6. Generate witness for sample input
    witness = ezkl.gen_witness("input.pt", "network.ezkl", "witness.json")
    print("✅ Witness generated")
    
    # 7. Generate proof
    proof = ezkl.prove(
        "witness.json",
        "network.ezkl",
        "pk.key",
        "proof.json",
        "settings.json"
    )
    print("✅ Proof generated")
    
    # 8. Verify proof
    result = ezkl.verify(
        "proof.json",
        "settings.json",
        "vk.key"
    )
    print(f"✅ Proof verification: {result}")
    
    # 9. Export verification key for Solidity
    ezkl.create_evm_verifier(
        "vk.key",
        "settings.json",
        "verifier.sol"
    )
    print("✅ Solidity verifier created")
    
    print("\n🎉 EZKL setup complete!")
    print("Files created:")
    print("- model.onnx: Neural network model")
    print("- settings.json: EZKL settings") 
    print("- network.ezkl: Compiled circuit")
    print("- vk.key: Verification key")
    print("- pk.key: Proving key")
    print("- verifier.sol: Solidity verifier contract")
    
    return True

def generate_proof_for_inputs(input1_val, input2_val):
    """Generate EZKL proof for given inputs"""
    
    # Create input tensor
    input_tensor = torch.tensor([[float(input1_val), float(input2_val)]])
    
    # Save input
    torch.save(input_tensor, "current_input.pt")
    
    # Generate witness
    ezkl.gen_witness("current_input.pt", "network.ezkl", "current_witness.json")
    
    # Generate proof
    ezkl.prove(
        "current_witness.json",
        "network.ezkl", 
        "pk.key",
        "current_proof.json",
        "settings.json"
    )
    
    # Load proof
    with open("current_proof.json", "r") as f:
        proof_data = json.load(f)
    
    # Run inference to get output
    model = SimpleModel()
    with torch.no_grad():
        model.linear1.weight.data = torch.tensor([
            [0.5, 0.3],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.7, 0.4]
        ])
        model.linear1.bias.data = torch.tensor([0.1, 0.2, 0.3, 0.4])
        model.linear2.weight.data = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        model.linear2.bias.data = torch.tensor([0.0])
    
    output = model(input_tensor)
    
    return {
        "proof": proof_data,
        "output": output.item(),
        "inputs": [input1_val, input2_val]
    }

def verify_proof(proof_data):
    """Verify EZKL proof"""
    
    # Save proof data
    with open("verify_proof.json", "w") as f:
        json.dump(proof_data, f)
    
    # Verify
    result = ezkl.verify(
        "verify_proof.json",
        "settings.json",
        "vk.key"
    )
    
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prove":
        if len(sys.argv) < 4:
            print("Usage: python ezkl_setup.py prove <input1> <input2>")
            sys.exit(1)
        
        input1 = sys.argv[2]
        input2 = sys.argv[3]
        
        os.chdir('ezkl_demo')
        result = generate_proof_for_inputs(input1, input2)
        print(json.dumps(result, indent=2))
        
    elif len(sys.argv) > 1 and sys.argv[1] == "verify":
        if len(sys.argv) < 3:
            print("Usage: python ezkl_setup.py verify <proof_json>")
            sys.exit(1)
        
        proof_json = sys.argv[2]
        proof_data = json.loads(proof_json)
        
        os.chdir('ezkl_demo')
        result = verify_proof(proof_data["proof"])
        print(f"Verification result: {result}")
        
    else:
        setup_ezkl_demo() 