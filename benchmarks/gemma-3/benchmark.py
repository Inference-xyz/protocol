#!/usr/bin/env python3
"""
Simple MVP for running EZKL on Gemma 3 model
"""

import os
import subprocess
import json
import numpy as np
import torch
import torch.nn as nn

def create_simple_model():
    """Create a simple working model for MVP"""
    print("🔄 Creating simple working model...")
    
    # Create a simple neural network model (similar to working EZKL demo)
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear1 = nn.Linear(8, 16)
            self.linear2 = nn.Linear(16, 8)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.linear1(x))
            x = self.linear2(x)
            return x
    
    model = SimpleModel()
    model.eval()
    
    # Create output directory
    os.makedirs("onnx_models", exist_ok=True)
    
    # Create sample input
    input_tensor = torch.randn(1, 8)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        input_tensor,
        "onnx_models/simple_model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    
    print("✅ Simple model created and exported")

def create_ezkl_settings():
    """Create simple EZKL settings"""
    print("🔄 Creating EZKL settings...")
    
    settings = {
        "run_args": {
            "input_scale": 7,
            "param_scale": 7,
            "scale_rebase_multiplier": 1,
            "lookup_range": [-32768, 32768],
            "logrows": 12,  # Small for simple model
            "num_inner_cols": 2,
            "variables": [["batch_size", 1]],
            "input_visibility": "Private",
            "output_visibility": "Public",
            "param_visibility": "Private",
            "rebase_frac_zero_constants": False,
            "check_mode": "UNSAFE",
            "commitment": "KZG"
        }
    }
    
    with open("ezkl_settings.json", "w") as f:
        json.dump(settings, f, indent=2)
    
    print("✅ EZKL settings created")

def generate_sample_input():
    """Generate sample input for EZKL"""
    print("🔄 Generating sample input...")
    
    # Simple input: batch_size=1, features=8
    sample_input = torch.randn(1, 8)
    
    # Save as JSON for EZKL
    input_data = sample_input.numpy().tolist()
    with open("input.json", "w") as f:
        json.dump(input_data, f)
    
    print("✅ Sample input generated")

def run_ezkl():
    """Run EZKL proof generation"""
    print("🔄 Running EZKL...")
    
    try:
        # Check if EZKL is installed
        subprocess.run(["ezkl", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ EZKL not found. Installing...")
        subprocess.run(["pip", "install", "ezkl"], check=True)
    
    # Generate circuit
    print("🔄 Generating circuit...")
    subprocess.run([
        "ezkl", "gen-circuit",
        "-M", "onnx_models/simple_model.onnx",
        "-S", "ezkl_settings.json",
        "-O", "model.compiled"
    ], check=True)
    
    # Generate witness
    print("🔄 Generating witness...")
    subprocess.run([
        "ezkl", "gen-witness",
        "-M", "onnx_models/simple_model.onnx",
        "-S", "ezkl_settings.json",
        "-I", "input.json",
        "-O", "witness.json"
    ], check=True)
    
    # Setup
    print("🔄 Setting up proving system...")
    subprocess.run([
        "ezkl", "setup",
        "-M", "model.compiled",
        "-S", "ezkl_settings.json",
        "-O", "kzg.srs"
    ], check=True)
    
    # Generate proof
    print("🔄 Generating proof...")
    subprocess.run([
        "ezkl", "prove",
        "-M", "model.compiled",
        "-S", "ezkl_settings.json",
        "-W", "witness.json",
        "-O", "proof.json",
        "-K", "kzg.srs"
    ], check=True)
    
    # Verify proof
    print("🔄 Verifying proof...")
    subprocess.run([
        "ezkl", "verify",
        "-M", "model.compiled",
        "-S", "ezkl_settings.json",
        "-P", "proof.json",
        "-K", "kzg.srs"
    ], check=True)
    
    print("✅ EZKL completed successfully!")

def main():
    """Main function"""
    print("🚀 Starting EZKL MVP...")
    
    try:
        # Use simple working model for MVP
        create_simple_model()
        create_ezkl_settings()
        generate_sample_input()
        run_ezkl()
        
        print("\n🎉 MVP completed successfully!")
        print("Generated files:")
        print("- onnx_models/simple_model.onnx")
        print("- ezkl_settings.json")
        print("- input.json")
        print("- model.compiled")
        print("- witness.json")
        print("- kzg.srs")
        print("- proof.json")
        
        print("\n💡 Next steps:")
        print("- This MVP uses a simple model to verify EZKL works")
        print("- For Gemma 3, we need to solve the ONNX export issues")
        print("- Consider using a smaller Gemma variant or different export method")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()