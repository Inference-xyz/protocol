#!/usr/bin/env python3
"""
Generate input data for different model types.
"""

import json
import numpy as np
import argparse
import os

def generate_vision_input(output_path, batch_size=1, height=224, width=224, channels=3):
    """Generate input for vision models (MobileNet, ResNet, ViT)."""
    # Use minimal input - just zeros to reduce file size
    input_data = {
        'input_data': [[[[0.0] * width for _ in range(height)] for _ in range(channels)] for _ in range(batch_size)]
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(input_data, f)
    
    print(f"Generated vision input: {output_path}")
    print(f"Shape: [{batch_size}, {channels}, {height}, {width}]")

def generate_minimal_vision_input(output_path, batch_size=1, height=32, width=32, channels=3):
    """Generate minimal input for vision models."""
    # Use very small image size for testing
    input_data = {
        'input_data': [[[[0.0] * width for _ in range(height)] for _ in range(channels)] for _ in range(batch_size)]
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(input_data, f)
    
    print(f"Generated minimal vision input: {output_path}")
    print(f"Shape: [{batch_size}, {channels}, {height}, {width}]")

def generate_language_input(output_path, sequence_length=5):
    """Generate input for language models (GPT, BERT, T5)."""
    input_data = {
        'input_data': [list(range(sequence_length))]
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(input_data, f)
    
    print(f"Generated language input: {output_path}")
    print(f"Sequence length: {sequence_length}")

def main():
    parser = argparse.ArgumentParser(description="Generate input data for models")
    parser.add_argument("model_name", help="Model name (e.g., mobilenet_v2, gpt2, resnet18)")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--height", type=int, default=224, help="Image height (default: 224)")
    parser.add_argument("--width", type=int, default=224, help="Image width (default: 224)")
    parser.add_argument("--channels", type=int, default=3, help="Image channels (default: 3)")
    parser.add_argument("--sequence-length", type=int, default=5, help="Sequence length for language models (default: 5)")
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"inputs/{args.model_name}_input.json"
    
    # Generate input based on model type
    model_lower = args.model_name.lower()
    
    if any(keyword in model_lower for keyword in ['mobilenet', 'resnet', 'vit', 'efficientnet']):
        # Use minimal input for testing
        generate_minimal_vision_input(output_path, args.batch_size, 32, 32, args.channels)
    else:
        generate_language_input(output_path, args.sequence_length)

if __name__ == "__main__":
    main()
