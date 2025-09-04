#!/usr/bin/env python3
"""
Simple model downloader from Hugging Face registry.
Downloads and converts models to ONNX format.
"""

import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model(model_name, output_path="model.onnx"):
    """Download model and convert to ONNX."""
    logger.info(f"Downloading {model_name}...")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
    os.makedirs(output_dir, exist_ok=True)
    
    # Download tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    # Create dummy input from "hello world"
    dummy_text = "hello world"
    inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True)
    dummy_input = inputs["input_ids"]
    
    # Export to ONNX
    logger.info("Converting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=11,
        export_params=True
    )
    
    logger.info(f"Model saved as: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Download model from Hugging Face and convert to ONNX")
    parser.add_argument("model_name", help="Hugging Face model name (e.g., gpt2, microsoft/DialoGPT-small)")
    parser.add_argument("-o", "--output", default="models/model.onnx", help="Output ONNX file path (default: models/model.onnx)")
    parser.add_argument("-d", "--dir", default="models", help="Output directory prefix (default: models)")
    
    args = parser.parse_args()
    
    # Create output path with directory prefix
    if args.output == "models/model.onnx":  # Default case
        output_path = f"{args.dir}/model.onnx"
    else:
        output_path = args.output
    
    download_model(args.model_name, output_path)

if __name__ == "__main__":
    main()
