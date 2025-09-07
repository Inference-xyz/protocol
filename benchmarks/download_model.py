import os
import argparse
from huggingface_hub import snapshot_download
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model(model_name, output_path="model"):
    """Download model from Hugging Face."""
    logger.info(f"Downloading {model_name}...")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
    os.makedirs(output_dir, exist_ok=True)
    
    # Download model files
    snapshot_download(
        repo_id=model_name,
        local_dir=output_path,
        local_dir_use_symlinks=False
    )
    
    logger.info(f"Model downloaded to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Download model from Hugging Face")
    parser.add_argument("model_name", help="Hugging Face model name (e.g., gpt2, google/mobilenet_v2_1.0_224)")
    parser.add_argument("-o", "--output", default="model", help="Output directory (default: model)")
    
    args = parser.parse_args()
    download_model(args.model_name, args.output)

if __name__ == "__main__":
    main()
