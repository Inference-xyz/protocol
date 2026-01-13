#!/usr/bin/env python3
import hashlib
import logging
import os
import sys

import uvicorn
from src.inference_server import app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_file_hash(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_model_hash() -> bool:
    model_path = os.environ.get('MODEL_PATH', '/app/models/model.onnx')
    expected_hash = os.environ.get('MODEL_HASH', '')
    enforce = os.environ.get('ENFORCE_HASH', 'true').lower() == 'true'

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False

    if not expected_hash:
        return True

    if expected_hash.startswith('sha256:'):
        expected_hash = expected_hash[7:]

    actual_hash = compute_file_hash(model_path)
    if actual_hash == expected_hash:
        return True
    else:
        if enforce:
            logger.error("Model hash verification failed")
            return False
        return True


def main():
    if not validate_model_hash():
        sys.exit(1)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info"
    )


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

