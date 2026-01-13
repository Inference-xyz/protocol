import base64
import hashlib
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attestation import AttestationError, get_attestation_client
from src.crypto_utils import TEECrypto
from src.model_loader import ModelLoadError, VerifiedModelLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get('MODEL_PATH', '/app/models/model.onnx')
MODEL_HASH = os.environ.get('MODEL_HASH', None)
ENFORCE_HASH = os.environ.get('ENFORCE_HASH', 'true').lower() == 'true'

app = FastAPI(
    title='TEE Inference Worker',
    description='Confidential inference with attestation and signing',
    version='1.0.0',
)

model_loader: VerifiedModelLoader = None
attestation_client = None
crypto: TEECrypto = None


class InferenceRequest(BaseModel):
    input: List[float] = Field(..., min_items=1)


class InferenceResponse(BaseModel):
    output: List[float]
    input_hash: str
    output_hash: str
    signature: str
    public_key: str
    timestamp: str
    attestation_available: bool


class AttestationResponse(BaseModel):
    attestation_report: str
    public_key: str
    model_hash: str
    vm_instance_id: str
    attestation_summary: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    attestation_available: bool
    model_info: Dict[str, Any]


@app.on_event('startup')
async def startup_event():
    global model_loader, attestation_client, crypto

    try:
        model_loader = VerifiedModelLoader(
            model_path=MODEL_PATH,
            expected_hash=MODEL_HASH,
            enforce_hash=ENFORCE_HASH,
        )
        model_loader.load_model()
    except ModelLoadError as e:
        logger.error(f'Model loading failed: {e}')
        raise RuntimeError('Model verification failed')

    try:
        attestation_client = get_attestation_client()
        attestation_client.fetch_attestation()
    except AttestationError as e:
        logger.warning(f'Attestation fetch failed: {e}')
        attestation_client = None

    if attestation_client and attestation_client.measurement:
        crypto = TEECrypto(
            attestation_measurement=attestation_client.measurement,
            vm_instance_id=attestation_client.vm_instance_id,
        )
    else:
        fallback_measurement = hashlib.sha256(
            f'{MODEL_PATH}:{model_loader.actual_hash}'.encode()
        ).digest()
        crypto = TEECrypto(
            attestation_measurement=fallback_measurement, vm_instance_id='fallback'
        )


@app.get('/health', response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status='healthy' if model_loader and crypto else 'unhealthy',
        model_loaded=model_loader is not None,
        attestation_available=attestation_client is not None,
        model_info=model_loader.get_model_info() if model_loader else {},
    )


@app.get('/attestation', response_model=AttestationResponse)
async def get_attestation():
    if not attestation_client:
        raise HTTPException(
            status_code=503, detail='Attestation not available on this instance'
        )

    if not attestation_client.attestation_token:
        raise HTTPException(status_code=500, detail='Attestation fetch failed')

    return AttestationResponse(
        attestation_report=attestation_client.attestation_token,
        public_key=base64.b64encode(crypto.public_key_bytes).decode('utf-8'),
        model_hash=f'sha256:{model_loader.actual_hash}',
        vm_instance_id=attestation_client.vm_instance_id,
        attestation_summary=attestation_client.get_attestation_summary(),
    )


@app.post('/inference', response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    try:
        input_array = np.array([request.input], dtype=np.float32)
        output_array = model_loader.run_inference(input_array)
        output_list = output_array.flatten().tolist()

        input_bytes = input_array.tobytes()
        output_bytes = output_array.tobytes()
        input_hash = crypto.compute_hash(input_bytes)
        output_hash = crypto.compute_hash(output_bytes)
        signature = crypto.sign_inference_result(input_hash, output_hash)
        timestamp = datetime.utcnow().isoformat() + 'Z'

        return InferenceResponse(
            output=output_list,
            input_hash=input_hash.hex(),
            output_hash=output_hash.hex(),
            signature=base64.b64encode(signature).decode('utf-8'),
            public_key=base64.b64encode(crypto.public_key_bytes).decode('utf-8'),
            timestamp=timestamp,
            attestation_available=attestation_client is not None,
        )

    except Exception as e:
        logger.error(f'Inference failed: {e}')
        raise HTTPException(status_code=500, detail=f'Inference failed: {str(e)}')


@app.get('/')
async def root():
    return {
        'service': 'TEE Inference Worker',
        'version': '1.0.0',
        'endpoints': {
            '/health': 'Health check',
            '/attestation': 'Get attestation report and public key',
            '/inference': 'Run inference (POST)',
        },
        'model': {
            'path': MODEL_PATH,
            'hash_enforced': ENFORCE_HASH,
            'loaded': model_loader is not None,
        },
        'security': {
            'attestation_enabled': attestation_client is not None,
            'signing_enabled': crypto is not None,
        },
    }


@app.exception_handler(ModelLoadError)
async def model_load_error_handler(request, exc):
    return JSONResponse(
        status_code=500, content={'error': 'Model load error', 'detail': str(exc)}
    )


@app.exception_handler(AttestationError)
async def attestation_error_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={'error': 'Attestation error', 'detail': str(exc)},
    )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')

