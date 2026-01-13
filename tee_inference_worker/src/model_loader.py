import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    pass


class VerifiedModelLoader:
    def __init__(
        self,
        model_path: str,
        expected_hash: Optional[str] = None,
        enforce_hash: bool = True,
    ):
        self.model_path = Path(model_path)
        self.expected_hash = self._normalize_hash(expected_hash)
        self.enforce_hash = enforce_hash
        self.actual_hash: Optional[str] = None
        self.model: Optional[Any] = None
        self.session: Optional[ort.InferenceSession] = None

    @staticmethod
    def _normalize_hash(hash_str: Optional[str]) -> Optional[str]:
        if hash_str is None:
            return None
        hash_str = hash_str.strip()
        if hash_str.startswith('sha256:'):
            return hash_str[7:]
        return hash_str

    def _compute_hash(self) -> str:
        if not self.model_path.exists():
            raise ModelLoadError(f'Model file not found: {self.model_path}')
        try:
            sha256 = hashlib.sha256()
            with open(self.model_path, 'rb') as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            raise ModelLoadError(f'Failed to compute hash: {e}')

    def verify_hash(self) -> bool:
        self.actual_hash = self._compute_hash()
        if self.expected_hash is None:
            return True
        if self.actual_hash == self.expected_hash:
            return True
        else:
            error_msg = (
                f'Model hash mismatch!\n'
                f'  Expected: sha256:{self.expected_hash}\n'
                f'  Actual:   sha256:{self.actual_hash}'
            )
            if self.enforce_hash:
                raise ModelLoadError(error_msg)
            return False

    def load_model(self) -> ort.InferenceSession:
        self.verify_hash()
        try:
            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.add_session_config_entry(
                'session.intra_op.allow_spinning', '0'
            )
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=['CPUExecutionProvider'],
            )
            return self.session
        except Exception as e:
            raise ModelLoadError(f'Failed to load model: {e}')

    def get_model_info(self) -> dict:
        if self.session is None:
            return {'status': 'not_loaded'}
        model_path_display = str(self.model_path)
        if model_path_display.startswith(os.getcwd()):
            model_path_display = os.path.relpath(self.model_path)
        return {
            'status': 'loaded',
            'model_path': model_path_display,
            'actual_hash': f'sha256:{self.actual_hash}' if self.actual_hash else None,
            'expected_hash': (
                f'sha256:{self.expected_hash}' if self.expected_hash else None
            ),
            'hash_verified': self.actual_hash == self.expected_hash,
            'input_names': [inp.name for inp in self.session.get_inputs()],
            'output_names': [out.name for out in self.session.get_outputs()],
            'input_shapes': [inp.shape for inp in self.session.get_inputs()],
            'output_shapes': [out.shape for out in self.session.get_outputs()],
        }

    def run_inference(self, input_data: np.ndarray) -> np.ndarray:
        if self.session is None:
            raise ModelLoadError('Model not loaded - call load_model() first')
        try:
            input_name = self.session.get_inputs()[0].name
            input_data = input_data.astype(np.float32)
            outputs = self.session.run(None, {input_name: input_data})
            return outputs[0]
        except Exception as e:
            raise ModelLoadError(f'Inference failed: {e}')


def create_simple_test_model(output_path: str = 'models/test_model.onnx') -> str:
    try:
        import onnx
        from onnx import TensorProto, helper

        input_size = 512
        input_tensor = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, [None, input_size]
        )
        output_tensor = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, [None, input_size]
        )

        weight_value = np.full((1, input_size), 2.0, dtype=np.float32)
        bias_value = np.full((input_size,), 1.0, dtype=np.float32)

        weight = helper.make_tensor('weight', TensorProto.FLOAT, [1, input_size], weight_value)
        bias = helper.make_tensor('bias', TensorProto.FLOAT, [input_size], bias_value)

        mul_node = helper.make_node('Mul', ['input', 'weight'], ['mul_out'])
        add_node = helper.make_node('Add', ['mul_out', 'bias'], ['output'])

        graph = helper.make_graph(
            [mul_node, add_node],
            'simple_model',
            [input_tensor],
            [output_tensor],
            [weight, bias],
        )

        model = helper.make_model(
            graph, 
            producer_name='test',
            ir_version=9,
            opset_imports=[helper.make_opsetid("", 21)]
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        onnx.save(model, output_path)

        sha256 = hashlib.sha256()
        with open(output_path, 'rb') as f:
            sha256.update(f.read())

        return sha256.hexdigest()

    except ImportError:
        raise ModelLoadError('onnx package required to create test model')

