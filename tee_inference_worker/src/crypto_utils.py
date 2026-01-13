import hashlib
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


class TEECrypto:
    def __init__(self, attestation_measurement: bytes, vm_instance_id: str):
        self.attestation_measurement = attestation_measurement
        self.vm_instance_id = vm_instance_id
        self._private_key = None
        self._public_key = None
        self._derive_signing_key()

    def _derive_signing_key(self) -> None:
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.vm_instance_id.encode('utf-8'),
            info=b'tee-inference-signing-key-v1',
        )
        signing_seed = hkdf.derive(self.attestation_measurement)
        self._private_key = Ed25519PrivateKey.from_private_bytes(signing_seed)
        self._public_key = self._private_key.public_key()

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self._public_key

    @property
    def public_key_bytes(self) -> bytes:
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def sign_inference_result(
        self, input_hash: bytes, output_hash: bytes
    ) -> bytes:
        message = input_hash + output_hash
        return self._private_key.sign(message)

    @staticmethod
    def compute_hash(data: bytes) -> bytes:
        return hashlib.sha256(data).digest()

    @staticmethod
    def hash_file(filepath: str) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def verify_signature(
        signature: bytes,
        input_hash: bytes,
        output_hash: bytes,
        public_key_bytes: bytes,
    ) -> bool:
        public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
        message = input_hash + output_hash
        public_key.verify(signature, message)
        return True

