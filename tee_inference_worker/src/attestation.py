import hashlib
import json
import logging
from typing import Dict, Optional

import httpx
import jwt

logger = logging.getLogger(__name__)


class AttestationError(Exception):
    pass


class GCPAttestationClient:
    METADATA_SERVER = 'http://metadata.google.internal'
    ATTESTATION_ENDPOINT = (
        '/computeMetadata/v1/instance/guest-attributes/gce-tcb-integrity'
    )
    INSTANCE_ID_ENDPOINT = '/computeMetadata/v1/instance/id'

    def __init__(self):
        self.attestation_token: Optional[str] = None
        self.attestation_claims: Optional[Dict] = None
        self.measurement: Optional[bytes] = None
        self.vm_instance_id: Optional[str] = None

    def fetch_attestation(self, audience: str = 'tee-inference-worker') -> str:
        try:
            self.vm_instance_id = self._fetch_instance_id()
            headers = {'Metadata-Flavor': 'Google'}
            params = {'audience': audience, 'format': 'full'}
            
            response = httpx.get(
                f'{self.METADATA_SERVER}{self.ATTESTATION_ENDPOINT}',
                headers=headers,
                params=params,
                timeout=10.0,
            )
            response.raise_for_status()

            self.attestation_token = response.text.strip()
            self.attestation_claims = jwt.decode(
                self.attestation_token,
                options={'verify_signature': False},
            )
            self._extract_measurement()
            return self.attestation_token

        except httpx.HTTPError as e:
            raise AttestationError(f'Failed to fetch attestation: {e}')
        except Exception as e:
            raise AttestationError(f'Attestation processing error: {e}')

    def _fetch_instance_id(self) -> str:
        try:
            headers = {'Metadata-Flavor': 'Google'}
            response = httpx.get(
                f'{self.METADATA_SERVER}{self.INSTANCE_ID_ENDPOINT}',
                headers=headers,
                timeout=5.0,
            )
            response.raise_for_status()
            return response.text.strip()
        except Exception as e:
            raise AttestationError(f'Failed to fetch instance ID: {e}')

    def _extract_measurement(self) -> None:
        if not self.attestation_claims:
            raise AttestationError('No attestation claims available')

        submods = self.attestation_claims.get('submods', {})
        confidential_compute = submods.get('confidential_space', {})
        measurement_data = {
            'swname': self.attestation_claims.get('swname', ''),
            'swversion': self.attestation_claims.get('swversion', ''),
            'platform': confidential_compute.get('support_attributes', []),
            'container': submods.get('container', {}),
        }
        measurement_json = json.dumps(measurement_data, sort_keys=True)
        self.measurement = hashlib.sha256(measurement_json.encode()).digest()

    def get_attestation_summary(self) -> Dict:
        if not self.attestation_claims:
            return {'status': 'not_available'}

        return {
            'status': 'available',
            'vm_instance_id': self.vm_instance_id,
            'measurement_hex': self.measurement.hex() if self.measurement else None,
            'issuer': self.attestation_claims.get('iss'),
            'issued_at': self.attestation_claims.get('iat'),
            'audience': self.attestation_claims.get('aud'),
            'confidential_compute_enabled': self.attestation_claims.get(
                'submods', {}
            )
            .get('confidential_space', {})
            .get('support_attributes', []),
        }


def get_attestation_client():
    return GCPAttestationClient()

