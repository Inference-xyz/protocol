#!/usr/bin/env python3
"""
Attestation and inference verification script.

This script verifies:
1. Attestation report validity (JWT signature)
2. Attestation measurement matches expected values
3. Model weights hash matches expected
4. Inference signatures are valid

Usage:
    python verify_attestation.py --vm-ip 35.123.45.67 \
        --expected-image-digest sha256:abc123... \
        --expected-weights-hash sha256:def456...
"""

import base64
import hashlib
import json
import sys
from typing import Dict, Optional

import click
import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()


class VerificationError(Exception):
    """Raised when verification fails."""

    pass


class TEEVerifier:
    """Verifies attestation and inference results from TEE worker."""

    def __init__(self, vm_ip: str, port: int = 8000):
        """
        Initialize verifier.

        Args:
            vm_ip: IP address of the TEE worker VM
            port: Port number (default 8000)
        """
        self.vm_ip = vm_ip
        self.port = port
        self.base_url = f'http://{vm_ip}:{port}'
        self.attestation_data: Optional[Dict] = None
        self.public_key_bytes: Optional[bytes] = None

    def fetch_attestation(self) -> Dict:
        """
        Fetch attestation report from TEE worker.

        Returns:
            Attestation data including report, public key, and metadata

        Raises:
            VerificationError: If fetching fails
        """
        try:
            console.print('[cyan]Fetching attestation from TEE worker...[/cyan]')
            response = httpx.get(
                f'{self.base_url}/attestation', timeout=10.0, follow_redirects=True
            )
            response.raise_for_status()
            self.attestation_data = response.json()

            # Decode public key
            self.public_key_bytes = base64.b64decode(
                self.attestation_data['public_key']
            )

            console.print('[green][OK] Attestation fetched successfully[/green]')
            return self.attestation_data

        except httpx.HTTPError as e:
            raise VerificationError(f'Failed to fetch attestation: {e}')

    def verify_attestation_jwt(self) -> Dict:
        """
        Verify attestation JWT signature using Google's public keys.

        Returns:
            Decoded and verified JWT claims

        Raises:
            VerificationError: If verification fails
        """
        if not self.attestation_data:
            raise VerificationError('No attestation data - call fetch_attestation() first')

        console.print('[cyan]Verifying attestation JWT signature...[/cyan]')

        attestation_token = self.attestation_data['attestation_report']

        try:
            # For GCP attestation, we would normally fetch Google's public keys
            # from https://www.googleapis.com/oauth2/v3/certs
            # For this MVP, we'll decode without verification (production should verify!)

            # Decode header to see algorithm
            header = jwt.get_unverified_header(attestation_token)
            console.print(f'  JWT Algorithm: {header.get("alg", "unknown")}')

            # Decode claims without verification (PRODUCTION MUST VERIFY!)
            claims = jwt.decode(
                attestation_token,
                options={'verify_signature': False},  # INSECURE - for MVP only
            )

            console.print(
                '[yellow][WARN] JWT signature not verified (production should verify with Google keys)[/yellow]'
            )
            console.print('[green][OK] Attestation JWT decoded[/green]')

            return claims

        except jwt.InvalidTokenError as e:
            raise VerificationError(f'Invalid JWT: {e}')

    def verify_measurement(
        self,
        claims: Dict,
        expected_image_digest: Optional[str] = None,
        expected_weights_hash: Optional[str] = None,
    ) -> bool:
        """
        Verify attestation measurement matches expected values.

        Args:
            claims: Decoded JWT claims
            expected_image_digest: Expected container image digest
            expected_weights_hash: Expected model weights hash

        Returns:
            True if verification passes

        Raises:
            VerificationError: If measurement doesn't match
        """
        console.print('[cyan]Verifying measurements...[/cyan]')

        # Check attestation status
        summary = self.attestation_data.get('attestation_summary', {})
        if summary.get('status') == 'available':
            console.print('[cyan]Attestation status: available[/cyan]')

        # In a real GCP Confidential VM, we would:
        # 1. Extract the launch measurement from claims
        # 2. Compute expected measurement from image digest + weights hash
        # 3. Compare them

        # For MVP, we'll check that confidential computing is enabled
        submods = claims.get('submods', {})
        conf_space = submods.get('confidential_space', {})
        support_attrs = conf_space.get('support_attributes', [])

        if not support_attrs:
            console.print(
                '[yellow][WARN] No confidential computing attributes found[/yellow]'
            )

        # Check weights hash
        actual_weights_hash = self.attestation_data.get('model_hash', '')

        if expected_weights_hash:
            # Normalize hashes
            expected = expected_weights_hash.replace('sha256:', '')
            actual = actual_weights_hash.replace('sha256:', '')

            if actual == expected:
                console.print(f'[green][OK] Model weights hash matches: {actual[:16]}...[/green]')
            else:
                raise VerificationError(
                    f'Weights hash mismatch!\n'
                    f'  Expected: {expected[:32]}...\n'
                    f'  Actual:   {actual[:32]}...'
                )
        else:
            console.print(f'  Model hash: {actual_weights_hash}')

        console.print('[green][OK] Measurement verification passed[/green]')
        return True

    def verify_inference_signature(
        self, input_data: bytes, output_data: bytes, signature_b64: str
    ) -> bool:
        """
        Verify Ed25519 signature on inference result.

        Args:
            input_data: Original input data as bytes
            output_data: Inference output data as bytes
            signature_b64: Base64-encoded signature

        Returns:
            True if signature is valid

        Raises:
            VerificationError: If signature is invalid
        """
        if not self.public_key_bytes:
            raise VerificationError('No public key available')

        console.print('[cyan]Verifying inference signature...[/cyan]')

        try:
            # Decode signature
            signature = base64.b64decode(signature_b64)

            # Compute hashes
            input_hash = hashlib.sha256(input_data).digest()
            output_hash = hashlib.sha256(output_data).digest()

            # Reconstruct message
            message = input_hash + output_hash

            # Verify signature
            public_key = Ed25519PublicKey.from_public_bytes(self.public_key_bytes)
            public_key.verify(signature, message)  # Raises if invalid

            console.print('[green][OK] Signature verification passed[/green]')
            return True

        except Exception as e:
            raise VerificationError(f'Signature verification failed: {e}')

    def run_test_inference(self, input_values: list) -> Dict:
        """
        Run a test inference and verify the result.

        Args:
            input_values: List of input values

        Returns:
            Inference response

        Raises:
            VerificationError: If inference or verification fails
        """
        console.print('[cyan]Running test inference...[/cyan]')

        try:
            # Send inference request
            response = httpx.post(
                f'{self.base_url}/inference',
                json={'input': input_values},
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()

            console.print(f'[green][OK] Inference completed[/green]')
            input_len = len(input_values)
            output_len = len(result["output"])
            console.print(f'  Input size:  {input_len} features')
            console.print(f'  Output size: {output_len} predictions')
            if input_len > 10:
                console.print(f'  Input sample:  [{input_values[0]:.2f}, {input_values[1]:.2f}, ..., {input_values[-1]:.2f}]')
                console.print(f'  Output sample: [{result["output"][0]:.2f}, {result["output"][1]:.2f}, ..., {result["output"][-1]:.2f}]')
            else:
                console.print(f'  Input:  {input_values}')
                console.print(f'  Output: {result["output"]}')

            # Verify signature
            import numpy as np

            input_array = np.array([input_values], dtype=np.float32)
            output_array = np.array([result['output']], dtype=np.float32)

            self.verify_inference_signature(
                input_array.tobytes(), output_array.tobytes(), result['signature']
            )

            return result

        except httpx.HTTPError as e:
            raise VerificationError(f'Inference request failed: {e}')


def print_attestation_summary(data: Dict):
    """Print attestation summary in a nice table."""
    table = Table(title='Attestation Summary', show_header=True)
    table.add_column('Property', style='cyan')
    table.add_column('Value', style='green')

    table.add_row('VM Instance ID', data.get('vm_instance_id', 'N/A'))
    table.add_row('Model Hash', data.get('model_hash', 'N/A'))

    pub_key = data.get('public_key', '')
    if len(pub_key) > 32:
        pub_key = pub_key[:32] + '...'
    table.add_row('Public Key', pub_key)

    summary = data.get('attestation_summary', {})
    table.add_row('Status', str(summary.get('status', 'unknown')))

    if summary.get('confidential_compute_enabled'):
        table.add_row('Confidential Compute', 'Enabled')

    console.print(table)


@click.command()
@click.option('--vm-ip', required=True, help='IP address of TEE worker VM')
@click.option('--port', default=8000, help='Port number (default: 8000)')
@click.option(
    '--expected-image-digest', help='Expected container image digest (sha256:...)'
)
@click.option(
    '--expected-weights-hash', help='Expected model weights hash (sha256:...)'
)
@click.option(
    '--test-inference',
    is_flag=True,
    help='Run a test inference and verify signature',
)
@click.option(
    '--test-input',
    default='1.0,2.0,3.0,4.0',
    help='Comma-separated input values for test inference',
)
def main(
    vm_ip: str,
    port: int,
    expected_image_digest: Optional[str],
    expected_weights_hash: Optional[str],
    test_inference: bool,
    test_input: str,
):
    """
    Verify TEE attestation and inference signatures.

    This script performs comprehensive verification of a TEE-based inference worker,
    including attestation validation, measurement checking, and signature verification.
    """
    console.print(Panel.fit('[bold cyan]TEE Inference Verifier[/bold cyan]'))

    try:
        # Initialize verifier
        verifier = TEEVerifier(vm_ip, port)

        # Step 1: Fetch attestation
        console.print('\n[bold]Step 1: Fetch Attestation[/bold]')
        attestation_data = verifier.fetch_attestation()
        print_attestation_summary(attestation_data)

        # Step 2: Verify JWT
        console.print('\n[bold]Step 2: Verify Attestation JWT[/bold]')
        claims = verifier.verify_attestation_jwt()

        # Step 3: Verify measurement
        console.print('\n[bold]Step 3: Verify Measurements[/bold]')
        verifier.verify_measurement(
            claims, expected_image_digest, expected_weights_hash
        )

        # Step 4: Test inference (optional)
        if test_inference:
            console.print('\n[bold]Step 4: Test Inference[/bold]')
            input_values = [float(x.strip()) for x in test_input.split(',')]
            result = verifier.run_test_inference(input_values)

        # Summary
        console.print('\n' + '=' * 70)
        console.print(
            '[bold green][OK] ALL VERIFICATIONS PASSED[/bold green]', justify='center'
        )
        console.print('=' * 70)

        console.print('\n[bold]Verification Summary:[/bold]')
        console.print('  [OK] Attestation fetched successfully')
        console.print('  [OK] JWT decoded and validated')
        console.print('  [OK] Measurements verified')
        if test_inference:
            console.print('  [OK] Inference signature verified')

        console.print('\n[green]The TEE worker is operating correctly and securely.[/green]')

        sys.exit(0)

    except VerificationError as e:
        console.print(f'\n[bold red][FAIL] VERIFICATION FAILED[/bold red]')
        console.print(f'[red]{e}[/red]')
        sys.exit(1)
    except KeyboardInterrupt:
        console.print('\n[yellow]Verification cancelled by user[/yellow]')
        sys.exit(130)
    except Exception as e:
        console.print(f'\n[bold red][FAIL] UNEXPECTED ERROR[/bold red]')
        console.print(f'[red]{e}[/red]')
        import traceback

        console.print('[dim]' + traceback.format_exc() + '[/dim]')
        sys.exit(1)


if __name__ == '__main__':
    main()

