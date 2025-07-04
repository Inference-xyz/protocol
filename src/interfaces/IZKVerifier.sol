// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IZKVerifier {
    struct ProofData {
        bytes proof;
        bytes32[] publicInputs;
        bytes32 modelHash;      // Hash of the ONNX model
        bytes32 inputHash;      // Hash of the input data
        bytes32 outputHash;     // Hash of the output data
        uint256 timestamp;
    }
    
    struct VerificationResult {
        bool isValid;
        uint256 gasUsed;
        uint256 computationComplexity;
        string failureReason;
    }
    
    // Core Verification Functions
    function verifyProof(
        bytes calldata proof,
        bytes32[] calldata publicInputs
    ) external view returns (bool);
    
    function verifyProofWithMetadata(
        ProofData calldata proofData
    ) external view returns (VerificationResult memory); // Verify proof with additional metadata
    
    function batchVerifyProofs(
        ProofData[] calldata proofs
    ) external view returns (VerificationResult[] memory); // Verify multiple proofs in batch
    
    // Model and Circuit Management
    function registerModel(bytes32 modelHash, string calldata metadataURI) external;
    function isModelRegistered(bytes32 modelHash) external view returns (bool);
    function getModelMetadata(bytes32 modelHash) external view returns (string memory);
    
    // Circuit Key Management
    function setVerifyingKey(bytes32 modelHash, bytes calldata vk) external; // Set verifying key for ONNX model
    function getVerifyingKey(bytes32 modelHash) external view returns (bytes memory);
    function hasVerifyingKey(bytes32 modelHash) external view returns (bool);
    
    // Proof Cost Estimation
    function estimateVerificationGas(bytes calldata proof) external view returns (uint256);
    function getVerificationFee(bytes32 modelHash) external view returns (uint256);
    function setVerificationFee(bytes32 modelHash, uint256 fee) external;
    
    // Admin Functions
    function addTrustedProver(address prover) external;
    function removeTrustedProver(address prover) external;
    function isTrustedProver(address prover) external view returns (bool);
    function setMaxProofSize(uint256 maxSize) external;
    function setMaxPublicInputs(uint256 maxInputs) external;
    
    // View Functions
    function getSupportedModels() external view returns (bytes32[] memory);
    function getVerificationStats() external view returns (
        uint256 totalVerifications,
        uint256 successfulVerifications,
        uint256 failedVerifications
    );
    function getProofComplexity(bytes calldata proof) external view returns (uint256);
    
    // Events
    event ProofVerified(
        address indexed verifier,
        bytes32 indexed modelHash,
        bool success,
        uint256 gasUsed
    );
    event ModelRegistered(bytes32 indexed modelHash, string metadataURI);
    event VerifyingKeySet(bytes32 indexed modelHash);
    event TrustedProverUpdated(address indexed prover, bool trusted);
    event VerificationFeeUpdated(bytes32 indexed modelHash, uint256 fee);
    event BatchVerificationCompleted(uint256 proofCount, uint256 successCount);
} 