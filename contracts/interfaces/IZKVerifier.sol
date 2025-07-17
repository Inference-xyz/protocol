// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IZKVerifier {
    struct ProofData {
        bytes proof;
        bytes32[] publicInputs;
        bytes32 modelHash;
        bytes32 inputHash;
        bytes32 outputHash;
        uint256 timestamp;
        ProofType proofType;
    }
    
    enum ProofType {
        Inference
    }
    
    struct VerificationResult {
        bool isValid;
        uint256 gasUsed;
        uint256 computationComplexity;
        string failureReason;
        ProofType proofType;
    }
    
    // Core verification functions
    function verifyProof(
        bytes calldata proof,
        bytes32[] calldata publicInputs
    ) external view returns (bool);
    
    function verifyProofWithMetadata(
        ProofData calldata proofData
    ) external view returns (VerificationResult memory);
    
    // Model info
    function isModelRegistered(bytes32 modelHash) external view returns (bool);
    function getModelMetadata(bytes32 modelHash) external view returns (string memory);
    
    // Key management
    function getVerifyingKey(bytes32 modelHash) external view returns (bytes memory);
    function hasVerifyingKey(bytes32 modelHash) external view returns (bool);
    
    // Gas estimation
    function estimateVerificationGas(bytes calldata proof) external view returns (uint256);
    
    // Events
    event ProofVerified(
        address indexed verifier,
        bytes32 indexed modelHash,
        bool success,
        uint256 gasUsed,
        ProofType proofType
    );
} 