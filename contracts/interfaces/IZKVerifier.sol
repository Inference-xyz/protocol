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
        ProofType proofType;
    }
    
    enum ProofType {
        Inference,
        Scoring
    }
    
    struct VerificationResult {
        bool isValid;
        uint256 gasUsed;
        uint256 computationComplexity;
        string failureReason;
        ProofType proofType;
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
    
    function verifyScoringProof(
        bytes calldata proof,
        bytes32[] calldata publicInputs,
        bytes32 scoringModelHash
    ) external view returns (bool);
    
    function verifyScoringProofWithMetadata(
        ProofData calldata proofData
    ) external view returns (VerificationResult memory);
    
    // Model and Circuit Management
    function registerModel(bytes32 modelHash, string calldata metadataURI) external;
    function registerScoringModel(bytes32 modelHash, string calldata metadataURI) external;
    function isModelRegistered(bytes32 modelHash) external view returns (bool);
    function isScoringModelRegistered(bytes32 modelHash) external view returns (bool);
    function getModelMetadata(bytes32 modelHash) external view returns (string memory);
    function getScoringModelMetadata(bytes32 modelHash) external view returns (string memory);
    
    // Circuit Key Management
    function setVerifyingKey(bytes32 modelHash, bytes calldata vk) external; // Set verifying key for ONNX model
    function setScoringVerifyingKey(bytes32 modelHash, bytes calldata vk) external; // Set verifying key for scoring model
    function getVerifyingKey(bytes32 modelHash) external view returns (bytes memory);
    function getScoringVerifyingKey(bytes32 modelHash) external view returns (bytes memory);
    function hasVerifyingKey(bytes32 modelHash) external view returns (bool);
    function hasScoringVerifyingKey(bytes32 modelHash) external view returns (bool);
    
    // Proof Cost Estimation
    function estimateVerificationGas(bytes calldata proof) external view returns (uint256);
    function estimateScoringVerificationGas(bytes calldata proof) external view returns (uint256);
    function getVerificationFee(bytes32 modelHash) external view returns (uint256);
    function getScoringVerificationFee(bytes32 modelHash) external view returns (uint256);
    function setVerificationFee(bytes32 modelHash, uint256 fee) external;
    function setScoringVerificationFee(bytes32 modelHash, uint256 fee) external;
    
    // Admin Functions
    function addTrustedProver(address prover) external;
    function removeTrustedProver(address prover) external;
    function isTrustedProver(address prover) external view returns (bool);
    function addTrustedScoringProver(address prover) external;
    function removeTrustedScoringProver(address prover) external;
    function isTrustedScoringProver(address prover) external view returns (bool);
    function setMaxProofSize(uint256 maxSize) external;
    function setMaxPublicInputs(uint256 maxInputs) external;
    function setMaxScoringProofSize(uint256 maxSize) external;
    function setMaxScoringPublicInputs(uint256 maxInputs) external;
    
    // View Functions
    function getSupportedModels() external view returns (bytes32[] memory);
    function getSupportedScoringModels() external view returns (bytes32[] memory);
    function getVerificationStats() external view returns (
        uint256 totalVerifications,
        uint256 successfulVerifications,
        uint256 failedVerifications
    );
    function getScoringVerificationStats() external view returns (
        uint256 totalVerifications,
        uint256 successfulVerifications,
        uint256 failedVerifications
    );
    function getProofComplexity(bytes calldata proof) external view returns (uint256);
    function getScoringProofComplexity(bytes calldata proof) external view returns (uint256);
    
    // Events
    event ProofVerified(
        address indexed verifier,
        bytes32 indexed modelHash,
        bool success,
        uint256 gasUsed,
        ProofType proofType
    );
    event ModelRegistered(bytes32 indexed modelHash, string metadataURI, ProofType proofType);
    event VerifyingKeySet(bytes32 indexed modelHash, ProofType proofType);
    event TrustedProverUpdated(address indexed prover, bool trusted, ProofType proofType);
    event VerificationFeeUpdated(bytes32 indexed modelHash, uint256 fee, ProofType proofType);
    event BatchVerificationCompleted(uint256 proofCount, uint256 successCount, ProofType proofType);
} 