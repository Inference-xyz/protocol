// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IZKScoringVerifier {
    struct ScoringProofData {
        bytes proof;
        bytes32[] publicInputs;
        bytes32 scoringModelHash;
        bytes32 inputHash;       // Hash of participant outputs
        bytes32 scoreHash;       // Hash of computed scores
        uint256 timestamp;
    }
    
    struct ScoringVerificationResult {
        bool isValid;
        uint256 gasUsed;
        uint256[] computedScores;
        string failureReason;
    }
    
    // Core Functions
    function verifyScoringProof(
        bytes calldata proof,
        bytes32[] calldata publicInputs
    ) external view returns (bool);
    
    function verifyScoringProofWithMetadata(
        ScoringProofData calldata proofData
    ) external view returns (ScoringVerificationResult memory);
    
    function batchVerifyScoringProofs(
        ScoringProofData[] calldata proofs
    ) external view returns (ScoringVerificationResult[] memory);
    
    // Model Management
    function registerScoringModel(bytes32 modelHash, string calldata metadataURI) external;
    function setScoringVerifyingKey(bytes32 modelHash, bytes calldata vk) external;
    function isScoringModelRegistered(bytes32 modelHash) external view returns (bool);
    function getScoringModelMetadata(bytes32 modelHash) external view returns (string memory);
    
    // Circuit Key Management
    function getScoringVerifyingKey(bytes32 modelHash) external view returns (bytes memory);
    function hasScoringVerifyingKey(bytes32 modelHash) external view returns (bool);
    
    // Proof Cost Estimation
    function estimateScoringVerificationGas(bytes calldata proof) external view returns (uint256);
    function getScoringVerificationFee(bytes32 modelHash) external view returns (uint256);
    function setScoringVerificationFee(bytes32 modelHash, uint256 fee) external;
    
    // Admin Functions
    function addTrustedScoringProver(address prover) external;
    function removeTrustedScoringProver(address prover) external;
    function isTrustedScoringProver(address prover) external view returns (bool);
    function setMaxScoringProofSize(uint256 maxSize) external;
    function setMaxScoringPublicInputs(uint256 maxInputs) external;
    
    // View Functions
    function getSupportedScoringModels() external view returns (bytes32[] memory);
    function getScoringVerificationStats() external view returns (
        uint256 totalVerifications,
        uint256 successfulVerifications,
        uint256 failedVerifications
    );
    function getScoringProofComplexity(bytes calldata proof) external view returns (uint256);
    
    // Events
    event ScoringProofVerified(
        address indexed verifier,
        bytes32 indexed modelHash,
        bool success,
        uint256 gasUsed
    );
    event ScoringModelRegistered(bytes32 indexed modelHash, string metadataURI);
    event ScoringVerifyingKeySet(bytes32 indexed modelHash);
    event TrustedScoringProverUpdated(address indexed prover, bool trusted);
    event ScoringVerificationFeeUpdated(bytes32 indexed modelHash, uint256 fee);
    event BatchScoringVerificationCompleted(uint256 proofCount, uint256 successCount);
} 