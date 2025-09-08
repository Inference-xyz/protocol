// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IComputeMarketplace {
    // Events
    event JobPosted(
        uint256 indexed jobId,
        address indexed client,
        bytes32 indexed modelId,
        bytes32 inputHash,
        uint256 rngSeed,
        uint256 maxTokens,
        uint256 payment,
        uint256 timeout
    );

    event ClaimSubmitted(
        uint256 indexed jobId,
        address indexed provider,
        bytes32 finalHash,
        bytes32 outputCommit,
        uint256 bond
    );

    event JobFinalized(uint256 indexed jobId, address indexed provider, uint256 payment);

    event JobDisputed(uint256 indexed jobId, address indexed challenger);

    event DisputeResolved(
        uint256 indexed jobId,
        bool providerWon,
        address winner,
        uint256 bondAmount
    );

    // Structs
    struct Job {
        uint256 id;
        address client;
        bytes32 modelId;
        bytes32 inputHash;
        uint256 rngSeed;
        uint256 maxTokens;
        uint256 payment;
        uint256 timeout;
        uint256 createdAt;
        address provider;
        bytes32 finalHash;
        bytes32 outputCommit;
        uint256 bond;
        bool finalized;
        bool disputed;
        uint256 disputePeriod;
    }

    struct Dispute {
        uint256 jobId;
        address challenger;
        address provider;
        uint256 leftBound;
        uint256 rightBound;
        uint256 currentMidpoint;
        bool providerMoveSubmitted;
        bool challengerMoveSubmitted;
        bytes32 providerHash;
        bytes32 challengerHash;
        bool resolved;
    }

    // Core Functions
    function openJob(
        bytes32 modelId,
        bytes32 inputHash,
        uint256 rngSeed,
        uint256 maxTokens,
        uint256 payment,
        address paymentToken,
        uint256 timeout
    ) external returns (uint256 jobId);

    function submitClaim(
        uint256 jobId,
        bytes32 finalHash,
        bytes32 outputCommit,
        uint256 bond
    ) external;

    function challenge(uint256 jobId) external;

    function submitMove(
        uint256 disputeId,
        uint256 midpoint,
        bytes32 hash
    ) external;

    function proveOneStep(
        uint256 disputeId,
        uint256 step,
        bytes32 hashS,
        bytes32 hashSPlus1,
        bytes calldata proof
    ) external;

    // View Functions
    function getJob(uint256 jobId) external view returns (Job memory);
    function getDispute(uint256 disputeId) external view returns (Dispute memory);
    function isJobDisputable(uint256 jobId) external view returns (bool);
    function getDisputePeriod() external view returns (uint256);

    // Admin Functions
    function setDisputePeriod(uint256 newPeriod) external;
    function setMinBond(uint256 newMinBond) external;
    function pause() external;
    function unpause() external;
}
