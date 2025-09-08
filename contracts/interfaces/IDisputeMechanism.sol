// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IDisputeMechanism {
    // Events
    event DisputeInitiated(
        uint256 indexed disputeId,
        uint256 indexed jobId,
        address indexed challenger,
        address provider,
        uint256 leftBound,
        uint256 rightBound
    );

    event MoveSubmitted(
        uint256 indexed disputeId,
        address indexed player,
        uint256 midpoint,
        bytes32 hash
    );

    event DisputeResolved(
        uint256 indexed disputeId,
        bool challengerWon,
        address winner,
        uint256 bondAmount
    );

    // Structs
    struct DisputeState {
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
        uint256 createdAt;
    }

    // Core Functions
    function initiateDispute(
        uint256 jobId,
        address challenger,
        address provider,
        uint256 leftBound,
        uint256 rightBound
    ) external returns (uint256 disputeId);

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
        bytes calldata ezklProof
    ) external;

    // View Functions
    function getDispute(uint256 disputeId) external view returns (DisputeState memory);
    function isDisputeResolved(uint256 disputeId) external view returns (bool);
    function getCurrentBounds(uint256 disputeId) external view returns (uint256 left, uint256 right);
    function canSubmitMove(uint256 disputeId, address player) external view returns (bool);

    // Admin Functions
    function setEZKLVerifier(address verifier) external;
    function setMoveTimeout(uint256 timeout) external;
}