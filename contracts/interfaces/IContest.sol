// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IContest {
    enum ContestStatus {
        Active,
        Paused,
        Finalized,
        Cancelled
    }

    struct ContestInfo {
        address creator;
        string metadataURI;
        RewardSplit rewardSplit;
        uint256 startTime;
        uint256 duration;
        uint256 currentEpoch;
        ContestStatus status;
        bool isEverlasting;
        address validatorSet;
        address scoringVerifier;
    }

    struct RewardSplit {
        uint256 ownerPct;
        uint256 participantPct;
        uint256 validatorPct;
        uint256 totalPct;
    }

    struct Participant {
        address account;
        uint256 joinedAt;
        uint256 currentScore;
        uint256 totalRewards;
        uint256 lastSubmissionTime;
        bool isActive;
        uint256 stakedAmount;
    }

    // Initialization
    function initialize(
        address creator,
        string calldata metadataURI,
        uint256 creatorFeePct,
        uint256 duration,
        bool isEverlasting
    ) external;

    // Participation Functions
    function joinContest() external payable;

    // Review Functions
    function commitReview(uint256 submissionId, bytes32 commitHash) external;
    function revealReview(uint256 submissionId, uint256 score, uint256 nonce) external;

    // Review Assignment Functions
    function generateReviewAssignments() external;
    function getReviewAssignmentsForReviewer(address reviewer) external view returns (uint256[] memory);
    function isReviewerAssigned(address reviewer, uint256 submissionId) external view returns (bool);

    // Contest Finalization
    function finalizeContest() external;

    // Reward Management
    function claimReward() external;

    // Admin Functions
    function updateMetadata(string calldata newMetadataURI) external;

    // View Functions
    function getContestInfo() external view returns (ContestInfo memory);
    function getParticipant(address account) external view returns (Participant memory);
    function getParticipants() external view returns (address[] memory);
    function getCurrentEpoch() external view returns (uint256);
    function getClaimableRewards(address participant) external view returns (uint256);
    function isParticipant(address account) external view returns (bool);
    function isActive() external view returns (bool);
    function getRewardSplit() external view returns (RewardSplit memory);

    // Events
    event ContestInitialized(address indexed creator, string metadataURI, bool isEverlasting);
    event ParticipantJoined(address indexed participant, uint256 timestamp);
    event SubmissionSubmitted(address indexed participant, uint256 indexed submissionId, string ipfsURI, bytes32 outputHash);
    event ReviewSubmitted(address indexed reviewer, uint256 indexed submissionId, uint256 score);
    event ReviewCommitted(address indexed reviewer, uint256 indexed submissionId, bytes32 commitHash);
    event ReviewRevealed(address indexed reviewer, uint256 indexed submissionId, uint256 score);
    event ReviewAssignmentsGenerated(uint256 totalAssignments);
    event ContestFinalized(address indexed winner, uint256 rewardAmount);
    event RewardClaimed(address indexed participant, uint256 amount);
}
