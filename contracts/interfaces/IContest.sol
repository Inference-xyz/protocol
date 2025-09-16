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
    function submitEntry(bytes32 outputHash, string calldata ipfsURI) external;

    // Review Functions
    function submitReview(uint256 submissionId, uint256 score) external;

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
    event ContestFinalized(address indexed winner, uint256 rewardAmount);
    event RewardClaimed(address indexed participant, uint256 amount);
}
