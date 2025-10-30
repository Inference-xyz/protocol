// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title IContest
 * @notice Interface for Contest contract - defines all public functions and events
 */
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
        uint256 startTime;
        uint256 duration;
        ContestStatus status;
    }

    struct Participant {
        address account;
        uint256 joinedAt;
        uint256 totalRewards;
        uint256 lastSubmissionTime;
        bool isActive;
        uint256 stakedAmount;
    }

    struct InitParams {
        uint256 reviewCount;
        uint256 epsilonReward;
        uint256 epsilonSlash;
        uint256 alpha;
        uint256 beta;
        uint256 gamma;
        uint256 minStakeAmount;
        uint256 maxParticipants;
        uint256 joinPriceAdjustment;
    }

    function initialize(
        address creator,
        string calldata metadataURI,
        uint256 duration,
        address contestManager,
        address infToken,
        InitParams calldata params
    ) external;

    function joinContest(uint256 stakeAmount) external;

    function commitReview(uint256 outputId, bytes32 commitHash) external;
    
    function revealReview(uint256 outputId, uint256 score, uint256 nonce) external;
    
    function aggregateReviews(uint256 contestId) external;
    
    function evaluateReviewer(address reviewer) external;

    function generateReviewAssignments() external;
    
    function getReviewAssignmentsForReviewer(address reviewer) external view returns (uint256[] memory);
    
    function isReviewerAssigned(address reviewer, uint256 submissionId) external view returns (bool);

    function finalizeContest() external;

    function claimReward() external;

    function updateMetadata(string calldata newMetadataURI) external;

    function getContestInfo() external view returns (ContestInfo memory);
    
    function getParticipant(address account) external view returns (Participant memory);
    
    function getParticipants() external view returns (address[] memory);
    
    
    function getClaimableRewards(address participant) external view returns (uint256);
    
    function isParticipant(address account) external view returns (bool);
    
    function isActive() external view returns (bool);

    event ContestInitialized(address indexed creator, string metadataURI);
    event ParticipantJoined(address indexed participant, uint256 timestamp);
    event SubmissionSubmitted(address indexed participant, uint256 indexed submissionId, string ipfsURI, bytes32 outputHash);
    event ReviewSubmitted(address indexed reviewer, uint256 indexed submissionId, uint256 score);
    event ReviewCommitted(address indexed reviewer, uint256 indexed submissionId, bytes32 commitHash);
    event ReviewRevealed(address indexed reviewer, uint256 indexed submissionId, uint256 score);
    event ReviewAssignmentsGenerated(uint256 totalAssignments);
    event ContestFinalized(address indexed winner, uint256 rewardAmount);
    event RewardClaimed(address indexed participant, uint256 amount);
}
