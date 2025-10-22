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
        address qcAddress;
    }

    struct Participant {
        address account;
        uint256 joinedAt;
        uint256 totalRewards;
        uint256 lastSubmissionTime;
        bool isActive;
        uint256 stakedAmount;
    }

    function initialize(
        address creator,
        string calldata metadataURI,
        uint256 duration,
        address contestManager,
        address infToken,
        address qcAddress
    ) external;

    function joinContest(uint256 stakeAmount) external;

    function commitReview(uint256 submissionId, bytes32 commitHash) external;
    
    function revealReview(uint256 submissionId, uint256 score, uint256 nonce) external;

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
