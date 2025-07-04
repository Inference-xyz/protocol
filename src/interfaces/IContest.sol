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
        uint256 creatorFeePct;
        uint256 startTime;
        uint256 duration;           // 0 for everlasting contests
        uint256 currentEpoch;
        ContestStatus status;
        uint256 totalRewardPool;
        uint256 participantCount;
        bool isEverlasting;
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
    
    struct EpochResult {
        uint256 epoch;
        address[] winners;
        uint256[] rewards;
        uint256 totalDistributed;
        uint256 timestamp;
        bytes32 resultHash;
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
    function leaveContest() external;
    function submitEntry(bytes32 submissionHash, string calldata metadataURI) external;
    
    // Epoch Management (for everlasting contests)
    function finalizeEpoch(address[] calldata winners, uint256[] calldata rewards, bytes32 resultHash) external; // Finalize current epoch and distribute rewards
    function startNewEpoch() external;                                                                              // Start new epoch for everlasting contests
    
    // Contest Finalization (for temporary contests)
    function finalizeContest(address[] calldata winners, uint256[] calldata rewards, bytes32 resultHash) external;
    
    // Reward Management
    function claimReward() external;
    function claimCreatorFee() external;
    function distributeEpochRewards(uint256 amount) external;
    
    // Admin Functions
    function pause() external;
    function unpause() external;
    function updateMetadata(string calldata newMetadataURI) external;
    function setMinimumScore(uint256 score) external;
    
    // View Functions
    function getContestInfo() external view returns (ContestInfo memory);
    function getParticipant(address account) external view returns (Participant memory);
    function getEpochResult(uint256 epoch) external view returns (EpochResult memory);
    function getParticipants() external view returns (address[] memory);
    function getCurrentEpoch() external view returns (uint256);
    function getClaimableRewards(address participant) external view returns (uint256);
    function getClaimableCreatorFee() external view returns (uint256);
    function isParticipant(address account) external view returns (bool);
    function getPerformanceRank() external view returns (uint256); // Get contest performance rank (for slot replacement)
    function isActive() external view returns (bool);
    
    // Events
    event ContestInitialized(address indexed creator, string metadataURI, bool isEverlasting);
    event ParticipantJoined(address indexed participant, uint256 timestamp);
    event ParticipantLeft(address indexed participant, uint256 timestamp);
    event EntrySubmitted(address indexed participant, bytes32 submissionHash, string metadataURI);
    event EpochFinalized(uint256 indexed epoch, address[] winners, uint256[] rewards, uint256 totalDistributed);
    event ContestFinalized(address[] winners, uint256[] rewards, uint256 totalDistributed);
    event RewardClaimed(address indexed participant, uint256 amount);
    event CreatorFeeClaimed(address indexed creator, uint256 amount);
    event ContestPaused();
    event ContestUnpaused();
    event EpochRewardsDistributed(uint256 indexed epoch, uint256 amount);
} 