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
        uint256 duration; // 0 for everlasting contests
        uint256 currentEpoch;
        ContestStatus status;
        bool isEverlasting;
        address validatorSet;
        address scoringVerifier;
    }

    struct RewardSplit {
        uint256 ownerPct; // Contest owner percentage
        uint256 participantPct; // Participant rewards percentage
        uint256 validatorPct; // Validator set percentage
        uint256 totalPct; // Must equal 10000
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

    struct ZKSubmission {
        bytes32[] inputs;
        bytes32[] outputs;
        bytes proof;
        bytes32 modelHash;
        uint256 timestamp;
        bool verified;
    }

    struct ScoringSubmission {
        address[] participants;
        uint256[] scores;
        bytes proof;
        bytes32 scoringModelHash;
        uint256 epoch;
        bool verified;
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

    function submitZKProof(bytes32[] calldata inputs, bytes32[] calldata outputs, bytes calldata proof) external;
    function submitScores(address[] calldata participants, uint256[] calldata scores, bytes calldata proof) external;

    // Epoch Management (for everlasting contests)
    function finalizeEpoch() external; // Now uses ZK proofs and validator weights
    function startNewEpoch() external; // Start new epoch for everlasting contests

    // Contest Finalization (for temporary contests)
    function finalizeContest(address[] calldata winners, uint256[] calldata rewards, bytes32 resultHash) external;

    // Reward Management
    function claimReward() external;
    function claimCreatorFee() external;
    function distributeEpochRewards(uint256 amount) external;

    function setRewardSplit(uint256 ownerPct, uint256 participantPct, uint256 validatorPct) external;
    function setValidatorSet(address validatorSet) external;
    function setScoringVerifier(address verifier) external;

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

    function getWeightedRewards() external view returns (uint256[] memory);
    function getZKSubmission(address participant, uint256 epoch) external view returns (ZKSubmission memory);
    function getScoringSubmission(uint256 epoch) external view returns (ScoringSubmission memory);
    function getRewardSplit() external view returns (RewardSplit memory);

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

    event ZKProofSubmitted(address indexed participant, uint256 indexed epoch, bytes32[] inputs, bytes32[] outputs);
    event ScoresSubmitted(uint256 indexed epoch, address[] participants, uint256[] scores);
    event RewardSplitUpdated(uint256 ownerPct, uint256 participantPct, uint256 validatorPct);
    event ValidatorSetUpdated(address indexed validatorSet);
    event ScoringVerifierUpdated(address indexed verifier);
}
