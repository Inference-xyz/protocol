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
        uint256 startTime;
        uint256 duration; // 0 for everlasting contests
        ContestStatus status;
        uint256 totalRewards;
        uint256 participantCount;
        bytes32 scoringModelHash;
        address[] validators;
        uint256 currentEpoch;
        uint256 epochDuration;
        address modelRegistry;
        address verifierRegistry;
    }

    struct Participant {
        address account;
        uint256 joinedAt;
        uint256 totalRewards;
        bool isActive;
        uint256 score;
    }

    struct EpochInfo {
        uint256 epochNumber;
        uint256 startTime;
        uint256 endTime;
        bytes32[] inputHashes;
        bool finalized;
        uint256 totalSubmissions;
        uint256 totalScores;
    }

    struct InferenceSubmission {
        address participant;
        uint256 epochNumber;
        bytes32 inputHash;
        bytes32 modelHash;
        bytes32 outputHash;
        bytes zkProof;
        uint256 timestamp;
        bool verified;
    }

    struct ScoringSubmission {
        address scorer;
        uint256 epochNumber;
        bytes32 inputHash;
        address participant;
        bytes32 outputHash;
        uint256[] scores;
        bytes zkProof;
        uint256 timestamp;
        bool verified;
    }

    struct Winner {
        address participant;
        uint256 score;
        uint256 reward;
    }

    // Initialization
    function initialize(
        address creator,
        string calldata metadataURI,
        uint256 duration,
        uint256 epochDuration,
        bytes32 scoringModelHash,
        address modelRegistry,
        address verifierRegistry,
        address[] calldata validators
    ) external;

    // Token Setup
    function setInfToken(address _infToken) external;

    // Validator Management
    function addValidator(address validator) external;
    function removeValidator(address validator) external;

    // Participation Functions
    function joinContest() external;
    function leaveContest() external;

    // Epoch Management
    function postInputHashes(bytes32[] calldata inputHashes) external;
    function startNewEpoch() external;
    function finalizeEpoch(uint256 epochNumber) external;

    // Submission Functions
    function submitInference(
        uint256 epochNumber,
        bytes32 inputHash,
        bytes32 modelHash,
        bytes32 outputHash,
        bytes calldata zkProof,
        uint256[] calldata publicInputs
    ) external;

    function submitScoring(
        uint256 epochNumber,
        bytes32 inputHash,
        address participant,
        bytes32 outputHash,
        uint256[] calldata scores,
        bytes calldata zkProof,
        uint256[] calldata publicInputs
    ) external;

    // Contest Management
    function finalizeContest() external;
    function pause() external;
    function unpause() external;

    // Reward Management
    function distributeRewards(uint256 amount) external;

    // View Functions
    function getContestInfo() external view returns (ContestInfo memory);
    function getParticipant(address account) external view returns (Participant memory);
    function getParticipants() external view returns (address[] memory);
    function getEpochInfo(uint256 epochNumber) external view returns (EpochInfo memory);
    function getCurrentEpoch() external view returns (uint256);
    function getInferenceSubmissions(uint256 epochNumber) external view returns (InferenceSubmission[] memory);
    function getScoringSubmissions(uint256 epochNumber) external view returns (ScoringSubmission[] memory);
    function getWinners() external view returns (Winner[] memory);
    function getClaimableRewards(address participant) external view returns (uint256);
    function isParticipant(address account) external view returns (bool);
    function isActive() external view returns (bool);
    function getValidators() external view returns (address[] memory);
    function isValidator(address validator) external view returns (bool);
    function calculateWinners() external view returns (Winner[] memory);

    // Events
    event ContestInitialized(
        address indexed creator,
        string metadataURI,
        uint256 duration, 
        uint256 epochDuration,
        bytes32 scoringModelHash,
        address modelRegistry,
        address verifierRegistry,
        address[] validators
    );
    event ParticipantJoined(address indexed participant, uint256 timestamp);
    event ParticipantLeft(address indexed participant, uint256 timestamp);
    event EpochStarted(uint256 indexed epochNumber, uint256 startTime, uint256 endTime);
    event InputHashesPosted(uint256 indexed epochNumber, bytes32[] inputHashes);
    event InferenceSubmitted(
        address indexed participant,
        uint256 indexed epochNumber,
        bytes32 inputHash,
        bytes32 modelHash,
        bytes32 outputHash,
        uint256 timestamp
    );
    event ScoringSubmitted(
        address indexed scorer,
        uint256 indexed epochNumber,
        bytes32 inputHash,
        address participant,
        bytes32 outputHash,
        uint256[] scores,
        uint256 timestamp
    );
    event InferenceVerified(address indexed participant, uint256 epochNumber, bool success, address verifier);
    event ScoringVerified(address indexed scorer, uint256 epochNumber, bool success, address verifier);
    event EpochFinalized(uint256 indexed epochNumber, uint256 totalSubmissions, uint256 totalScores);
    event ContestFinalized(Winner[] winners, uint256 totalDistributed);
    event ContestPaused();
    event ContestUnpaused();
    event RewardsDistributed(uint256 amount);
    event ValidatorAdded(address indexed validator);
    event ValidatorRemoved(address indexed validator);
}
