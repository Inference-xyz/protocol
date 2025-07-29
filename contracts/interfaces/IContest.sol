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
    }

    struct Participant {
        address account;
        uint256 joinedAt;
        uint256 totalRewards;
        bool isActive;
        uint256 score;
    }

    struct Submission {
        address participant;
        string metadataURI;
        uint256 timestamp;
        bytes32 inputHash;
        bytes32 outputHash;
        bytes zkProof;
        bool verified;
        address validator;
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
        bytes32 scoringModelHash,
        address[] calldata validators
    ) external;

    // Participation Functions
    function joinContest() external;
    function leaveContest() external;
    function submitEntry(
        string calldata metadataURI,
        bytes32 inputHash,
        bytes32 outputHash,
        bytes calldata zkProof,
        uint256[] calldata publicInputs
    ) external;

    // Contest Management
    function finalizeContest() external;
    function pause() external;
    function unpause() external;

    // Reward Management
    function claimReward() external;
    function distributeRewards(uint256 amount) external;

    // Validator Management
    function addValidator(address validator) external;
    function removeValidator(address validator) external;

    // Configuration Functions
    function setInfToken(address _infToken) external;
    function setVerifier(address _verifier) external;

    // View Functions
    function getContestInfo() external view returns (ContestInfo memory);
    function getParticipant(address account) external view returns (Participant memory);
    function getParticipants() external view returns (address[] memory);
    function getSubmissions() external view returns (Submission[] memory);
    function getWinners() external view returns (Winner[] memory);
    function getClaimableRewards(address participant) external view returns (uint256);
    function isParticipant(address account) external view returns (bool);
    function isActive() external view returns (bool);
    function calculateWinners() external view returns (Winner[] memory);
    function getValidators() external view returns (address[] memory);
    function isValidator(address validator) external view returns (bool);

    // Events
    event ContestInitialized(address indexed creator, string metadataURI, uint256 duration, bytes32 scoringModelHash, address[] validators);
    event ParticipantJoined(address indexed participant, uint256 timestamp);
    event ParticipantLeft(address indexed participant, uint256 timestamp);
    event EntrySubmitted(address indexed participant, string metadataURI, bytes32 inputHash, bytes32 outputHash, uint256 timestamp);
    event EntryVerified(address indexed participant, bool isValid, uint256 score, address validator);
    event ContestFinalized(Winner[] winners, uint256 totalDistributed);
    event RewardClaimed(address indexed participant, uint256 amount);
    event RewardsDistributed(uint256 amount);
    event ContestPaused();
    event ContestUnpaused();
    event ValidatorAdded(address indexed validator);
    event ValidatorRemoved(address indexed validator);
}
