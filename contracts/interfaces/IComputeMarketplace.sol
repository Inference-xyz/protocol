// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IComputeMarketplace {
    // Events
    event JobPosted(
        uint256 indexed jobId,
        address indexed client,
        bytes32 indexed modelHash,
        bytes32 inputHash1,
        bytes32 inputHash2,
        bytes encryptedInputs,
        uint256 bounty,
        address bountyToken,
        uint256 timeout
    );

    event JobClaimed(uint256 indexed jobId, address indexed provider);

    event JobCompleted(uint256 indexed jobId, address indexed provider, bytes encryptedOutput, bytes zkProof);

    event JobTimedOut(uint256 indexed jobId, address indexed client);

    event ModelHashRegistered(bytes32 indexed modelHash, address indexed registrant);

    // Structs
    struct Job {
        uint256 id;
        address client;
        bytes32 modelHash;
        bytes32 inputHash1;
        bytes32 inputHash2;
        bytes encryptedInputs;
        uint256 bounty;
        address bountyToken;
        uint256 timeout;
        uint256 createdAt;
        uint256 claimedAt;
        address provider;
        bool completed;
        bool timedOut;
    }

    struct ProviderStats {
        uint256 totalJobs;
        uint256 completedJobs;
        uint256 totalEarnings;
        uint256 averageResponseTime;
        uint256 reputationScore;
    }

    // Core Functions
    function postJob(bytes32 modelHash, bytes32 inputHash, uint256 bounty, uint256 timeout) external;

    function claimJob(uint256 jobId) external;

    function completeJob(uint256 jobId, bytes calldata encryptedOutput, bytes calldata zkProof) external;

    function handleJobTimeout(uint256 jobId) external;

    // Model Management
    function registerModelHash(bytes32 modelHash) external;
    function isModelHashRegistered(bytes32 modelHash) external view returns (bool);
    function getModelHashRegistrant(bytes32 modelHash) external view returns (address);

    // View Functions
    function getJob(uint256 jobId) external view returns (Job memory);
    function getJobsByClient(address client) external view returns (uint256[] memory);
    function getJobsByProvider(address provider) external view returns (uint256[] memory);
    function getProviderStats(address provider) external view returns (ProviderStats memory);
    function getAllJobs() external view returns (Job[] memory);
    function getAvailableJobs() external view returns (Job[] memory);

    // Admin Functions
    function setSupportedTokens(address[] calldata tokens) external;
    function isSupportedToken(address token) external view returns (bool);
    function setMinBounty(uint256 newMinBounty) external;
    function getMinBounty() external view returns (uint256);
    function setJobTimeout(uint256 newTimeout) external;
    function getJobTimeout() external view returns (uint256);
    function pause() external;
    function unpause() external;

    // Emergency Functions
    function emergencyWithdraw(address token) external;
}
