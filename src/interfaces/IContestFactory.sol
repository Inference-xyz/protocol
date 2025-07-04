// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IContestFactory {
    struct ContestConfig {
        string metadataURI;
        uint256 creatorFeePct;      // Creator fee percentage (out of 10000)
        uint256 duration;           // Contest duration in seconds (0 for everlasting)
        uint256 minParticipants;    // Minimum participants required
        uint256 maxParticipants;    // Maximum participants allowed (0 for unlimited)
        bool requiresStaking;       // Whether participants need to stake tokens
        uint256 stakingAmount;      // Amount to stake if required
    }
    
    // Core Functions
    function createContest(ContestConfig calldata config) external returns (address contestAddress);
    function createContestWithInitialWeight(ContestConfig calldata config, uint256 initialWeight) external returns (address contestAddress); // Create contest with initial emission weight
    
    // Management Functions
    function pauseContest(address contest) external;
    function unpauseContest(address contest) external;
    function setContestTemplate(address template) external;
    function setRewardDistributor(address distributor) external;
    function setContestRegistry(address registry) external;
    
    // View Functions
    function getCreatedContests() external view returns (address[] memory);
    function getContestsByCreator(address creator) external view returns (address[] memory);
    function isValidContest(address contest) external view returns (bool);
    function getContestTemplate() external view returns (address);
    function getTotalContestsCreated() external view returns (uint256);
    
    // Events
    event ContestCreated(address indexed contestAddress, address indexed creator, string metadataURI, uint256 creatorFeePct);
    event ContestPaused(address indexed contest);
    event ContestUnpaused(address indexed contest);
    event ContestTemplateUpdated(address indexed newTemplate);
    event RewardDistributorUpdated(address indexed newDistributor);
} 