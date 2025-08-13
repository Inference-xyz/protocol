// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IRewardDistributor {
    struct ContestRewardData {
        uint256 totalRewardsDistributed;
        uint256 lastDistributionTime;
        uint256 emissionPercentage;
        bool isActive;
    }

    function executeDistribution() external;
    function registerContest() external;
    function unregisterContest() external;
    function updateContestEmissionPercentage(address contest) external;

    function getNextDistributionTime() external view returns (uint256);
    function getRegisteredContests() external view returns (address[] memory);
    function getContestRewardData(address contest) external view returns (
        uint256 totalRewardsDistributed,
        uint256 lastDistributionTime,
        uint256 emissionPercentage,
        bool isActive
    );
    function isContestRegistered(address contest) external view returns (bool);

    // Events
    event ContestRegistered(address indexed contest, uint256 emissionPercentage);
    event ContestUnregistered(address indexed contest);
    event DistributionExecuted(uint256 indexed roundId, uint256 totalRewards);
    event ContestRewarded(address indexed contest, uint256 amount, uint256 roundId);
    event DistributionIntervalUpdated(uint256 newInterval);
}
