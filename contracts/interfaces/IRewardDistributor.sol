// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IRewardDistributor {
    struct EmissionConfig {
        uint256 emissionRate; // Tokens per epoch
        uint256 lastEmissionBlock; // Last emission block number
        uint256 minEmissionInterval; // Minimum blocks between emissions
        uint256 epochDuration; // Duration between emissions
        uint256 totalEmitted; // Total tokens emitted
    }

    struct ContestWeight {
        uint256 weight;
        bool active;
        uint256 lastUpdateBlock;
    }

    // Core Functions (Modified)
    function setContestWeight(address contest, uint256 weight) external;
    function setContestActive(address contest, bool active) external;
    function checkpointEmissions() external returns (uint256);
    function distributeRewards() external; // Tamper-proof, math-based

    function getEmissionAmount() external view returns (uint256);
    function canEmit() external view returns (bool);
    function getContestEmissionShare(address contest) external view returns (uint256);

    // DAO Integration
    function setEmissionSplit(uint256[] calldata contestWeights) external; // DAO only
    function getActiveContests() external view returns (address[] memory);

    // Configuration Functions
    function setEmissionRate(uint256 rate) external;
    function setEpochDuration(uint256 duration) external;
    function setMinEmissionInterval(uint256 interval) external;
    function setInferenceToken(address token) external;

    // View Functions
    function getEmissionConfig() external view returns (EmissionConfig memory);
    function getContestWeight(address contest) external view returns (uint256);
    function isContestActive(address contest) external view returns (bool);
    function getTotalContestWeights() external view returns (uint256);
    function timeUntilNextEmission() external view returns (uint256);

    // Events
    event EmissionDistributed(uint256 indexed epoch, uint256 totalAmount, address[] contests, uint256[] amounts);
    event ContestWeightUpdated(address indexed contest, uint256 newWeight);
    event ContestActiveUpdated(address indexed contest, bool active);
    event EmissionRateUpdated(uint256 newRate);
    event EpochDurationUpdated(uint256 newDuration);
    event MinEmissionIntervalUpdated(uint256 newInterval);
    event EmissionSplitUpdated(uint256[] contestWeights);
}
