// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IRewardDistributor {
    struct EmissionConfig {
        uint256 emissionRate;      // Tokens per epoch
        uint256 computeSplitPct;   // Percentage for compute providers (out of 10000)
        uint256 lastEmissionTime;  // Last emission checkpoint
        uint256 epochDuration;     // Duration between emissions
    }
    
    struct ContestWeight {
        uint256 weight;
        bool active;
    }
    
    // Core Functions
    function setComputeSplit(uint256 percent) external;
    function setContestWeight(address contest, uint256 weight) external;
    function setContestActive(address contest, bool active) external;
    function checkpointEmissions() external;
    function distributeRewards() external;
    
    // Configuration Functions
    function setEmissionRate(uint256 rate) external;
    function setEpochDuration(uint256 duration) external;
    function setInferenceToken(address token) external;
    function setComputeMarketplace(address marketplace) external;
    
    // View Functions
    function getEmissionConfig() external view returns (EmissionConfig memory);
    function getContestWeight(address contest) external view returns (uint256);
    function isContestActive(address contest) external view returns (bool);
    function getTotalContestWeights() external view returns (uint256);
    function getNextEmissionAmount() external view returns (uint256 computeAmount, uint256 contestAmount);
    function timeUntilNextEmission() external view returns (uint256);
    
    // Events
    event EmissionDistributed(uint256 indexed epoch, uint256 computeAmount, uint256 contestAmount);
    event ComputeSplitUpdated(uint256 newPercent);
    event ContestWeightUpdated(address indexed contest, uint256 newWeight);
    event ContestActiveUpdated(address indexed contest, bool active);
    event EmissionRateUpdated(uint256 newRate);
    event EpochDurationUpdated(uint256 newDuration);
} 