// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IValidatorSet {
    struct Validator {
        address account;
        uint256 stake;
        uint256[] weights; // Weights for each participant
        uint256 lastUpdate;
        bool isActive;
    }

    struct ValidatorConfig {
        uint256 maxValidators;
        uint256 minStake;
        uint256 maxStake;
        uint256 weightPrecision; // For fractional weights
    }

    // Core Functions
    function stake(uint256 amount) external;
    function unstake(uint256 amount) external;
    function setWeights(address[] calldata participants, uint256[] calldata weights) external;
    function getWeightedScores() external view returns (uint256[] memory);

    // Management Functions
    function addValidator(address validator) external;
    function removeValidator(address validator) external;
    function updateValidatorConfig(ValidatorConfig calldata config) external;
    function setContest(address contest) external;

    // View Functions
    function getValidator(address account) external view returns (Validator memory);
    function getValidators() external view returns (address[] memory);
    function getTotalStake() external view returns (uint256);
    function getValidatorCount() external view returns (uint256);
    function getValidatorConfig() external view returns (ValidatorConfig memory);
    function getContest() external view returns (address);
    function isValidator(address account) external view returns (bool);
    function getValidatorStake(address account) external view returns (uint256);
    function getValidatorWeights(address account) external view returns (uint256[] memory);

    // Events
    event ValidatorStaked(address indexed validator, uint256 amount);
    event ValidatorUnstaked(address indexed validator, uint256 amount);
    event WeightsUpdated(address indexed validator, address[] participants, uint256[] weights);
    event ValidatorAdded(address indexed validator);
    event ValidatorRemoved(address indexed validator);
    event ConfigUpdated(ValidatorConfig config);
    event ContestSet(address indexed contest);
}
