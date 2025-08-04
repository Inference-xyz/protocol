// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IContestFactory {
    struct ContestConfig {
        string metadataURI;
        uint256 duration;
        uint256 epochDuration;
        bytes32 scoringModelHash;
        address[] validators;
        uint256 rewardAmount; // Pre-funded reward amount
    }

    function createContest(ContestConfig calldata config) external returns (address contestAddress);
    function setContestTemplate(address template) external;
    function setInfToken(address _infToken) external;
    function setModelRegistry(address _modelRegistry) external;
    function setVerifierRegistry(address _verifierRegistry) external;

    // View Functions
    function getCreatedContests() external view returns (address[] memory);
    function getContestsByCreator(address creator) external view returns (address[] memory);
    function isValidContest(address contest) external view returns (bool);
    function getContestCount() external view returns (uint256);

    // Events
    event ContestCreated(
        address indexed contestAddress,
        address indexed creator,
        string metadataURI,
        uint256 duration,
        uint256 epochDuration,
        bytes32 scoringModelHash,
        address[] validators,
        uint256 rewardAmount
    );
    event ContestTemplateUpdated(address indexed template);
}
