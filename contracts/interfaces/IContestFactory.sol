// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IContestFactory {
    struct ContestConfig {
        string metadataURI;
        uint256 duration; // Contest duration in seconds (0 for everlasting)
        bytes32 scoringModelHash; // Hash of the scoring model for ZK verification
        address[] validators; // Array of validator addresses
    }

    // Core Functions
    function createContest(ContestConfig calldata config) external returns (address contestAddress);

    // Management Functions
    function setContestTemplate(address template) external;
    function setInfToken(address _infToken) external;

    // View Functions
    function getCreatedContests() external view returns (address[] memory);
    function getContestsByCreator(address creator) external view returns (address[] memory);
    function isValidContest(address contest) external view returns (bool);
    function getContestTemplate() external view returns (address);
    function getTotalContestsCreated() external view returns (uint256);
    function getInfToken() external view returns (address);

    // Events
    event ContestCreated(
        address indexed contestAddress, 
        address indexed creator, 
        string metadataURI, 
        uint256 duration,
        bytes32 scoringModelHash,
        address[] validators
    );
    event ContestTemplateUpdated(address indexed newTemplate);
}
