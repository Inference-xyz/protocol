// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./interfaces/IContestFactory.sol";
import "./interfaces/IContest.sol";
import "./InfToken.sol";
import "@openzeppelin/contracts/proxy/Clones.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract ContestFactory is IContestFactory, Ownable {
    address public contestTemplate;
    address public infToken;
    address[] public createdContests;
    mapping(address => address[]) public contestsByCreator;
    mapping(address => bool) public validContests;
    
    constructor(address _contestTemplate, address _infToken) Ownable(msg.sender) {
        require(_contestTemplate != address(0), "Invalid template");
        require(_infToken != address(0), "Invalid InfToken");
        contestTemplate = _contestTemplate;
        infToken = _infToken;
    }
    
    function createContest(ContestConfig calldata config) external override returns (address contestAddress) {
        require(bytes(config.metadataURI).length > 0, "Empty metadata URI");
        require(config.scoringModelHash != bytes32(0), "Invalid scoring model hash");
        require(config.validators.length > 0, "Must have at least one validator");
        
        // Clone the contest template
        contestAddress = Clones.clone(contestTemplate);
        
        // Initialize the cloned contest
        IContest(contestAddress).initialize(
            msg.sender,
            config.metadataURI,
            config.duration,
            config.scoringModelHash,
            config.validators
        );
        
        // Set InfToken for the contest
        IContest(contestAddress).setInfToken(infToken);
        
        // Track the created contest
        createdContests.push(contestAddress);
        contestsByCreator[msg.sender].push(contestAddress);
        validContests[contestAddress] = true;
        
        emit ContestCreated(
            contestAddress, 
            msg.sender, 
            config.metadataURI, 
            config.duration,
            config.scoringModelHash,
            config.validators
        );
    }
    
    function setContestTemplate(address template) external override onlyOwner {
        require(template != address(0), "Invalid template");
        contestTemplate = template;
        emit ContestTemplateUpdated(template);
    }
    
    function setInfToken(address _infToken) external onlyOwner {
        require(_infToken != address(0), "Invalid InfToken");
        infToken = _infToken;
    }
    
    // View Functions
    function getCreatedContests() external view override returns (address[] memory) {
        return createdContests;
    }
    
    function getContestsByCreator(address creator) external view override returns (address[] memory) {
        return contestsByCreator[creator];
    }
    
    function isValidContest(address contest) external view override returns (bool) {
        return validContests[contest];
    }
    
    function getContestTemplate() external view override returns (address) {
        return contestTemplate;
    }
    
    function getTotalContestsCreated() external view override returns (uint256) {
        return createdContests.length;
    }
    
    function getInfToken() external view returns (address) {
        return infToken;
    }
} 