// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./interfaces/IContestFactory.sol";
import "./interfaces/IContest.sol";
import "./InfToken.sol";
import "./ModelRegistry.sol";
import "./ZKVerifierRegistry.sol";
import "@openzeppelin/contracts/proxy/Clones.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract ContestFactory is IContestFactory, Ownable {
    address public contestTemplate;
    address public infToken;
    address public modelRegistry;
    address public verifierRegistry;
    address[] public createdContests;
    mapping(address => address[]) public contestsByCreator;
    mapping(address => bool) public validContests;
    
    constructor(
        address _contestTemplate, 
        address _infToken,
        address _modelRegistry,
        address _verifierRegistry
    ) Ownable(msg.sender) {
        require(_contestTemplate != address(0), "Invalid template");
        require(_infToken != address(0), "Invalid InfToken");
        require(_modelRegistry != address(0), "Invalid model registry");
        require(_verifierRegistry != address(0), "Invalid verifier registry");
        
        contestTemplate = _contestTemplate;
        infToken = _infToken;
        modelRegistry = _modelRegistry;
        verifierRegistry = _verifierRegistry;
    }
    
    function createContest(IContestFactory.ContestConfig calldata config) external override returns (address contestAddress) {
        require(bytes(config.metadataURI).length > 0, "Empty metadata URI");
        require(config.duration > 0 || config.duration == 0, "Invalid duration"); // 0 = everlasting
        require(config.epochDuration > 0, "Epoch duration must be positive");
        require(config.scoringModelHash != bytes32(0), "Invalid scoring model hash");
        
        // Verify scoring model hash exists in verifier registry
        require(
            ZKVerifierRegistry(verifierRegistry).isVerifierActive(config.scoringModelHash),
            "Scoring model not registered or inactive"
        );
        
        // Clone the contest template
        contestAddress = Clones.clone(contestTemplate);
        
        // Initialize the cloned contest
        IContest(contestAddress).initialize(
            msg.sender,
            config.metadataURI,
            config.duration,
            config.epochDuration,
            config.scoringModelHash,
            modelRegistry,
            verifierRegistry
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
            config.epochDuration,
            config.scoringModelHash,

            config.initialRewardAmount
        );
    }
    
    function setContestTemplate(address template) external override onlyOwner {
        require(template != address(0), "Invalid template");
        contestTemplate = template;
        emit ContestTemplateUpdated(template);
    }
    
    function setInfToken(address _infToken) external override onlyOwner {
        require(_infToken != address(0), "Invalid InfToken");
        infToken = _infToken;
    }
    
    function setModelRegistry(address _modelRegistry) external override onlyOwner {
        require(_modelRegistry != address(0), "Invalid model registry");
        modelRegistry = _modelRegistry;
    }
    
    function setVerifierRegistry(address _verifierRegistry) external override onlyOwner {
        require(_verifierRegistry != address(0), "Invalid verifier registry");
        verifierRegistry = _verifierRegistry;
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
    
    function getContestCount() external view override returns (uint256) {
        return createdContests.length;
    }
} 