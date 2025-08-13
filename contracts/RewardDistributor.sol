// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "./InfToken.sol";
import "./InferenceDAO.sol";

contract RewardDistributor is Ownable {
    InfToken public immutable infToken;
    InferenceDAO public immutable dao;
    
    uint256 public distributionInterval = 1 days; // Daily distributions
    uint256 public lastDistributionTime;
    mapping(address => bool) public registeredContests;

    event ContestRegistered(address indexed contest);
    event ContestUnregistered(address indexed contest);
    event DistributionExecuted(uint256 totalRewards, address[] contests, uint256[] amounts);

    constructor(InfToken _infToken, InferenceDAO _dao) Ownable(msg.sender) {
        infToken = _infToken;
        dao = _dao;
    }

    function registerContest() external {
        require(!registeredContests[msg.sender], "Contest already registered");
        require(dao.getContestEmissionPercentage(msg.sender) > 0, "Contest not approved by DAO");
        
        registeredContests[msg.sender] = true;
        emit ContestRegistered(msg.sender);
    }

    function unregisterContest() external {
        require(registeredContests[msg.sender], "Contest not registered");
        registeredContests[msg.sender] = false;
        emit ContestUnregistered(msg.sender);
    }

    function executeDistribution() external {
        require(block.timestamp >= lastDistributionTime + distributionInterval, "Distribution not ready");
        
        uint256 availableRewards = infToken.balanceOf(address(this));
        require(availableRewards > 0, "No rewards available");
        
        // Get all registered contests and their emission percentages
        address[] memory contests = new address[](100);
        uint256[] memory percentages = new uint256[](100);
        uint256 contestCount = 0;
        uint256 totalPercentage = 0;
        
        if (registeredContests[msg.sender] && dao.getContestEmissionPercentage(msg.sender) > 0) {
            contests[0] = msg.sender;
            percentages[0] = dao.getContestEmissionPercentage(msg.sender);
            contestCount = 1;
            totalPercentage = percentages[0];
        }
        
        require(totalPercentage > 0, "No active contests with emission");
        
        // Distribute rewards
        uint256[] memory amounts = new uint256[](contestCount);
        for (uint256 i = 0; i < contestCount; i++) {
            uint256 amount = (availableRewards * percentages[i]) / totalPercentage;
            if (amount > 0) {
                infToken.transfer(contests[i], amount);
                amounts[i] = amount;
            }
        }
        
        lastDistributionTime = block.timestamp;
        emit DistributionExecuted(availableRewards, contests, amounts);
    }

    function isContestRegistered(address contest) external view returns (bool) {
        return registeredContests[contest];
    }

    function getNextDistributionTime() external view returns (uint256) {
        return lastDistributionTime + distributionInterval;
    }

    // Admin functions
    function setDistributionInterval(uint256 _interval) external onlyOwner {
        require(_interval > 0, "Interval must be positive");
        distributionInterval = _interval;
    }

    function withdrawTokens(address token, uint256 amount) external onlyOwner {
        require(token != address(infToken), "Cannot withdraw INF tokens");
        IERC20(token).transfer(owner(), amount);
    }
}
