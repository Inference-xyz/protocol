// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./interfaces/IContestRegistry.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract ContestRegistry is IContestRegistry, Ownable {
    ContestSlot[] public contestSlots;
    mapping(address => uint256) public contestToSlotIndex;
    mapping(address => bool) public registeredContests;
    
    uint256 public maxSlots;
    
    constructor(uint256 _maxSlots) Ownable(msg.sender) {
        require(_maxSlots > 0, "Max slots must be greater than 0");
        maxSlots = _maxSlots;
    }
    
    function registerContest(address contest) external override {
        require(contest != address(0), "Invalid contest address");
        require(!registeredContests[contest], "Contest already registered");
        require(contestSlots.length < maxSlots, "Registry is full");
        
        ContestSlot memory newSlot = ContestSlot({
            contest: contest,
            registeredAt: block.timestamp,
            active: true
        });
        
        contestSlots.push(newSlot);
        contestToSlotIndex[contest] = contestSlots.length - 1;
        registeredContests[contest] = true;
        
        emit ContestRegistered(contest, block.timestamp);
    }
    
    function unregisterContest(address contest) external override {
        require(registeredContests[contest], "Contest not registered");
        
        uint256 slotIndex = contestToSlotIndex[contest];
        require(slotIndex < contestSlots.length, "Invalid slot index");
        
        // Mark as inactive instead of removing to preserve indices
        contestSlots[slotIndex].active = false;
        registeredContests[contest] = false;
        
        emit ContestUnregistered(contest, block.timestamp);
    }
    
    // View Functions
    function getContestSlots() external view override returns (ContestSlot[] memory) {
        return contestSlots;
    }
    
    function getActiveContests() external view override returns (address[] memory) {
        uint256 activeCount = 0;
        
        // First pass: count active contests
        for (uint256 i = 0; i < contestSlots.length; i++) {
            if (contestSlots[i].active) {
                activeCount++;
            }
        }
        
        // Second pass: collect active contests
        address[] memory activeContests = new address[](activeCount);
        uint256 currentIndex = 0;
        
        for (uint256 i = 0; i < contestSlots.length; i++) {
            if (contestSlots[i].active) {
                activeContests[currentIndex] = contestSlots[i].contest;
                currentIndex++;
            }
        }
        
        return activeContests;
    }
    
    function isRegistered(address contest) external view override returns (bool) {
        return registeredContests[contest];
    }
    
    function getMaxSlots() external view override returns (uint256) {
        return maxSlots;
    }
    
    function getCurrentSlotCount() external view override returns (uint256) {
        uint256 activeCount = 0;
        for (uint256 i = 0; i < contestSlots.length; i++) {
            if (contestSlots[i].active) {
                activeCount++;
            }
        }
        return activeCount;
    }
    
    // Configuration Functions
    function setMaxSlots(uint256 _maxSlots) external override onlyOwner {
        require(_maxSlots > 0, "Max slots must be greater than 0");
        maxSlots = _maxSlots;
        emit MaxSlotsUpdated(_maxSlots);
    }
} 