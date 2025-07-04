// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IContestRegistry {
    struct ContestSlot {
        address contest;
        uint256 performanceScore;   // Contest performance score for ranking
        uint256 registeredAt;
        uint256 lastUpdate;
        bool active;
    }
    
    struct DutchAuction {
        address newContest;
        address contestToReplace;
        uint256 startPrice;
        uint256 currentPrice;
        uint256 priceDecayRate;
        uint256 startTime;
        uint256 duration;
        bool active;
    }
    
    // Core Registry Functions
    function registerContest(address contest) external;
    function unregisterContest(address contest) external;
    function updateContestPerformance(address contest, uint256 score) external;
    
    // Slot Management
    function getContestSlots() external view returns (ContestSlot[] memory);
    function getActiveContests() external view returns (address[] memory);
    function isSlotAvailable() external view returns (bool);
    function getLowestPerformingContest() external view returns (address);
    
    // Dutch Auction for Slot Replacement
    function initiateReplacement(address newContest) external returns (uint256 auctionId); // Start auction to replace worst contest
    function participateInAuction(uint256 auctionId) external payable;                      // Bid in replacement auction
    function finalizeAuction(uint256 auctionId) external;                                   // Finalize auction and replace contest
    function cancelAuction(uint256 auctionId) external;                                     // Cancel replacement auction
    
    // Configuration Functions
    function setMaxSlots(uint256 maxSlots) external;
    function setMinPerformanceThreshold(uint256 threshold) external;
    function setAuctionDuration(uint256 duration) external;
    function setPerformanceUpdateInterval(uint256 interval) external;
    
    // View Functions
    function getMaxSlots() external view returns (uint256);
    function getCurrentSlotCount() external view returns (uint256);
    function getContestSlot(address contest) external view returns (ContestSlot memory);
    function getAuction(uint256 auctionId) external view returns (DutchAuction memory);
    function getActiveAuctions() external view returns (uint256[] memory);
    function canReplaceContest(address contest) external view returns (bool);
    function getReplacementCost(address contest) external view returns (uint256);
    
    // Events
    event ContestRegistered(address indexed contest, uint256 slotIndex);
    event ContestUnregistered(address indexed contest, uint256 slotIndex);
    event ContestPerformanceUpdated(address indexed contest, uint256 oldScore, uint256 newScore);
    event ReplacementInitiated(uint256 indexed auctionId, address indexed newContest, address indexed contestToReplace);
    event AuctionParticipation(uint256 indexed auctionId, address indexed participant, uint256 amount);
    event ContestReplaced(address indexed oldContest, address indexed newContest, uint256 finalPrice);
    event AuctionCancelled(uint256 indexed auctionId);
    event MaxSlotsUpdated(uint256 newMaxSlots);
    event PerformanceThresholdUpdated(uint256 newThreshold);
} 