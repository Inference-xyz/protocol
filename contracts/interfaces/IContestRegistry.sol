// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IContestRegistry {
    struct ContestSlot {
        address contest;
        uint256 registeredAt;
        bool active;
    }

    // Core Registry Functions
    function registerContest(address contest) external;
    function unregisterContest(address contest) external;

    // View Functions
    function getContestSlots() external view returns (ContestSlot[] memory);
    function getActiveContests() external view returns (address[] memory);
    function isRegistered(address contest) external view returns (bool);
    function getMaxSlots() external view returns (uint256);
    function getCurrentSlotCount() external view returns (uint256);

    // Configuration Functions
    function setMaxSlots(uint256 maxSlots) external;

    // Events
    event ContestRegistered(address indexed contest, uint256 timestamp);
    event ContestUnregistered(address indexed contest, uint256 timestamp);
    event MaxSlotsUpdated(uint256 newMaxSlots);
}
