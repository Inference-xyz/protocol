// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../contracts/ContestRegistry.sol";
import "../contracts/interfaces/IContestRegistry.sol";

contract ContestRegistryTest is Test {
    ContestRegistry public registry;
    
    address public owner = address(0x1);
    address public contest1 = address(0x2);
    address public contest2 = address(0x3);
    address public contest3 = address(0x4);
    address public contest4 = address(0x5);
    
    uint256 constant MAX_SLOTS = 3;
    
    event ContestRegistered(address indexed contest, uint256 timestamp);
    event ContestUnregistered(address indexed contest, uint256 timestamp);
    event MaxSlotsUpdated(uint256 newMaxSlots);
    
    function setUp() public {
        vm.prank(owner);
        registry = new ContestRegistry(MAX_SLOTS);
    }
    
    function testConstructor() public {
        assertEq(registry.maxSlots(), MAX_SLOTS);
        assertEq(registry.owner(), owner);
        assertEq(registry.getCurrentSlotCount(), 0);
    }
    
    function testConstructorRevert() public {
        vm.expectRevert("Max slots must be greater than 0");
        new ContestRegistry(0);
    }
    
    function testRegisterContest() public {
        vm.expectEmit(true, false, false, false);
        emit ContestRegistered(contest1, block.timestamp);
        
        registry.registerContest(contest1);
        
        assertTrue(registry.isRegistered(contest1));
        assertEq(registry.getCurrentSlotCount(), 1);
        
        IContestRegistry.ContestSlot[] memory slots = registry.getContestSlots();
        assertEq(slots.length, 1);
        assertEq(slots[0].contest, contest1);
        assertTrue(slots[0].active);
        assertTrue(slots[0].registeredAt > 0);
        
        address[] memory activeContests = registry.getActiveContests();
        assertEq(activeContests.length, 1);
        assertEq(activeContests[0], contest1);
    }
    
    function testRegisterContestRevert() public {
        // Test invalid address
        vm.expectRevert("Invalid contest address");
        registry.registerContest(address(0));
        
        // Test double registration
        registry.registerContest(contest1);
        vm.expectRevert("Contest already registered");
        registry.registerContest(contest1);
        
        // Test registry full
        registry.registerContest(contest2);
        registry.registerContest(contest3);
        
        vm.expectRevert("Registry is full");
        registry.registerContest(contest4);
    }
    
    function testRegisterMultipleContests() public {
        registry.registerContest(contest1);
        registry.registerContest(contest2);
        registry.registerContest(contest3);
        
        assertEq(registry.getCurrentSlotCount(), 3);
        
        assertTrue(registry.isRegistered(contest1));
        assertTrue(registry.isRegistered(contest2));
        assertTrue(registry.isRegistered(contest3));
        assertFalse(registry.isRegistered(contest4));
        
        address[] memory activeContests = registry.getActiveContests();
        assertEq(activeContests.length, 3);
        assertEq(activeContests[0], contest1);
        assertEq(activeContests[1], contest2);
        assertEq(activeContests[2], contest3);
        
        IContestRegistry.ContestSlot[] memory slots = registry.getContestSlots();
        assertEq(slots.length, 3);
        for (uint256 i = 0; i < 3; i++) {
            assertTrue(slots[i].active);
            assertTrue(slots[i].registeredAt > 0);
        }
    }
    
    function testUnregisterContest() public {
        registry.registerContest(contest1);
        registry.registerContest(contest2);
        
        assertEq(registry.getCurrentSlotCount(), 2);
        
        vm.expectEmit(true, false, false, false);
        emit ContestUnregistered(contest1, block.timestamp);
        
        registry.unregisterContest(contest1);
        
        assertFalse(registry.isRegistered(contest1));
        assertTrue(registry.isRegistered(contest2));
        assertEq(registry.getCurrentSlotCount(), 1);
        
        address[] memory activeContests = registry.getActiveContests();
        assertEq(activeContests.length, 1);
        assertEq(activeContests[0], contest2);
        
        // Slots array should still contain the contest but marked inactive
        IContestRegistry.ContestSlot[] memory slots = registry.getContestSlots();
        assertEq(slots.length, 2);
        assertFalse(slots[0].active); // contest1 slot
        assertTrue(slots[1].active);  // contest2 slot
    }
    
    function testUnregisterContestRevert() public {
        vm.expectRevert("Contest not registered");
        registry.unregisterContest(contest1);
        
        registry.registerContest(contest1);
        registry.unregisterContest(contest1);
        
        // Test unregistering already unregistered contest
        vm.expectRevert("Contest not registered");
        registry.unregisterContest(contest1);
    }
    
    function testUnregisterAndRegisterAgain() public {
        // Fill registry
        registry.registerContest(contest1);
        registry.registerContest(contest2);
        registry.registerContest(contest3);
        
        // Should be full
        vm.expectRevert("Registry is full");
        registry.registerContest(contest4);
        
        // Unregister one
        registry.unregisterContest(contest1);
        assertEq(registry.getCurrentSlotCount(), 2);
        
        // Should be able to register new contest now
        registry.registerContest(contest4);
        assertEq(registry.getCurrentSlotCount(), 3);
        assertTrue(registry.isRegistered(contest4));
        
        address[] memory activeContests = registry.getActiveContests();
        assertEq(activeContests.length, 3);
        // Order: contest2, contest3, contest4
        assertEq(activeContests[0], contest2);
        assertEq(activeContests[1], contest3);
        assertEq(activeContests[2], contest4);
    }
    
    function testSetMaxSlots() public {
        uint256 newMaxSlots = 5;
        
        vm.expectEmit(false, false, false, true);
        emit MaxSlotsUpdated(newMaxSlots);
        
        vm.prank(owner);
        registry.setMaxSlots(newMaxSlots);
        
        assertEq(registry.getMaxSlots(), newMaxSlots);
        
        // Should now be able to register more contests
        registry.registerContest(contest1);
        registry.registerContest(contest2);
        registry.registerContest(contest3);
        registry.registerContest(contest4); // This should work now
        
        assertEq(registry.getCurrentSlotCount(), 4);
    }
    
    function testSetMaxSlotsRevert() public {
        // Test non-owner setting max slots
        vm.expectRevert("Ownable: caller is not the owner");
        vm.prank(contest1);
        registry.setMaxSlots(5);
        
        // Test setting invalid max slots
        vm.expectRevert("Max slots must be greater than 0");
        vm.prank(owner);
        registry.setMaxSlots(0);
    }
    
    function testSetMaxSlotsLowerThanCurrent() public {
        // Register 3 contests
        registry.registerContest(contest1);
        registry.registerContest(contest2);
        registry.registerContest(contest3);
        
        assertEq(registry.getCurrentSlotCount(), 3);
        
        // Set max slots to 2 (lower than current count)
        vm.prank(owner);
        registry.setMaxSlots(2);
        
        assertEq(registry.getMaxSlots(), 2);
        
        // Should not be able to register new contests
        vm.expectRevert("Registry is full");
        registry.registerContest(contest4);
    }
    
    function testGetContestSlots() public {
        // Test empty registry
        IContestRegistry.ContestSlot[] memory emptySlots = registry.getContestSlots();
        assertEq(emptySlots.length, 0);
        
        // Register contests
        registry.registerContest(contest1);
        vm.warp(block.timestamp + 100);
        registry.registerContest(contest2);
        
        IContestRegistry.ContestSlot[] memory slots = registry.getContestSlots();
        assertEq(slots.length, 2);
        
        assertEq(slots[0].contest, contest1);
        assertTrue(slots[0].active);
        assertTrue(slots[0].registeredAt > 0);
        
        assertEq(slots[1].contest, contest2);
        assertTrue(slots[1].active);
        assertTrue(slots[1].registeredAt > slots[0].registeredAt);
    }
    
    function testGetActiveContests() public {
        // Test empty registry
        address[] memory emptyContests = registry.getActiveContests();
        assertEq(emptyContests.length, 0);
        
        // Register contests
        registry.registerContest(contest1);
        registry.registerContest(contest2);
        registry.registerContest(contest3);
        
        address[] memory activeContests = registry.getActiveContests();
        assertEq(activeContests.length, 3);
        assertEq(activeContests[0], contest1);
        assertEq(activeContests[1], contest2);
        assertEq(activeContests[2], contest3);
        
        // Unregister one
        registry.unregisterContest(contest2);
        
        activeContests = registry.getActiveContests();
        assertEq(activeContests.length, 2);
        assertEq(activeContests[0], contest1);
        assertEq(activeContests[1], contest3);
    }
    
    function testIsRegistered() public {
        assertFalse(registry.isRegistered(contest1));
        assertFalse(registry.isRegistered(contest2));
        
        registry.registerContest(contest1);
        assertTrue(registry.isRegistered(contest1));
        assertFalse(registry.isRegistered(contest2));
        
        registry.registerContest(contest2);
        assertTrue(registry.isRegistered(contest1));
        assertTrue(registry.isRegistered(contest2));
        
        registry.unregisterContest(contest1);
        assertFalse(registry.isRegistered(contest1));
        assertTrue(registry.isRegistered(contest2));
    }
    
    function testGetCurrentSlotCount() public {
        assertEq(registry.getCurrentSlotCount(), 0);
        
        registry.registerContest(contest1);
        assertEq(registry.getCurrentSlotCount(), 1);
        
        registry.registerContest(contest2);
        assertEq(registry.getCurrentSlotCount(), 2);
        
        registry.unregisterContest(contest1);
        assertEq(registry.getCurrentSlotCount(), 1);
        
        registry.unregisterContest(contest2);
        assertEq(registry.getCurrentSlotCount(), 0);
    }
    
    function testComplexScenario() public {
        // Register multiple contests
        registry.registerContest(contest1);
        registry.registerContest(contest2);
        registry.registerContest(contest3);
        
        // Registry should be full
        assertEq(registry.getCurrentSlotCount(), 3);
        assertEq(registry.getMaxSlots(), 3);
        
        // Unregister contest2
        registry.unregisterContest(contest2);
        assertEq(registry.getCurrentSlotCount(), 2);
        
        // Register contest4
        registry.registerContest(contest4);
        assertEq(registry.getCurrentSlotCount(), 3);
        
        // Check active contests
        address[] memory activeContests = registry.getActiveContests();
        assertEq(activeContests.length, 3);
        assertEq(activeContests[0], contest1);
        assertEq(activeContests[1], contest3);
        assertEq(activeContests[2], contest4);
        
        // Check all slots (including inactive)
        IContestRegistry.ContestSlot[] memory allSlots = registry.getContestSlots();
        assertEq(allSlots.length, 4); // contest1, contest2 (inactive), contest3, contest4
        assertTrue(allSlots[0].active);  // contest1
        assertFalse(allSlots[1].active); // contest2 (unregistered)
        assertTrue(allSlots[2].active);  // contest3
        assertTrue(allSlots[3].active);  // contest4
        
        // Increase max slots
        vm.prank(owner);
        registry.setMaxSlots(5);
        
        // Should be able to register more contests
        address contest5 = address(0x6);
        registry.registerContest(contest5);
        assertEq(registry.getCurrentSlotCount(), 4);
    }
} 