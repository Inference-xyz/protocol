// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../contracts/ContestManager.sol";
import "../contracts/Contest.sol";
import "../contracts/InfToken.sol";
import "../contracts/interfaces/IContestManager.sol";
import "../contracts/types/ReviewStructs.sol";

contract ContestManagerTest is Test {
    ContestManager public manager;
    InfToken public infToken;
    
    address public creator = address(0x101);
    address public participant1 = address(0x102);
    address public participant2 = address(0x103);
    address public participant3 = address(0x104);
    
    uint256 public constant STAKE_AMOUNT = 100 ether;
    uint256 public constant REWARD_POOL = 1000 ether;
    
    IContestManager.ContestConfig public testConfig;
    
    function setUp() public {
        infToken = new InfToken();
        manager = new ContestManager(address(infToken));
        
        infToken.transfer(creator, 10000 ether);
        infToken.transfer(participant1, 10000 ether);
        infToken.transfer(participant2, 10000 ether);
        infToken.transfer(participant3, 10000 ether);
        
        testConfig = IContestManager.ContestConfig({
            metadataURI: "https://example.com/contest",
            duration: 14 days,
            stakingAmount: STAKE_AMOUNT,
            qcAddress: address(0)
        });
    }

    function testCreateContest() public {
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        
        assertTrue(manager.isValidContest(contestAddress));
        assertEq(manager.getCreatedContests().length, 1);
        
        address[] memory creatorContests = manager.getContestsByCreator(creator);
        assertEq(creatorContests.length, 1);
        assertEq(creatorContests[0], contestAddress);
    }

    function testStakeViaContest() public {
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        Contest contestContract = Contest(contestAddress);
        
        vm.startPrank(participant1);
        infToken.approve(address(contestContract), STAKE_AMOUNT);
        contestContract.joinContest(STAKE_AMOUNT);
        vm.stopPrank();
        
        IContestManager.StakeInfo memory stakeInfo = manager.getStakeInfo(contestAddress, participant1);
        assertEq(stakeInfo.amount, STAKE_AMOUNT);
        assertFalse(stakeInfo.isSlashed);
    }

    function testUnstakeFromContest() public {
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        Contest contestContract = Contest(contestAddress);
        
        vm.startPrank(participant1);
        infToken.approve(address(contestContract), STAKE_AMOUNT);
        contestContract.joinContest(STAKE_AMOUNT);
        
        // Warp to after contest end + challenge period
        vm.warp(block.timestamp + 14 days + 7 days + 1);
        
        uint256 balanceBefore = infToken.balanceOf(participant1);
        manager.unstakeFromContest(contestAddress);
        uint256 balanceAfter = infToken.balanceOf(participant1);
        
        assertEq(balanceAfter - balanceBefore, STAKE_AMOUNT);
        vm.stopPrank();
    }

    function testCannotUnstakeBeforeContestEnds() public {
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        Contest contestContract = Contest(contestAddress);
        
        vm.startPrank(participant1);
        infToken.approve(address(contestContract), STAKE_AMOUNT);
        contestContract.joinContest(STAKE_AMOUNT);
        
        // Try to unstake immediately
        vm.expectRevert("Stake is still locked");
        manager.unstakeFromContest(contestAddress);
        
        // Try after contest duration but before challenge period ends
        vm.warp(block.timestamp + 14 days + 1);
        vm.expectRevert("Stake is still locked");
        manager.unstakeFromContest(contestAddress);
        vm.stopPrank();
    }
    
    function testCanUnstakeAfterChallengePeriod() public {
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        Contest contestContract = Contest(contestAddress);
        
        vm.startPrank(participant1);
        infToken.approve(address(contestContract), STAKE_AMOUNT);
        contestContract.joinContest(STAKE_AMOUNT);
        
        // Should work exactly after contest + challenge period
        vm.warp(block.timestamp + 14 days + 7 days);
        
        uint256 balanceBefore = infToken.balanceOf(participant1);
        manager.unstakeFromContest(contestAddress);
        uint256 balanceAfter = infToken.balanceOf(participant1);
        
        assertEq(balanceAfter - balanceBefore, STAKE_AMOUNT);
        vm.stopPrank();
    }

    function testSlashParticipant() public {
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        Contest contestContract = Contest(contestAddress);
        
        vm.startPrank(participant1);
        infToken.approve(address(contestContract), STAKE_AMOUNT);
        contestContract.joinContest(STAKE_AMOUNT);
        vm.stopPrank();
        
        uint256 slashAmount = 50 ether;
        
        vm.prank(creator);
        contestContract.slashParticipantForMisbehavior(participant1, slashAmount, "Bad behavior");
        
        IContestManager.StakeInfo memory stakeInfo = manager.getStakeInfo(contestAddress, participant1);
        assertEq(stakeInfo.amount, STAKE_AMOUNT - slashAmount);
        assertTrue(stakeInfo.isSlashed);
    }

    function testFundRewardPool() public {
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        
        vm.startPrank(creator);
        infToken.approve(address(manager), REWARD_POOL);
        manager.fundRewardPool(contestAddress, REWARD_POOL);
        vm.stopPrank();
        
        assertEq(manager.getRewardPool(contestAddress), REWARD_POOL);
    }

    function testDistributeRewards() public {
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        
        vm.startPrank(creator);
        infToken.approve(address(manager), REWARD_POOL);
        manager.fundRewardPool(contestAddress, REWARD_POOL);
        vm.stopPrank();
        
        address[] memory winners = new address[](2);
        winners[0] = participant1;
        winners[1] = participant2;
        
        uint256[] memory amounts = new uint256[](2);
        amounts[0] = 600 ether;
        amounts[1] = 400 ether;
        
        uint256 balance1Before = infToken.balanceOf(participant1);
        uint256 balance2Before = infToken.balanceOf(participant2);
        
        vm.prank(contestAddress);
        manager.distributeRewards(contestAddress, winners, amounts);
        
        assertEq(infToken.balanceOf(participant1) - balance1Before, 600 ether);
        assertEq(infToken.balanceOf(participant2) - balance2Before, 400 ether);
    }

    function testSetContestTemplate() public {
        Contest newTemplate = new Contest();
        address oldTemplate = manager.getContestTemplate();
        
        manager.setContestTemplate(address(newTemplate));
        
        assertEq(manager.getContestTemplate(), address(newTemplate));
        assertTrue(manager.getContestTemplate() != oldTemplate);
    }

    function testOnlyOwnerCanSetTemplate() public {
        Contest newTemplate = new Contest();
        
        vm.prank(participant1);
        vm.expectRevert();
        manager.setContestTemplate(address(newTemplate));
    }

    function testCannotUnstakeFromInvalidContest() public {
        vm.prank(participant1);
        vm.expectRevert("Invalid contest");
        manager.unstakeFromContest(address(0x999));
    }

    function testMultipleContests() public {
        vm.startPrank(creator);
        address contest1 = manager.createContest(testConfig);
        address contest2 = manager.createContest(testConfig);
        address contest3 = manager.createContest(testConfig);
        vm.stopPrank();
        
        assertEq(manager.getCreatedContests().length, 3);
        
        address[] memory creatorContests = manager.getContestsByCreator(creator);
        assertEq(creatorContests.length, 3);
        assertEq(creatorContests[0], contest1);
        assertEq(creatorContests[1], contest2);
        assertEq(creatorContests[2], contest3);
    }

    // Emergency Withdraw Tests
    function testEmergencyWithdrawFromManager() public {
        // Send tokens to manager
        vm.prank(creator);
        infToken.transfer(address(manager), 1000 ether);
        
        uint256 ownerBalanceBefore = infToken.balanceOf(address(this));
        uint256 managerBalance = infToken.balanceOf(address(manager));
        
        manager.emergencyWithdrawFromManager();
        
        uint256 ownerBalanceAfter = infToken.balanceOf(address(this));
        
        assertEq(ownerBalanceAfter - ownerBalanceBefore, managerBalance);
        assertEq(infToken.balanceOf(address(manager)), 0);
    }

    function testCannotEmergencyWithdrawWithZeroBalance() public {
        vm.expectRevert("No balance to withdraw");
        manager.emergencyWithdrawFromManager();
    }

    function testOnlyOwnerCanEmergencyWithdraw() public {
        vm.prank(creator);
        infToken.transfer(address(manager), 1000 ether);
        
        vm.prank(participant1);
        vm.expectRevert();
        manager.emergencyWithdrawFromManager();
    }

    // Reward Distribution Edge Cases
    function testCannotDistributeRewardsToZeroAddress() public {
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        
        vm.startPrank(creator);
        infToken.approve(address(manager), REWARD_POOL);
        manager.fundRewardPool(contestAddress, REWARD_POOL);
        vm.stopPrank();
        
        address[] memory winners = new address[](1);
        winners[0] = address(0);
        uint256[] memory amounts = new uint256[](1);
        amounts[0] = 100 ether;
        
        vm.prank(contestAddress);
        vm.expectRevert("Invalid participant address");
        manager.distributeRewards(contestAddress, winners, amounts);
    }

    function testCannotDistributeRewardsWithEmptyArray() public {
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        
        address[] memory winners = new address[](0);
        uint256[] memory amounts = new uint256[](0);
        
        vm.prank(contestAddress);
        vm.expectRevert("No participants");
        manager.distributeRewards(contestAddress, winners, amounts);
    }

    function testCannotDistributeMoreThanRewardPool() public {
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        
        vm.startPrank(creator);
        infToken.approve(address(manager), REWARD_POOL);
        manager.fundRewardPool(contestAddress, REWARD_POOL);
        vm.stopPrank();
        
        address[] memory winners = new address[](1);
        winners[0] = participant1;
        uint256[] memory amounts = new uint256[](1);
        amounts[0] = REWARD_POOL + 1 ether;
        
        vm.prank(contestAddress);
        vm.expectRevert("Insufficient reward pool");
        manager.distributeRewards(contestAddress, winners, amounts);
    }

    function testCannotDistributeRewardsFromInvalidContest() public {
        address[] memory winners = new address[](1);
        winners[0] = participant1;
        uint256[] memory amounts = new uint256[](1);
        amounts[0] = 100 ether;
        
        vm.prank(address(0x999));
        vm.expectRevert("Invalid contest");
        manager.distributeRewards(address(0x999), winners, amounts);
    }

    function testCannotDistributeRewardsUnauthorized() public {
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        
        vm.startPrank(creator);
        infToken.approve(address(manager), REWARD_POOL);
        manager.fundRewardPool(contestAddress, REWARD_POOL);
        vm.stopPrank();
        
        address[] memory winners = new address[](1);
        winners[0] = participant1;
        uint256[] memory amounts = new uint256[](1);
        amounts[0] = 100 ether;
        
        vm.prank(participant1);
        vm.expectRevert("Not authorized");
        manager.distributeRewards(contestAddress, winners, amounts);
    }

    function testCannotDistributeRewardsWithMismatchedArrays() public {
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        
        address[] memory winners = new address[](2);
        winners[0] = participant1;
        winners[1] = participant2;
        uint256[] memory amounts = new uint256[](1);
        amounts[0] = 100 ether;
        
        vm.prank(contestAddress);
        vm.expectRevert("Arrays length mismatch");
        manager.distributeRewards(contestAddress, winners, amounts);
    }

    // Fund Reward Pool Edge Cases
    function testCannotFundInvalidContest() public {
        vm.startPrank(creator);
        infToken.approve(address(manager), REWARD_POOL);
        vm.expectRevert("Invalid contest");
        manager.fundRewardPool(address(0x999), REWARD_POOL);
        vm.stopPrank();
    }

    function testCannotFundWithZeroAmount() public {
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        
        vm.prank(creator);
        vm.expectRevert("Amount must be greater than 0");
        manager.fundRewardPool(contestAddress, 0);
    }

    // Slashing Edge Cases
    function testCannotSlashWithInsufficientStake() public {
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        Contest contestContract = Contest(contestAddress);
        
        vm.startPrank(participant1);
        infToken.approve(address(contestContract), STAKE_AMOUNT);
        contestContract.joinContest(STAKE_AMOUNT);
        vm.stopPrank();
        
        vm.prank(contestAddress);
        vm.expectRevert("Insufficient stake to slash");
        manager.slashParticipant(contestAddress, participant1, STAKE_AMOUNT + 1 ether, "Over-slashing");
    }

    function testCannotSlashFromInvalidContest() public {
        vm.prank(address(0x999));
        vm.expectRevert("Invalid contest");
        manager.slashParticipant(address(0x999), participant1, 50 ether, "Invalid");
    }

    function testCannotSlashUnauthorized() public {
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        Contest contestContract = Contest(contestAddress);
        
        vm.startPrank(participant1);
        infToken.approve(address(contestContract), STAKE_AMOUNT);
        contestContract.joinContest(STAKE_AMOUNT);
        vm.stopPrank();
        
        vm.prank(participant2);
        vm.expectRevert("Not authorized");
        manager.slashParticipant(contestAddress, participant1, 50 ether, "Unauthorized");
    }

    function testSlashingAddsToRewardPool() public {
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        Contest contestContract = Contest(contestAddress);
        
        vm.startPrank(participant1);
        infToken.approve(address(contestContract), STAKE_AMOUNT);
        contestContract.joinContest(STAKE_AMOUNT);
        vm.stopPrank();
        
        uint256 slashAmount = 50 ether;
        uint256 poolBefore = manager.getRewardPool(contestAddress);
        
        vm.prank(creator);
        contestContract.slashParticipantForMisbehavior(participant1, slashAmount, "Cheating");
        
        uint256 poolAfter = manager.getRewardPool(contestAddress);
        assertEq(poolAfter - poolBefore, slashAmount);
    }

    function testCannotUnstakeSlashedStake() public {
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        Contest contestContract = Contest(contestAddress);
        
        vm.startPrank(participant1);
        infToken.approve(address(contestContract), STAKE_AMOUNT);
        contestContract.joinContest(STAKE_AMOUNT);
        vm.stopPrank();
        
        vm.prank(creator);
        contestContract.slashParticipantForMisbehavior(participant1, 50 ether, "Cheating");
        
        vm.warp(block.timestamp + 14 days + 7 days + 1);
        
        vm.prank(participant1);
        vm.expectRevert("Stake was slashed");
        manager.unstakeFromContest(contestAddress);
    }

    // Record Stake Edge Cases
    function testCannotRecordStakeFromInvalidContest() public {
        vm.prank(address(0x999));
        vm.expectRevert("Only valid contests");
        manager.recordStake(participant1, STAKE_AMOUNT);
    }

    function testCannotRecordZeroStake() public {
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        
        vm.prank(contestAddress);
        vm.expectRevert("Amount must be greater than 0");
        manager.recordStake(participant1, 0);
    }

    function testCannotRecordStakeToZeroAddress() public {
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        
        vm.prank(contestAddress);
        vm.expectRevert("Invalid participant address");
        manager.recordStake(address(0), STAKE_AMOUNT);
    }

    function testCannotRecordStakeForSlashedParticipant() public {
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        Contest contestContract = Contest(contestAddress);
        
        vm.startPrank(participant1);
        infToken.approve(address(contestContract), STAKE_AMOUNT);
        contestContract.joinContest(STAKE_AMOUNT);
        vm.stopPrank();
        
        vm.prank(creator);
        contestContract.slashParticipantForMisbehavior(participant1, 50 ether, "Cheating");
        
        vm.prank(contestAddress);
        vm.expectRevert("Participant is slashed");
        manager.recordStake(participant1, STAKE_AMOUNT);
    }

    // Unstake Edge Cases
    function testCannotUnstakeWithNoStake() public {
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        
        vm.prank(participant1);
        vm.expectRevert("No stake to withdraw");
        manager.unstakeFromContest(contestAddress);
    }

    // Template Management Edge Cases
    function testCannotSetInvalidTemplate() public {
        vm.expectRevert("Invalid template");
        manager.setContestTemplate(address(0));
    }

    function testCannotSetInvalidInfToken() public {
        vm.expectRevert("Invalid token");
        manager.setInfToken(address(0));
    }
}

