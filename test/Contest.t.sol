// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../contracts/Contest.sol";
import "../contracts/ContestManager.sol";
import "../contracts/InfToken.sol";
import "../contracts/types/ReviewStructs.sol";
import "../contracts/interfaces/IContestManager.sol";

contract ContestTest is Test {
    Contest public contest;
    ContestManager public manager;
    InfToken public infToken;
    
    address public creator = address(0x101);
    address public participant1 = address(0x102);
    address public participant2 = address(0x103);
    address public participant3 = address(0x104);
    address public participant4 = address(0x105);
    address public qcAddress = address(0x106);
    
    uint256 public constant STAKE_AMOUNT = 100 ether;
    uint256 public constant CREATOR_FEE_PCT = 1000;
    string public constant METADATA_URI = "https://example.com/contest-metadata";
    
    IContestManager.ContestConfig public testConfig;
    
    uint256 public constant REVIEWER_PRIVATE_KEY = 0xA11CE;
    address public reviewer = vm.addr(REVIEWER_PRIVATE_KEY);
    
    function setUp() public {
        infToken = new InfToken();
        manager = new ContestManager(address(infToken));
        
        testConfig = IContestManager.ContestConfig({
            metadataURI: METADATA_URI,
            duration: 14 days,
            stakingAmount: STAKE_AMOUNT,
            qcAddress: address(0)
        });
        
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        contest = Contest(contestAddress);
        
        infToken.transfer(creator, 10000 ether);
        infToken.transfer(participant1, 10000 ether);
        infToken.transfer(participant2, 10000 ether);
        infToken.transfer(participant3, 10000 ether);
        infToken.transfer(participant4, 10000 ether);
        infToken.transfer(reviewer, 10000 ether);
    }

    function testContestInitialization() public {
        IContest.ContestInfo memory info = contest.getContestInfo();
        assertEq(info.creator, creator);
        assertEq(info.metadataURI, METADATA_URI);
        assertTrue(info.status == IContest.ContestStatus.Active);
    }

    function testJoinContest() public {
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT);
        contest.joinContest(STAKE_AMOUNT);
        vm.stopPrank();
        
        assertTrue(contest.isParticipant(participant1));
        
        IContest.Participant memory p = contest.getParticipant(participant1);
        assertEq(p.account, participant1);
        assertEq(p.stakedAmount, STAKE_AMOUNT);
        assertTrue(p.isActive);
    }

    function testJoinContestWithoutApproval() public {
        vm.prank(participant1);
        vm.expectRevert();
        contest.joinContest(STAKE_AMOUNT);
    }

    function testLeaveContest() public {
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT);
        contest.joinContest(STAKE_AMOUNT);
        
        contest.leaveContest();
        vm.stopPrank();
        
        assertFalse(contest.isParticipant(participant1));
    }

    function testCommitSubmission() public {
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT);
        contest.joinContest(STAKE_AMOUNT);
        
        bytes32 outputHash = keccak256("test-output");
        uint256 nonce = 12345;
        bytes32 commitHash = keccak256(abi.encodePacked("QmTestHash", outputHash, nonce));
        
        contest.commitSubmission(commitHash);
        vm.stopPrank();
        
        ReviewStructs.Submission memory submission = contest.getSubmission(1);
        assertEq(submission.participant, participant1);
        assertEq(submission.commitHash, commitHash);
        assertFalse(submission.isRevealed);
    }

    function testRevealSubmission() public {
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT);
        contest.joinContest(STAKE_AMOUNT);
        
        string memory ipfsURI = "QmTestHash";
        bytes32 outputHash = keccak256("test-output");
        uint256 nonce = 12345;
        bytes32 commitHash = keccak256(abi.encodePacked(ipfsURI, outputHash, nonce));
        
        contest.commitSubmission(commitHash);
        
        IContest.ContestInfo memory info = contest.getContestInfo();
        vm.warp(info.startTime + (info.duration / 4) + 1);
        
        contest.revealSubmission(1, ipfsURI, outputHash, nonce);
        vm.stopPrank();
        
        ReviewStructs.Submission memory submission = contest.getSubmission(1);
        assertTrue(submission.isRevealed);
        assertEq(submission.ipfsURI, ipfsURI);
        assertEq(submission.outputHash, outputHash);
    }

    function testRevealSubmissionInvalidCommit() public {
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT);
        contest.joinContest(STAKE_AMOUNT);
        
        string memory ipfsURI = "QmTestHash";
        bytes32 outputHash = keccak256("test-output");
        uint256 nonce = 12345;
        bytes32 commitHash = keccak256(abi.encodePacked(ipfsURI, outputHash, nonce));
        
        contest.commitSubmission(commitHash);
        
        IContest.ContestInfo memory info = contest.getContestInfo();
        vm.warp(info.startTime + (info.duration / 4) + 1);
        
        vm.expectRevert("Invalid reveal");
        contest.revealSubmission(1, ipfsURI, outputHash, 54321);
        vm.stopPrank();
    }

    function testFullWorkflow() public {
        IContest.ContestInfo memory info = contest.getContestInfo();
        uint256 startTime = info.startTime;
        
        address[] memory participants = new address[](3);
        participants[0] = participant1;
        participants[1] = participant2;
        participants[2] = participant3;
        
        for (uint256 i = 0; i < participants.length; i++) {
            vm.startPrank(participants[i]);
            infToken.approve(address(contest), STAKE_AMOUNT);
            contest.joinContest(STAKE_AMOUNT);
            vm.stopPrank();
        }
        
        bytes32[] memory outputHashes = new bytes32[](3);
        uint256[] memory nonces = new uint256[](3);
        
        for (uint256 i = 0; i < participants.length; i++) {
            string memory ipfsURI = string(abi.encodePacked("QmHash", vm.toString(i)));
            outputHashes[i] = keccak256(abi.encodePacked("output", i));
            nonces[i] = 10000 + i;
            bytes32 commitHash = keccak256(abi.encodePacked(ipfsURI, outputHashes[i], nonces[i]));
            
            vm.prank(participants[i]);
            contest.commitSubmission(commitHash);
        }
        
        vm.warp(startTime + (info.duration / 4) + 1);
        
        for (uint256 i = 0; i < participants.length; i++) {
            string memory ipfsURI = string(abi.encodePacked("QmHash", vm.toString(i)));
            vm.prank(participants[i]);
            contest.revealSubmission(i + 1, ipfsURI, outputHashes[i], nonces[i]);
        }
        
        vm.warp(startTime + (info.duration / 2) + 1);
        
        vm.prank(creator);
        contest.generateReviewAssignments();
        
        vm.warp(startTime + (3 * info.duration / 4) + 1);
        
        vm.startPrank(creator);
        infToken.approve(address(manager), 1000 ether);
        manager.fundRewardPool(address(contest), 1000 ether);
        vm.stopPrank();
        
        vm.prank(creator);
        contest.finalizeContest();
        
        assertTrue(contest.isFinalized());
    }

    function testPhasesProgression() public {
        IContest.ContestInfo memory info = contest.getContestInfo();
        
        uint256 commitDeadline = info.startTime + (info.duration / 4);
        uint256 revealDeadline = info.startTime + (info.duration / 2);
        uint256 reviewDeadline = info.startTime + (3 * info.duration / 4);
        
        assertEq(uint256(contest.getCurrentPhase()), uint256(ReviewStructs.Phase.Commit));
        
        vm.warp(commitDeadline + 1);
        assertEq(uint256(contest.getCurrentPhase()), uint256(ReviewStructs.Phase.Reveal));
        
        vm.warp(revealDeadline + 1);
        assertEq(uint256(contest.getCurrentPhase()), uint256(ReviewStructs.Phase.ReviewCommit));
        
        vm.warp(reviewDeadline + 1);
        assertEq(uint256(contest.getCurrentPhase()), uint256(ReviewStructs.Phase.ReviewReveal));
    }

    function testCannotCommitInWrongPhase() public {
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT);
        contest.joinContest(STAKE_AMOUNT);
        
        vm.warp(block.timestamp + (14 days / 4) + 1);
        
        bytes32 commitHash = keccak256(abi.encodePacked(keccak256("test"), uint256(123)));
        
        vm.expectRevert("Invalid phase");
        contest.commitSubmission(commitHash);
        vm.stopPrank();
    }

    function testCannotRevealInWrongPhase() public {
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT);
        contest.joinContest(STAKE_AMOUNT);
        
        vm.expectRevert("Invalid phase");
        contest.revealSubmission(1, "QmTest", keccak256("test"), 123);
        vm.stopPrank();
    }

    function testDoubleJoinPrevention() public {
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT * 2);
        contest.joinContest(STAKE_AMOUNT);
        
        vm.expectRevert("Already joined");
        contest.joinContest(STAKE_AMOUNT);
        vm.stopPrank();
    }

    function testNonParticipantCannotCommit() public {
        bytes32 commitHash = keccak256(abi.encodePacked(keccak256("test"), uint256(123)));
        
        vm.prank(participant1);
        vm.expectRevert("Not an active participant or QC");
        contest.commitSubmission(commitHash);
    }
 
    function testCannotLeaveAfterCommitPhase() public {
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT);
        contest.joinContest(STAKE_AMOUNT);
        
        vm.warp(block.timestamp + (14 days / 4) + 1);
        
        vm.expectRevert("Cannot leave after commit phase");
        contest.leaveContest();
        vm.stopPrank();
    }

    function testSlashParticipant() public {
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT);
        contest.joinContest(STAKE_AMOUNT);
        vm.stopPrank();
        
        vm.prank(creator);
        contest.slashParticipantForMisbehavior(participant1, 50 ether, "Cheating detected");
        
        IContest.Participant memory p = contest.getParticipant(participant1);
        assertEq(p.stakedAmount, STAKE_AMOUNT - 50 ether);
    }

    function testGenerateReviewAssignments() public {
        address[] memory participants = new address[](3);
        participants[0] = participant1;
        participants[1] = participant2;
        participants[2] = participant3;
        
        for (uint256 i = 0; i < participants.length; i++) {
            vm.startPrank(participants[i]);
            infToken.approve(address(contest), STAKE_AMOUNT);
            contest.joinContest(STAKE_AMOUNT);
            
            string memory ipfsURI = string(abi.encodePacked("QmHash", vm.toString(i)));
            bytes32 outputHash = keccak256(abi.encodePacked("output", i));
            uint256 nonce = 10000 + i;
            bytes32 commitHash = keccak256(abi.encodePacked(ipfsURI, outputHash, nonce));
            
            contest.commitSubmission(commitHash);
            vm.stopPrank();
        }
        
        IContest.ContestInfo memory info = contest.getContestInfo();
        vm.warp(info.startTime + (info.duration / 4) + 1);
        
        for (uint256 i = 0; i < participants.length; i++) {
            string memory ipfsURI = string(abi.encodePacked("QmHash", vm.toString(i)));
            bytes32 outputHash = keccak256(abi.encodePacked("output", i));
            uint256 nonce = 10000 + i;
            
            vm.prank(participants[i]);
            contest.revealSubmission(i + 1, ipfsURI, outputHash, nonce);
        }
        
        vm.prank(creator);
        contest.generateReviewAssignments();
        
        uint256[] memory assignments = contest.getReviewAssignmentsForReviewer(participant1);
        assertTrue(assignments.length > 0);
    }

    // Pause/Unpause Tests
    function testPauseContest() public {
        vm.prank(creator);
        contest.pauseContest();
        
        assertTrue(contest.paused());
        assertEq(uint256(contest.getContestInfo().status), uint256(IContest.ContestStatus.Paused));
    }

    function testUnpauseContest() public {
        vm.startPrank(creator);
        contest.pauseContest();
        contest.unpauseContest();
        vm.stopPrank();
        
        assertFalse(contest.paused());
        assertEq(uint256(contest.getContestInfo().status), uint256(IContest.ContestStatus.Active));
    }

    function testOnlyOwnerCanPause() public {
        vm.prank(participant1);
        vm.expectRevert();
        contest.pauseContest();
    }

    function testOnlyOwnerCanUnpause() public {
        vm.prank(creator);
        contest.pauseContest();
        
        vm.prank(participant1);
        vm.expectRevert();
        contest.unpauseContest();
    }

    function testCannotJoinWhenPaused() public {
        vm.prank(creator);
        contest.pauseContest();
        
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT);
        vm.expectRevert("Contest is paused");
        contest.joinContest(STAKE_AMOUNT);
        vm.stopPrank();
    }

    function testCannotCommitWhenPaused() public {
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT);
        contest.joinContest(STAKE_AMOUNT);
        vm.stopPrank();
        
        vm.prank(creator);
        contest.pauseContest();
        
        bytes32 commitHash = keccak256(abi.encodePacked("test"));
        vm.prank(participant1);
        vm.expectRevert("Contest is paused");
        contest.commitSubmission(commitHash);
    }

    function testCannotRevealWhenPaused() public {
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT);
        contest.joinContest(STAKE_AMOUNT);
        
        string memory ipfsURI = "QmTestHash";
        bytes32 outputHash = keccak256("test-output");
        uint256 nonce = 12345;
        bytes32 commitHash = keccak256(abi.encodePacked(ipfsURI, outputHash, nonce));
        
        contest.commitSubmission(commitHash);
        vm.stopPrank();
        
        IContest.ContestInfo memory info = contest.getContestInfo();
        vm.warp(info.startTime + (info.duration / 4) + 1);
        
        vm.prank(creator);
        contest.pauseContest();
        
        vm.prank(participant1);
        vm.expectRevert("Contest is paused");
        contest.revealSubmission(1, ipfsURI, outputHash, nonce);
    }

    function testCannotLeaveWhenPaused() public {
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT);
        contest.joinContest(STAKE_AMOUNT);
        vm.stopPrank();
        
        vm.prank(creator);
        contest.pauseContest();
        
        vm.prank(participant1);
        vm.expectRevert("Contest is paused");
        contest.leaveContest();
    }

    function testCannotReviewWhenPaused() public {
        // Setup: Create submissions and reach review phase
        address[] memory participants = new address[](2);
        participants[0] = participant1;
        participants[1] = participant2;
        
        for (uint256 i = 0; i < participants.length; i++) {
            vm.startPrank(participants[i]);
            infToken.approve(address(contest), STAKE_AMOUNT);
            contest.joinContest(STAKE_AMOUNT);
            
            string memory ipfsURI = string(abi.encodePacked("QmHash", vm.toString(i)));
            bytes32 outputHash = keccak256(abi.encodePacked("output", i));
            uint256 nonce = 10000 + i;
            bytes32 commitHash = keccak256(abi.encodePacked(ipfsURI, outputHash, nonce));
            
            contest.commitSubmission(commitHash);
            vm.stopPrank();
        }
        
        IContest.ContestInfo memory info = contest.getContestInfo();
        vm.warp(info.startTime + (info.duration / 4) + 1);
        
        for (uint256 i = 0; i < participants.length; i++) {
            string memory ipfsURI = string(abi.encodePacked("QmHash", vm.toString(i)));
            bytes32 outputHash = keccak256(abi.encodePacked("output", i));
            uint256 nonce = 10000 + i;
            
            vm.prank(participants[i]);
            contest.revealSubmission(i + 1, ipfsURI, outputHash, nonce);
        }
        
        vm.warp(info.startTime + (info.duration / 2) + 1);
        vm.prank(creator);
        contest.generateReviewAssignments();
        
        // Pause before review
        vm.prank(creator);
        contest.pauseContest();
        
        // Try to commit review
        uint256[] memory assignments = contest.getReviewAssignmentsForReviewer(participant1);
        require(assignments.length > 0, "Need assignments");
        
        bytes32 reviewCommitHash = keccak256(abi.encodePacked(assignments[0], uint256(80), uint256(999)));
        vm.prank(participant1);
        vm.expectRevert("Contest is paused");
        contest.commitReview(assignments[0], reviewCommitHash);
    }

    function testEmergencyWithdraw() public {
        vm.prank(participant1);
        infToken.transfer(address(contest), STAKE_AMOUNT);
        
        vm.prank(creator);
        contest.pauseContest();
        
        uint256 ownerBalanceBefore = infToken.balanceOf(creator);
        uint256 contractBalance = infToken.balanceOf(address(contest));
        
        vm.prank(creator);
        contest.emergencyWithdraw();
        
        uint256 ownerBalanceAfter = infToken.balanceOf(creator);
        
        assertEq(ownerBalanceAfter - ownerBalanceBefore, contractBalance);
        assertEq(infToken.balanceOf(address(contest)), 0);
    }

    function testCannotEmergencyWithdrawWithoutPause() public {
        vm.prank(creator);
        vm.expectRevert("Contest must be paused first");
        contest.emergencyWithdraw();
    }

    function testEmergencyWithdrawWithNoBalance() public {
        vm.prank(creator);
        contest.pauseContest();
        
        uint256 ownerBalanceBefore = infToken.balanceOf(creator);
        
        vm.prank(creator);
        contest.emergencyWithdraw();
        
        assertEq(infToken.balanceOf(creator), ownerBalanceBefore);
    }

    function testCannotPauseTwice() public {
        vm.startPrank(creator);
        contest.pauseContest();
        vm.expectRevert("Already paused");
        contest.pauseContest();
        vm.stopPrank();
    }

    function testCannotUnpauseWhenNotPaused() public {
        vm.prank(creator);
        vm.expectRevert("Not paused");
        contest.unpauseContest();
    }

    function testCannotPauseFinalized() public {
        // Quick finalize setup
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT);
        contest.joinContest(STAKE_AMOUNT);
        
        string memory ipfsURI = "QmHash";
        bytes32 outputHash = keccak256("output");
        uint256 nonce = 10000;
        bytes32 commitHash = keccak256(abi.encodePacked(ipfsURI, outputHash, nonce));
        contest.commitSubmission(commitHash);
        vm.stopPrank();
        
        IContest.ContestInfo memory info = contest.getContestInfo();
        vm.warp(info.startTime + (info.duration / 4) + 1);
        
        vm.prank(participant1);
        contest.revealSubmission(1, ipfsURI, outputHash, nonce);
        
        vm.startPrank(creator);
        infToken.approve(address(manager), 1000 ether);
        manager.fundRewardPool(address(contest), 1000 ether);
        contest.finalizeContest();
        
        vm.expectRevert("Contest already finalized");
        contest.pauseContest();
        vm.stopPrank();
    }

    // Metadata Tests
    function testMetadataUpdateEvent() public {
        string memory newMetadata = "https://example.com/new-metadata";
        
        vm.prank(creator);
        vm.expectEmit(false, false, false, true);
        emit MetadataUpdated(METADATA_URI, newMetadata);
        contest.updateMetadata(newMetadata);
        
        assertEq(contest.getContestInfo().metadataURI, newMetadata);
    }

    // Leave Contest Edge Cases
    function testParticipantLeftEvent() public {
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT);
        contest.joinContest(STAKE_AMOUNT);
        
        vm.expectEmit(true, false, false, false);
        emit ParticipantLeft(participant1);
        contest.leaveContest();
        vm.stopPrank();
    }

    function testCannotLeaveAfterSubmitting() public {
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT);
        contest.joinContest(STAKE_AMOUNT);
        
        bytes32 commitHash = keccak256(abi.encodePacked("test"));
        contest.commitSubmission(commitHash);
        
        vm.expectRevert("Cannot leave after submitting");
        contest.leaveContest();
        vm.stopPrank();
    }

    function testCannotLeaveWhenNotParticipant() public {
        vm.prank(participant1);
        vm.expectRevert("Not a participant");
        contest.leaveContest();
    }

    function testCannotLeaveWhenFinalized() public {
        // Join and submit
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT);
        contest.joinContest(STAKE_AMOUNT);
        
        string memory ipfsURI = "QmHash";
        bytes32 outputHash = keccak256("output");
        uint256 nonce = 10000;
        bytes32 commitHash = keccak256(abi.encodePacked(ipfsURI, outputHash, nonce));
        contest.commitSubmission(commitHash);
        vm.stopPrank();
        
        IContest.ContestInfo memory info = contest.getContestInfo();
        vm.warp(info.startTime + (info.duration / 4) + 1);
        
        vm.prank(participant1);
        contest.revealSubmission(1, ipfsURI, outputHash, nonce);
        
        // Finalize
        vm.startPrank(creator);
        infToken.approve(address(manager), 1000 ether);
        manager.fundRewardPool(address(contest), 1000 ether);
        contest.finalizeContest();
        vm.stopPrank();
        
        // Try to leave after finalized - should fail before checking phase
        vm.prank(participant1);
        vm.expectRevert("Contest already finalized");
        contest.leaveContest();
    }

    // Slashing Edge Cases
    function testCannotSlashMoreThanStaked() public {
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT);
        contest.joinContest(STAKE_AMOUNT);
        vm.stopPrank();
        
        vm.prank(creator);
        vm.expectRevert("Insufficient stake to slash");
        contest.slashParticipantForMisbehavior(participant1, STAKE_AMOUNT + 1 ether, "Over-slashing");
    }

    function testCannotSlashNonParticipant() public {
        vm.prank(creator);
        vm.expectRevert("Not an active participant");
        contest.slashParticipantForMisbehavior(participant1, 10 ether, "Not a participant");
    }

    function testCannotSlashZeroAmount() public {
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT);
        contest.joinContest(STAKE_AMOUNT);
        vm.stopPrank();
        
        vm.prank(creator);
        vm.expectRevert("Amount must be greater than 0");
        contest.slashParticipantForMisbehavior(participant1, 0, "Zero amount");
    }

    // Review Assignment Edge Cases
    function testCannotGenerateAssignmentsTwice() public {
        address[] memory participants = new address[](2);
        participants[0] = participant1;
        participants[1] = participant2;
        
        for (uint256 i = 0; i < participants.length; i++) {
            vm.startPrank(participants[i]);
            infToken.approve(address(contest), STAKE_AMOUNT);
            contest.joinContest(STAKE_AMOUNT);
            
            string memory ipfsURI = string(abi.encodePacked("QmHash", vm.toString(i)));
            bytes32 outputHash = keccak256(abi.encodePacked("output", i));
            uint256 nonce = 10000 + i;
            bytes32 commitHash = keccak256(abi.encodePacked(ipfsURI, outputHash, nonce));
            
            contest.commitSubmission(commitHash);
            vm.stopPrank();
        }
        
        IContest.ContestInfo memory info = contest.getContestInfo();
        vm.warp(info.startTime + (info.duration / 4) + 1);
        
        for (uint256 i = 0; i < participants.length; i++) {
            string memory ipfsURI = string(abi.encodePacked("QmHash", vm.toString(i)));
            bytes32 outputHash = keccak256(abi.encodePacked("output", i));
            uint256 nonce = 10000 + i;
            
            vm.prank(participants[i]);
            contest.revealSubmission(i + 1, ipfsURI, outputHash, nonce);
        }
        
        vm.startPrank(creator);
        contest.generateReviewAssignments();
        
        vm.expectRevert("Assignments already generated");
        contest.generateReviewAssignments();
        vm.stopPrank();
    }

    function testCannotGenerateAssignmentsWithNoSubmissions() public {
        vm.prank(creator);
        vm.expectRevert("No submissions to review");
        contest.generateReviewAssignments();
    }

    function testCannotGenerateAssignmentsWithOneParticipant() public {
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT);
        contest.joinContest(STAKE_AMOUNT);
        
        bytes32 commitHash = keccak256(abi.encodePacked("test"));
        contest.commitSubmission(commitHash);
        vm.stopPrank();
        
        vm.prank(creator);
        vm.expectRevert("Need at least 2 participants");
        contest.generateReviewAssignments();
    }

    // Claim Reward Edge Cases
    function testCannotClaimRewardWhenNotFinalized() public {
        vm.prank(participant1);
        vm.expectRevert("Contest not finalized");
        contest.claimReward();
    }

    function testCannotClaimRewardWithNoRewards() public {
        // Setup and finalize
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT);
        contest.joinContest(STAKE_AMOUNT);
        
        string memory ipfsURI = "QmHash";
        bytes32 outputHash = keccak256("output");
        uint256 nonce = 10000;
        bytes32 commitHash = keccak256(abi.encodePacked(ipfsURI, outputHash, nonce));
        contest.commitSubmission(commitHash);
        vm.stopPrank();
        
        IContest.ContestInfo memory info = contest.getContestInfo();
        vm.warp(info.startTime + (info.duration / 4) + 1);
        
        vm.prank(participant1);
        contest.revealSubmission(1, ipfsURI, outputHash, nonce);
        
        vm.startPrank(creator);
        infToken.approve(address(manager), 1000 ether);
        manager.fundRewardPool(address(contest), 1000 ether);
        contest.finalizeContest();
        vm.stopPrank();
        
        // Try to claim with participant2 who has no rewards
        vm.prank(participant2);
        vm.expectRevert("No rewards to claim");
        contest.claimReward();
    }

    // Initialize Edge Cases
    function testCannotInitializeTwice() public {
        vm.prank(creator);
        vm.expectRevert("Already initialized");
        contest.initialize(
            creator,
            "new-metadata",
            7 days,
            address(manager),
            address(infToken),
            address(0)
        );
    }

    // Join Contest Edge Cases
    function testCannotJoinWithZeroStake() public {
        vm.prank(participant1);
        vm.expectRevert("Stake amount must be greater than 0");
        contest.joinContest(0);
    }

    function testCannotJoinWhenNotActive() public {
        vm.prank(creator);
        contest.pauseContest();
        
        vm.startPrank(participant1);
        infToken.approve(address(contest), STAKE_AMOUNT);
        vm.expectRevert("Contest is paused");
        contest.joinContest(STAKE_AMOUNT);
        vm.stopPrank();
    }

    event MetadataUpdated(string oldMetadataURI, string newMetadataURI);
    event ParticipantLeft(address indexed participant);
}
