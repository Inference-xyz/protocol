// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../contracts/Contest.sol";
import "../contracts/ContestManager.sol";
import "../contracts/InfToken.sol";
import "../contracts/types/ReviewStructs.sol";
import "../contracts/interfaces/IContestManager.sol";

contract QualityControlTest is Test {
    Contest public contest;
    ContestManager public manager;
    InfToken public infToken;
    
    address public creator = address(0x101);
    address public participant1 = address(0x102);
    address public participant2 = address(0x103);
    address public participant3 = address(0x104);
    address public qcAddress = address(0x200);
    
    uint256 public constant STAKE_AMOUNT = 100 ether;
    string public constant METADATA_URI = "https://example.com/contest-metadata";
    
    IContestManager.ContestConfig public testConfig;
    
    function setUp() public {
        infToken = new InfToken();
        manager = new ContestManager(address(infToken));
        
        testConfig = IContestManager.ContestConfig({
            metadataURI: METADATA_URI,
            duration: 14 days,
            stakingAmount: STAKE_AMOUNT,
            qcAddress: qcAddress
        });
        
        vm.prank(creator);
        address contestAddress = manager.createContest(testConfig);
        contest = Contest(contestAddress);
        
        infToken.transfer(creator, 10000 ether);
        infToken.transfer(participant1, 10000 ether);
        infToken.transfer(participant2, 10000 ether);
        infToken.transfer(participant3, 10000 ether);
    }

    function testQCAddressSet() public {
        IContest.ContestInfo memory info = contest.getContestInfo();
        assertEq(info.qcAddress, qcAddress);
    }

    function testQCCanCommitReview() public {
        // Setup at least 2 participants and submissions
        address[] memory participants = new address[](2);
        participants[0] = participant1;
        participants[1] = participant2;
        
        for (uint256 i = 0; i < participants.length; i++) {
            vm.startPrank(participants[i]);
            infToken.approve(address(contest), STAKE_AMOUNT);
            contest.joinContest(STAKE_AMOUNT);
            
            string memory ipfsURI = string(abi.encodePacked("QmHash", vm.toString(i)));
            bytes32 outputHash = keccak256(abi.encodePacked("output", i));
            uint256 nonce = 1001 + i;
            bytes32 commitHash = keccak256(abi.encodePacked(ipfsURI, outputHash, nonce));
            
            contest.commitSubmission(commitHash);
            vm.stopPrank();
        }
        
        // Reveal
        IContest.ContestInfo memory info = contest.getContestInfo();
        vm.warp(info.startTime + (info.duration / 4) + 1);
        
        for (uint256 i = 0; i < participants.length; i++) {
            string memory ipfsURI = string(abi.encodePacked("QmHash", vm.toString(i)));
            bytes32 outputHash = keccak256(abi.encodePacked("output", i));
            uint256 nonce = 1001 + i;
            
            vm.prank(participants[i]);
            contest.revealSubmission(i + 1, ipfsURI, outputHash, nonce);
        }
        
        // Generate assignments
        vm.warp(info.startTime + (info.duration / 2) + 1);
        vm.prank(creator);
        contest.generateReviewAssignments();
        
        // Check QC got assignments
        uint256[] memory qcAssignments = contest.getReviewAssignmentsForReviewer(qcAddress);
        assertTrue(qcAssignments.length > 0, "QC should have review assignments");
    }

    function testQCCanOverrideScore() public {
        // Setup: create submissions and reviews
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
        
        // Reveal phase
        vm.warp(info.startTime + (info.duration / 4) + 1);
        for (uint256 i = 0; i < participants.length; i++) {
            string memory ipfsURI = string(abi.encodePacked("QmHash", vm.toString(i)));
            bytes32 outputHash = keccak256(abi.encodePacked("output", i));
            uint256 nonce = 10000 + i;
            
            vm.prank(participants[i]);
            contest.revealSubmission(i + 1, ipfsURI, outputHash, nonce);
        }
        
        // Generate review assignments
        vm.warp(info.startTime + (info.duration / 2) + 1);
        vm.prank(creator);
        contest.generateReviewAssignments();
        
        // Get QC's first assigned submission
        uint256[] memory qcAssignments = contest.getReviewAssignmentsForReviewer(qcAddress);
        require(qcAssignments.length > 0, "QC needs assignments");
        uint256 submissionId = qcAssignments[0];
        
        // QC overrides the score
        uint256 qcScore = 95;
        vm.prank(qcAddress);
        contest.overrideSubmissionScore(submissionId, qcScore);
        
        // Verify the score was overridden
        ReviewStructs.Submission memory submission = contest.getSubmission(submissionId);
        assertEq(submission.aggregatedScore, qcScore, "QC should override score");
    }

    function testOnlyQCCanOverride() public {
        // Setup at least 2 participants
        address[] memory participants = new address[](2);
        participants[0] = participant1;
        participants[1] = participant2;
        
        for (uint256 i = 0; i < participants.length; i++) {
            vm.startPrank(participants[i]);
            infToken.approve(address(contest), STAKE_AMOUNT);
            contest.joinContest(STAKE_AMOUNT);
            
            string memory ipfsURI = string(abi.encodePacked("QmHash", vm.toString(i)));
            bytes32 outputHash = keccak256(abi.encodePacked("output", i));
            uint256 nonce = 1001 + i;
            bytes32 commitHash = keccak256(abi.encodePacked(ipfsURI, outputHash, nonce));
            
            contest.commitSubmission(commitHash);
            vm.stopPrank();
        }
        
        IContest.ContestInfo memory info = contest.getContestInfo();
        vm.warp(info.startTime + (info.duration / 4) + 1);
        
        for (uint256 i = 0; i < participants.length; i++) {
            string memory ipfsURI = string(abi.encodePacked("QmHash", vm.toString(i)));
            bytes32 outputHash = keccak256(abi.encodePacked("output", i));
            uint256 nonce = 1001 + i;
            
            vm.prank(participants[i]);
            contest.revealSubmission(i + 1, ipfsURI, outputHash, nonce);
        }
        
        vm.warp(info.startTime + (info.duration / 2) + 1);
        vm.prank(creator);
        contest.generateReviewAssignments();
        
        // Regular participant tries to override - should fail
        vm.prank(participant1);
        vm.expectRevert("Only QC");
        contest.overrideSubmissionScore(1, 95);
    }

    function testQCDoesNotNeedStake() public {
        // Setup at least 2 participants with submissions
        address[] memory participants = new address[](2);
        participants[0] = participant1;
        participants[1] = participant2;
        
        for (uint256 i = 0; i < participants.length; i++) {
            vm.startPrank(participants[i]);
            infToken.approve(address(contest), STAKE_AMOUNT);
            contest.joinContest(STAKE_AMOUNT);
            
            string memory ipfsURI = string(abi.encodePacked("QmHash", vm.toString(i)));
            bytes32 outputHash = keccak256(abi.encodePacked("output", i));
            uint256 nonce = 1001 + i;
            bytes32 commitHash = keccak256(abi.encodePacked(ipfsURI, outputHash, nonce));
            
            contest.commitSubmission(commitHash);
            vm.stopPrank();
        }
        
        IContest.ContestInfo memory info = contest.getContestInfo();
        vm.warp(info.startTime + (info.duration / 4) + 1);
        
        for (uint256 i = 0; i < participants.length; i++) {
            string memory ipfsURI = string(abi.encodePacked("QmHash", vm.toString(i)));
            bytes32 outputHash = keccak256(abi.encodePacked("output", i));
            uint256 nonce = 1001 + i;
            
            vm.prank(participants[i]);
            contest.revealSubmission(i + 1, ipfsURI, outputHash, nonce);
        }
        
        vm.warp(info.startTime + (info.duration / 2) + 1);
        vm.prank(creator);
        contest.generateReviewAssignments();
        
        // QC gets assignments without joining/staking
        uint256[] memory qcAssignments = contest.getReviewAssignmentsForReviewer(qcAddress);
        assertTrue(qcAssignments.length > 0, "QC should get assignments without staking");
        
        // Verify QC is not in participant list
        assertFalse(contest.isParticipant(qcAddress), "QC should not be a participant");
    }

    function testContestWithoutQC() public {
        // Create contest without QC
        IContestManager.ContestConfig memory configNoQC = IContestManager.ContestConfig({
            metadataURI: METADATA_URI,
            duration: 14 days,
            stakingAmount: STAKE_AMOUNT,
            qcAddress: address(0)
        });
        
        vm.prank(creator);
        address contestAddress = manager.createContest(configNoQC);
        Contest contestNoQC = Contest(contestAddress);
        
        IContest.ContestInfo memory info = contestNoQC.getContestInfo();
        assertEq(info.qcAddress, address(0), "No QC should be set");
    }
}

