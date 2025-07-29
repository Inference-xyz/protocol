// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../contracts/Contest.sol";
import "../contracts/interfaces/IContest.sol";
import "../contracts/MockVerifier.sol";
import "../contracts/InfToken.sol";

contract ContestTest is Test {
    Contest public contest;
    MockVerifier public verifier;
    InfToken public infToken;
    
    address public creator = address(0x1);
    address public participant1 = address(0x2);
    address public participant2 = address(0x3);
    address public participant3 = address(0x4);
    address public validator1 = address(0x5);
    address public validator2 = address(0x6);
    
    string constant METADATA_URI = "ipfs://test-metadata";
    uint256 constant DURATION = 7 days;
    bytes32 constant SCORING_MODEL_HASH = bytes32(uint256(0x123456789));
    
    event ContestInitialized(address indexed creator, string metadataURI, uint256 duration, bytes32 scoringModelHash, address[] validators);
    event ParticipantJoined(address indexed participant, uint256 timestamp);
    event ParticipantLeft(address indexed participant, uint256 timestamp);
    event EntrySubmitted(address indexed participant, string metadataURI, bytes32 inputHash, bytes32 outputHash, uint256 timestamp);
    event EntryVerified(address indexed participant, bool isValid, uint256 score, address validator);
    event ContestFinalized(IContest.Winner[] winners, uint256 totalDistributed);
    event RewardClaimed(address indexed participant, uint256 amount);
    event RewardsDistributed(uint256 amount);
    event ContestPaused();
    event ContestUnpaused();
    event ValidatorAdded(address indexed validator);
    event ValidatorRemoved(address indexed validator);
    
    function setUp() public {
        contest = new Contest();
        verifier = new MockVerifier();
        infToken = new InfToken();
        
        // Fund creator with tokens
        infToken.transfer(creator, 1000 * 10**18);
    }
    
    function testInitialize() public {
        address[] memory validators = new address[](2);
        validators[0] = validator1;
        validators[1] = validator2;
        
        vm.expectEmit(true, false, false, true);
        emit ContestInitialized(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        contest.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        IContest.ContestInfo memory info = contest.getContestInfo();
        assertEq(info.creator, creator);
        assertEq(info.metadataURI, METADATA_URI);
        assertEq(info.duration, DURATION);
        assertEq(uint256(info.status), uint256(IContest.ContestStatus.Active));
        assertEq(info.participantCount, 0);
        assertEq(info.totalRewards, 0);
        assertEq(info.scoringModelHash, SCORING_MODEL_HASH);
        assertEq(info.validators.length, 2);
        assertTrue(info.startTime > 0);
        
        // Check validators
        assertTrue(contest.isValidator(validator1));
        assertTrue(contest.isValidator(validator2));
        assertFalse(contest.isValidator(participant1));
    }
    
    function testInitializeRevert() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        
        // Test double initialization
        contest.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        vm.expectRevert("Already initialized");
        contest.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        // Test invalid creator
        Contest newContest = new Contest();
        vm.expectRevert("Invalid creator");
        newContest.initialize(address(0), METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        // Test empty validators
        Contest newContest2 = new Contest();
        address[] memory emptyValidators = new address[](0);
        vm.expectRevert("Must have at least one validator");
        newContest2.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, emptyValidators);
        
        // Test invalid validator
        Contest newContest3 = new Contest();
        address[] memory invalidValidators = new address[](1);
        invalidValidators[0] = address(0);
        vm.expectRevert("Invalid validator");
        newContest3.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, invalidValidators);
    }
    
    function testSetVerifier() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        contest.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        vm.prank(creator);
        contest.setVerifier(address(verifier));
        
        // Test non-creator setting verifier
        vm.expectRevert("Only creator can call this");
        vm.prank(participant1);
        contest.setVerifier(address(verifier));
        
        // Test invalid verifier
        vm.expectRevert("Invalid verifier");
        vm.prank(creator);
        contest.setVerifier(address(0));
    }
    
    function testSetInfToken() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        contest.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        vm.prank(creator);
        contest.setInfToken(address(infToken));
        
        // Test non-creator setting InfToken
        vm.expectRevert("Only creator can call this");
        vm.prank(participant1);
        contest.setInfToken(address(infToken));
        
        // Test invalid InfToken
        vm.expectRevert("Invalid InfToken");
        vm.prank(creator);
        contest.setInfToken(address(0));
    }
    
    function testValidatorManagement() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        contest.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        // Test adding validator
        vm.expectEmit(true, false, false, false);
        emit ValidatorAdded(validator2);
        
        vm.prank(creator);
        contest.addValidator(validator2);
        
        assertTrue(contest.isValidator(validator2));
        address[] memory allValidators = contest.getValidators();
        assertEq(allValidators.length, 2);
        
        // Test adding duplicate validator
        vm.expectRevert("Already a validator");
        vm.prank(creator);
        contest.addValidator(validator2);
        
        // Test adding invalid validator
        vm.expectRevert("Invalid validator");
        vm.prank(creator);
        contest.addValidator(address(0));
        
        // Test non-creator adding validator
        vm.expectRevert("Only creator can call this");
        vm.prank(participant1);
        contest.addValidator(participant1);
        
        // Test removing validator
        vm.expectEmit(true, false, false, false);
        emit ValidatorRemoved(validator2);
        
        vm.prank(creator);
        contest.removeValidator(validator2);
        
        assertFalse(contest.isValidator(validator2));
        allValidators = contest.getValidators();
        assertEq(allValidators.length, 1);
        
        // Test removing last validator
        vm.expectRevert("Cannot remove last validator");
        vm.prank(creator);
        contest.removeValidator(validator1);
        
        // Test removing non-validator
        vm.expectRevert("Not a validator");
        vm.prank(creator);
        contest.removeValidator(participant1);
        
        // Test non-creator removing validator
        vm.expectRevert("Only creator can call this");
        vm.prank(participant1);
        contest.removeValidator(validator1);
    }
    
    function testJoinContest() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        contest.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        vm.expectEmit(true, false, false, false);
        emit ParticipantJoined(participant1, block.timestamp);
        
        vm.prank(participant1);
        contest.joinContest();
        
        assertTrue(contest.isParticipant(participant1));
        
        IContest.Participant memory participant = contest.getParticipant(participant1);
        assertEq(participant.account, participant1);
        assertTrue(participant.isActive);
        assertEq(participant.totalRewards, 0);
        assertEq(participant.score, 0);
        
        IContest.ContestInfo memory info = contest.getContestInfo();
        assertEq(info.participantCount, 1);
        
        address[] memory participants = contest.getParticipants();
        assertEq(participants.length, 1);
        assertEq(participants[0], participant1);
    }
    
    function testJoinContestRevert() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        contest.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        vm.prank(participant1);
        contest.joinContest();
        
        // Test double joining
        vm.expectRevert("Already a participant");
        vm.prank(participant1);
        contest.joinContest();
        
        // Test joining when paused
        vm.prank(creator);
        contest.pause();
        
        vm.expectRevert("Contest not active");
        vm.prank(participant2);
        contest.joinContest();
    }
    
    function testLeaveContest() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        contest.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        vm.prank(participant1);
        contest.joinContest();
        
        vm.prank(participant2);
        contest.joinContest();
        
        assertEq(contest.getParticipants().length, 2);
        
        vm.expectEmit(true, false, false, false);
        emit ParticipantLeft(participant1, block.timestamp);
        
        vm.prank(participant1);
        contest.leaveContest();
        
        assertFalse(contest.isParticipant(participant1));
        
        IContest.ContestInfo memory info = contest.getContestInfo();
        assertEq(info.participantCount, 1);
        
        address[] memory participants = contest.getParticipants();
        assertEq(participants.length, 1);
        assertEq(participants[0], participant2);
    }
    
    function testLeaveContestRevert() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        contest.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        // Test leaving without joining
        vm.expectRevert("Not a participant");
        vm.prank(participant1);
        contest.leaveContest();
    }
    
    function testSubmitEntry() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        contest.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        vm.prank(creator);
        contest.setVerifier(address(verifier));
        
        vm.prank(creator);
        contest.setInfToken(address(infToken));
        
        vm.prank(participant1);
        contest.joinContest();
        
        string memory submissionURI = "ipfs://submission-1";
        bytes32 inputHash = bytes32(uint256(0x111));
        bytes32 outputHash = bytes32(uint256(0x222));
        bytes memory zkProof = "0x123456";
        uint256[] memory publicInputs = new uint256[](1);
        publicInputs[0] = 100; // Score
        
        vm.expectEmit(true, false, false, false);
        emit EntrySubmitted(participant1, submissionURI, inputHash, outputHash, block.timestamp);
        
        vm.prank(participant1);
        contest.submitEntry(submissionURI, inputHash, outputHash, zkProof, publicInputs);
        
        IContest.Submission[] memory submissions = contest.getSubmissions();
        assertEq(submissions.length, 1);
        assertEq(submissions[0].participant, participant1);
        assertEq(submissions[0].metadataURI, submissionURI);
        assertEq(submissions[0].inputHash, inputHash);
        assertEq(submissions[0].outputHash, outputHash);
        assertTrue(submissions[0].verified);
        assertEq(submissions[0].validator, participant1);
        assertTrue(submissions[0].timestamp > 0);
        
        // Check participant score was updated
        IContest.Participant memory participant = contest.getParticipant(participant1);
        assertEq(participant.score, 100);
    }
    
    function testSubmitEntryRevert() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        contest.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        // Test submitting without joining
        vm.expectRevert("Not a participant");
        vm.prank(participant1);
        contest.submitEntry("ipfs://test", bytes32(0), bytes32(0), "0x", new uint256[](0));
        
        vm.prank(participant1);
        contest.joinContest();
        
        // Test submitting without verifier set
        vm.expectRevert("Verifier not set");
        vm.prank(participant1);
        contest.submitEntry("ipfs://test", bytes32(0), bytes32(0), "0x", new uint256[](0));
        
        vm.prank(creator);
        contest.setVerifier(address(verifier));
        
        // Test submitting when paused
        vm.prank(creator);
        contest.pause();
        
        vm.expectRevert("Contest not active");
        vm.prank(participant1);
        contest.submitEntry("ipfs://test", bytes32(0), bytes32(0), "0x", new uint256[](0));
        
        vm.prank(creator);
        contest.unpause();
        
        // Test missing ZK proof
        vm.expectRevert("Missing ZK proof");
        vm.prank(participant1);
        contest.submitEntry("ipfs://test", bytes32(0), bytes32(0), "", new uint256[](0));
        
        // Test missing public inputs
        vm.expectRevert("Missing public inputs");
        vm.prank(participant1);
        contest.submitEntry("ipfs://test", bytes32(0), bytes32(0), "0x123", new uint256[](0));
    }
    
    function testDistributeRewards() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        contest.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        vm.prank(creator);
        contest.setInfToken(address(infToken));
        
        uint256 rewardAmount = 1000 * 10**18;
        
        // Approve tokens first
        vm.prank(creator);
        infToken.approve(address(contest), rewardAmount);
        
        vm.expectEmit(false, false, false, true);
        emit RewardsDistributed(rewardAmount);
        
        vm.prank(creator);
        contest.distributeRewards(rewardAmount);
        
        IContest.ContestInfo memory info = contest.getContestInfo();
        assertEq(info.totalRewards, rewardAmount);
    }
    
    function testDistributeRewardsRevert() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        contest.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        vm.expectRevert("No rewards to distribute");
        vm.prank(creator);
        contest.distributeRewards(0);
        
        // Test non-creator distributing rewards
        vm.expectRevert("Only creator can call this");
        vm.deal(participant1, 1 ether);
        vm.prank(participant1);
        contest.distributeRewards(1000 * 10**18);
    }
    
    function testCalculateWinners() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        contest.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        vm.prank(creator);
        contest.setVerifier(address(verifier));
        
        // Add participants and submissions
        vm.prank(participant1);
        contest.joinContest();
        
        vm.prank(participant2);
        contest.joinContest();
        
        vm.prank(participant3);
        contest.joinContest();
        
        // Submit entries with different scores
        uint256[] memory publicInputs1 = new uint256[](1);
        publicInputs1[0] = 100;
        
        uint256[] memory publicInputs2 = new uint256[](1);
        publicInputs2[0] = 200;
        
        uint256[] memory publicInputs3 = new uint256[](1);
        publicInputs3[0] = 150;
        
        vm.prank(participant1);
        contest.submitEntry("ipfs://sub1", bytes32(0), bytes32(0), "0x123", publicInputs1);
        
        vm.prank(participant2);
        contest.submitEntry("ipfs://sub2", bytes32(0), bytes32(0), "0x123", publicInputs2);
        
        vm.prank(participant3);
        contest.submitEntry("ipfs://sub3", bytes32(0), bytes32(0), "0x123", publicInputs3);
        
        // Calculate winners
        IContest.Winner[] memory calculatedWinners = contest.calculateWinners();
        assertEq(calculatedWinners.length, 3);
        
        // Should be sorted by score (descending)
        assertEq(calculatedWinners[0].participant, participant2); // Score 200
        assertEq(calculatedWinners[0].score, 200);
        assertEq(calculatedWinners[1].participant, participant3); // Score 150
        assertEq(calculatedWinners[1].score, 150);
        assertEq(calculatedWinners[2].participant, participant1); // Score 100
        assertEq(calculatedWinners[2].score, 100);
    }
    
    function testFinalizeContest() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        contest.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        vm.prank(creator);
        contest.setVerifier(address(verifier));
        
        vm.prank(creator);
        contest.setInfToken(address(infToken));
        
        // Add participants and submissions
        vm.prank(participant1);
        contest.joinContest();
        
        vm.prank(participant2);
        contest.joinContest();
        
        uint256[] memory publicInputs1 = new uint256[](1);
        publicInputs1[0] = 100;
        
        uint256[] memory publicInputs2 = new uint256[](1);
        publicInputs2[0] = 200;
        
        vm.prank(participant1);
        contest.submitEntry("ipfs://sub1", bytes32(0), bytes32(0), "0x123", publicInputs1);
        
        vm.prank(participant2);
        contest.submitEntry("ipfs://sub2", bytes32(0), bytes32(0), "0x123", publicInputs2);
        
        // Distribute rewards
        uint256 totalRewards = 1000 * 10**18;
        vm.prank(creator);
        infToken.approve(address(contest), totalRewards);
        vm.prank(creator);
        contest.distributeRewards(totalRewards);
        
        // Finalize contest
        vm.prank(creator);
        contest.finalizeContest();
        
        IContest.ContestInfo memory info = contest.getContestInfo();
        assertEq(uint256(info.status), uint256(IContest.ContestStatus.Finalized));
        
        // Check rewards distribution (50% for 1st, 30% for 2nd)
        assertEq(contest.getClaimableRewards(participant2), (totalRewards * 50) / 100);
        assertEq(contest.getClaimableRewards(participant1), (totalRewards * 30) / 100);
        
        // Check winners
        IContest.Winner[] memory winners = contest.getWinners();
        assertEq(winners.length, 2);
        assertEq(winners[0].participant, participant2);
        assertEq(winners[1].participant, participant1);
    }
    
    function testFinalizeContestRevert() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        contest.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        // Test finalizing without rewards
        vm.expectRevert("No rewards to distribute");
        vm.prank(creator);
        contest.finalizeContest();
        
        // Test finalizing without submissions
        vm.prank(creator);
        contest.setInfToken(address(infToken));
        
        uint256 totalRewards = 1000 * 10**18;
        vm.prank(creator);
        infToken.approve(address(contest), totalRewards);
        vm.prank(creator);
        contest.distributeRewards(totalRewards);
        
        vm.expectRevert("No valid winners");
        vm.prank(creator);
        contest.finalizeContest();
        
        // Test non-creator finalizing
        vm.prank(participant1);
        contest.joinContest();
        
        uint256[] memory publicInputs = new uint256[](1);
        publicInputs[0] = 100;
        
        vm.prank(participant1);
        contest.submitEntry("ipfs://sub1", bytes32(0), bytes32(0), "0x123", publicInputs);
        
        vm.expectRevert("Only creator can call this");
        vm.prank(participant1);
        contest.finalizeContest();
    }
    
    function testClaimReward() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        contest.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        vm.prank(creator);
        contest.setVerifier(address(verifier));
        
        vm.prank(creator);
        contest.setInfToken(address(infToken));
        
        vm.prank(participant1);
        contest.joinContest();
        
        // Submit entry and finalize
        uint256[] memory publicInputs = new uint256[](1);
        publicInputs[0] = 100;
        
        vm.prank(participant1);
        contest.submitEntry("ipfs://sub1", bytes32(0), bytes32(0), "0x123", publicInputs);
        
        uint256 rewardAmount = 1000 * 10**18;
        vm.prank(creator);
        infToken.approve(address(contest), rewardAmount);
        vm.prank(creator);
        contest.distributeRewards(rewardAmount);
        
        vm.prank(creator);
        contest.finalizeContest();
        
        uint256 balanceBefore = infToken.balanceOf(participant1);
        uint256 claimableAmount = contest.getClaimableRewards(participant1);
        
        vm.expectEmit(true, false, false, true);
        emit RewardClaimed(participant1, claimableAmount);
        
        vm.prank(participant1);
        contest.claimReward();
        
        assertEq(infToken.balanceOf(participant1), balanceBefore + claimableAmount);
        assertEq(contest.getClaimableRewards(participant1), 0);
    }
    
    function testClaimRewardRevert() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        contest.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        vm.expectRevert("No rewards to claim");
        vm.prank(participant1);
        contest.claimReward();
    }
    
    function testPauseUnpause() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        contest.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        vm.expectEmit();
        emit ContestPaused();
        
        vm.prank(creator);
        contest.pause();
        
        IContest.ContestInfo memory info = contest.getContestInfo();
        assertEq(uint256(info.status), uint256(IContest.ContestStatus.Paused));
        
        vm.expectEmit();
        emit ContestUnpaused();
        
        vm.prank(creator);
        contest.unpause();
        
        info = contest.getContestInfo();
        assertEq(uint256(info.status), uint256(IContest.ContestStatus.Active));
    }
    
    function testPauseUnpauseRevert() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        contest.initialize(creator, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        // Test non-creator pausing
        vm.expectRevert("Only creator can call this");
        vm.prank(participant1);
        contest.pause();
        
        vm.prank(creator);
        contest.pause();
        
        // Test pausing already paused contest
        vm.expectRevert("Contest not active");
        vm.prank(creator);
        contest.pause();
        
        // Test non-creator unpausing
        vm.expectRevert("Only creator can call this");
        vm.prank(participant1);
        contest.unpause();
        
        vm.prank(creator);
        contest.unpause();
        
        // Test unpausing active contest
        vm.expectRevert("Contest not paused");
        vm.prank(creator);
        contest.unpause();
    }
    
    function testIsActive() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        
        // Test everlasting contest (duration = 0)
        contest.initialize(creator, METADATA_URI, 0, SCORING_MODEL_HASH, validators);
        assertTrue(contest.isActive());
        
        vm.prank(creator);
        contest.pause();
        assertFalse(contest.isActive());
        
        // Test time-limited contest
        Contest timedContest = new Contest();
        timedContest.initialize(creator, METADATA_URI, 1 hours, SCORING_MODEL_HASH, validators);
        assertTrue(timedContest.isActive());
        
        // Fast forward past duration
        vm.warp(block.timestamp + 2 hours);
        assertFalse(timedContest.isActive());
    }
} 