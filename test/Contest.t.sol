// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../contracts/Contest.sol";
import "../contracts/interfaces/IContest.sol";
import "../contracts/ModelRegistry.sol";
import "../contracts/ZKVerifierRegistry.sol";
import "../contracts/MockVerifier.sol";
import "../contracts/InfToken.sol";

contract ContestTest is Test {
    Contest public contest;
    ModelRegistry public modelRegistry;
    ZKVerifierRegistry public verifierRegistry;
    MockVerifier public mockVerifier;
    InfToken public infToken;
    
    address public creator = address(0x1);
    address public participant1 = address(0x2);
    address public participant2 = address(0x3);
    address public validator1 = address(0x5);
    address public validator2 = address(0x6);
    
    string constant METADATA_URI = "ipfs://test-metadata";
    uint256 constant DURATION = 7 days;
    uint256 constant EPOCH_DURATION = 1 days;
    bytes32 constant SCORING_MODEL_HASH = bytes32(uint256(0x123456789));
    bytes32 constant MODEL_HASH = bytes32(uint256(0x987654321));
    
    function setUp() public {
        // Deploy contracts
        contest = new Contest();
        modelRegistry = new ModelRegistry();
        verifierRegistry = new ZKVerifierRegistry();
        mockVerifier = new MockVerifier();
        infToken = new InfToken();
        
        // Register scoring model in verifier registry
        vm.prank(address(this));
        verifierRegistry.registerVerifier(
            SCORING_MODEL_HASH,
            address(mockVerifier),
            "ipfs://scoring-model-metadata"
        );
        
        // Register inference model in model registry
        vm.prank(participant1);
        modelRegistry.registerModel(MODEL_HASH, "ipfs://model-metadata");
        
        // Fund creator with tokens
        vm.deal(creator, 10 ether);
        infToken.transfer(creator, 1000 * 10**18);
    }
    
    function testInitialize() public {
        address[] memory validators = new address[](2);
        validators[0] = validator1;
        validators[1] = validator2;
        
        vm.expectEmit(true, false, false, true);
        emit IContest.ContestInitialized(
            creator,
            METADATA_URI,
            DURATION,
            EPOCH_DURATION,
            SCORING_MODEL_HASH,
            address(modelRegistry),
            address(verifierRegistry),
            validators
        );
        
        vm.prank(creator);
        contest.initialize(
            creator,
            METADATA_URI,
            DURATION,
            EPOCH_DURATION,
            SCORING_MODEL_HASH,
            address(modelRegistry),
            address(verifierRegistry),
            validators
        );
        
        // Verify initialization
        IContest.ContestInfo memory info = contest.getContestInfo();
        assertEq(info.creator, creator);
        assertEq(info.metadataURI, METADATA_URI);
        assertEq(info.duration, DURATION);
        assertEq(info.epochDuration, EPOCH_DURATION);
        assertEq(info.scoringModelHash, SCORING_MODEL_HASH);
        assertEq(info.currentEpoch, 1); // Should start at epoch 1
        assertEq(uint256(info.status), uint256(IContest.ContestStatus.Active));
        
        // Check validators
        address[] memory addedValidators = contest.getValidators();
        assertEq(addedValidators.length, 2);
        assertEq(addedValidators[0], validator1);
        assertEq(addedValidators[1], validator2);
    }
    
    function testJoinContest() public {
        // Initialize contest
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        
        vm.prank(creator);
        contest.initialize(
            creator,
            METADATA_URI,
            DURATION,
            EPOCH_DURATION,
            SCORING_MODEL_HASH,
            address(modelRegistry),
            address(verifierRegistry),
            validators
        );
        
        // Set InfToken
        vm.prank(creator);
        contest.setInfToken(address(infToken));
        
        // Join contest
        vm.prank(participant1);
        contest.joinContest();
        
        // Verify participant joined
        IContest.Participant memory participant = contest.getParticipant(participant1);
        assertTrue(participant.isActive);
        assertEq(participant.account, participant1);
        assertTrue(contest.isParticipant(participant1));
        assertEq(contest.getParticipantCount(), 1);
        
        // Try to join again (should fail)
        vm.prank(participant1);
        vm.expectRevert("Already a participant");
        contest.joinContest();
    }
    
    function testLeaveContest() public {
        // Initialize contest
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        
        vm.prank(creator);
        contest.initialize(
            creator,
            METADATA_URI,
            DURATION,
            EPOCH_DURATION,
            SCORING_MODEL_HASH,
            address(modelRegistry),
            address(verifierRegistry),
            validators
        );
        
        // Set InfToken
        vm.prank(creator);
        contest.setInfToken(address(infToken));
        
        // Join contest
        vm.prank(participant1);
        contest.joinContest();
        
        // Leave contest
        vm.prank(participant1);
        contest.leaveContest();
        
        // Verify participant left
        IContest.Participant memory participant = contest.getParticipant(participant1);
        assertFalse(participant.isActive);
        assertFalse(contest.isParticipant(participant1));
        assertEq(contest.getParticipantCount(), 0);
    }
    
    function testPostInputHashes() public {
        // Initialize contest
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        
        vm.prank(creator);
        contest.initialize(
            creator,
            METADATA_URI,
            DURATION,
            EPOCH_DURATION,
            SCORING_MODEL_HASH,
            address(modelRegistry),
            address(verifierRegistry),
            validators
        );
        
        // Post input hashes
        bytes32[] memory inputHashes = new bytes32[](2);
        inputHashes[0] = bytes32(uint256(0x111));
        inputHashes[1] = bytes32(uint256(0x222));
        
        vm.prank(creator);
        contest.postInputHashes(inputHashes);
        
        // Verify input hashes were posted
        IContest.EpochInfo memory epoch = contest.getEpochInfo(1);
        assertEq(epoch.inputHashes.length, 2);
        assertEq(epoch.inputHashes[0], bytes32(uint256(0x111)));
        assertEq(epoch.inputHashes[1], bytes32(uint256(0x222)));
    }
    
    function testSubmitInference() public {
        // Initialize contest
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        
        vm.prank(creator);
        contest.initialize(
            creator,
            METADATA_URI,
            DURATION,
            EPOCH_DURATION,
            SCORING_MODEL_HASH,
            address(modelRegistry),
            address(verifierRegistry),
            validators
        );
        
        // Set InfToken
        vm.prank(creator);
        contest.setInfToken(address(infToken));
        
        // Join contest
        vm.prank(participant1);
        contest.joinContest();
        
        // Post input hashes
        bytes32[] memory inputHashes = new bytes32[](1);
        inputHashes[0] = bytes32(uint256(0x111));
        
        vm.prank(creator);
        contest.postInputHashes(inputHashes);
        
        // Submit inference
        bytes32 inputHash = bytes32(uint256(0x111));
        bytes32 outputHash = bytes32(uint256(0x333));
        bytes memory zkProof = hex"123456";
        uint256[] memory publicInputs = new uint256[](0);
        
        vm.prank(participant1);
        contest.submitInference(1, inputHash, MODEL_HASH, outputHash, zkProof, publicInputs);
        
        // Verify inference submission
        IContest.InferenceSubmission[] memory submissions = contest.getInferenceSubmissions(1);
        assertEq(submissions.length, 1);
        assertEq(submissions[0].participant, participant1);
        assertEq(submissions[0].inputHash, inputHash);
        assertEq(submissions[0].modelHash, MODEL_HASH);
        assertEq(submissions[0].outputHash, outputHash);
    }
    
    function testSubmitScoring() public {
        // Initialize contest
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        
        vm.prank(creator);
        contest.initialize(
            creator,
            METADATA_URI,
            DURATION,
            EPOCH_DURATION,
            SCORING_MODEL_HASH,
            address(modelRegistry),
            address(verifierRegistry),
            validators
        );
        
        // Set InfToken
        vm.prank(creator);
        contest.setInfToken(address(infToken));
        
        // Join contest
        vm.prank(participant1);
        contest.joinContest();
        
        // Post input hashes
        bytes32[] memory inputHashes = new bytes32[](1);
        inputHashes[0] = bytes32(uint256(0x111));
        
        vm.prank(creator);
        contest.postInputHashes(inputHashes);
        
        // Submit inference
        bytes32 inputHash = bytes32(uint256(0x111));
        bytes32 outputHash = bytes32(uint256(0x333));
        bytes memory zkProof = hex"123456";
        uint256[] memory publicInputs = new uint256[](0);
        
        vm.prank(participant1);
        contest.submitInference(1, inputHash, MODEL_HASH, outputHash, zkProof, publicInputs);
        
        // Submit scoring
        uint256[] memory scores = new uint256[](2);
        scores[0] = 85;
        scores[1] = 92;
        
        vm.prank(validator1);
        contest.submitScoring(1, inputHash, participant1, outputHash, scores, zkProof, publicInputs);
        
        // Verify scoring submission
        IContest.ScoringSubmission[] memory scoringSubmissions = contest.getScoringSubmissions(1);
        assertEq(scoringSubmissions.length, 1);
        assertEq(scoringSubmissions[0].scorer, validator1);
        assertEq(scoringSubmissions[0].participant, participant1);
        assertEq(scoringSubmissions[0].scores.length, 2);
        assertEq(scoringSubmissions[0].scores[0], 85);
        assertEq(scoringSubmissions[0].scores[1], 92);
        
        // Verify participant score was updated
        IContest.Participant memory participant = contest.getParticipant(participant1);
        assertEq(participant.score, 88); // Average of 85 and 92
    }
    
    function testDistributeRewards() public {
        // Initialize contest
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        
        vm.prank(creator);
        contest.initialize(
            creator,
            METADATA_URI,
            DURATION,
            EPOCH_DURATION,
            SCORING_MODEL_HASH,
            address(modelRegistry),
            address(verifierRegistry),
            validators
        );
        
        // Set InfToken
        vm.prank(creator);
        contest.setInfToken(address(infToken));
        
        // Approve contest to spend tokens
        vm.prank(creator);
        infToken.approve(address(contest), 1000 * 10**18);
        
        // Distribute rewards
        vm.prank(creator);
        contest.distributeRewards(1000 * 10**18);
        
        // Verify rewards were added
        IContest.ContestInfo memory info = contest.getContestInfo();
        assertEq(info.totalRewards, 1000 * 10**18);
    }
    
    function testFinalizeContest() public {
        // Initialize contest
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        
        vm.prank(creator);
        contest.initialize(
            creator,
            METADATA_URI,
            DURATION,
            EPOCH_DURATION,
            SCORING_MODEL_HASH,
            address(modelRegistry),
            address(verifierRegistry),
            validators
        );
        
        // Set InfToken
        vm.prank(creator);
        contest.setInfToken(address(infToken));
        
        // Join contest
        vm.prank(participant1);
        contest.joinContest();
        
        vm.prank(participant2);
        contest.joinContest();
        
        // Approve and distribute rewards
        vm.prank(creator);
        infToken.approve(address(contest), 1000 * 10**18);
        
        vm.prank(creator);
        contest.distributeRewards(1000 * 10**18);
        
        // Post input hashes first
        bytes32[] memory inputHashes = new bytes32[](1);
        inputHashes[0] = bytes32(uint256(0x111));
        vm.prank(creator);
        contest.postInputHashes(inputHashes);
        
        // Submit inference first
        bytes32 inputHash = bytes32(uint256(0x111));
        bytes32 outputHash = bytes32(uint256(0x333));
        bytes memory zkProof = hex"123456";
        uint256[] memory publicInputs = new uint256[](0);
        
        vm.prank(participant1);
        contest.submitInference(1, inputHash, MODEL_HASH, outputHash, zkProof, publicInputs);
        
        vm.prank(participant2);
        contest.submitInference(1, inputHash, MODEL_HASH, outputHash, zkProof, publicInputs);
        
        // Set participant scores manually for testing
        uint256[] memory scores1 = new uint256[](1);
        scores1[0] = 100;
        vm.prank(validator1);
        contest.submitScoring(1, inputHash, participant1, outputHash, scores1, zkProof, publicInputs);
        
        uint256[] memory scores2 = new uint256[](1);
        scores2[0] = 50;
        vm.prank(validator1);
        contest.submitScoring(1, inputHash, participant2, outputHash, scores2, zkProof, publicInputs);
        
        // Finalize contest
        vm.prank(creator);
        contest.finalizeContest();
        
        // Verify contest was finalized
        IContest.ContestInfo memory info = contest.getContestInfo();
        assertEq(uint256(info.status), uint256(IContest.ContestStatus.Finalized));
        
        // Verify winners
        IContest.Winner[] memory winners = contest.getWinners();
        assertEq(winners.length, 2);
        assertEq(winners[0].participant, participant1);
        assertEq(winners[0].score, 100);
        assertEq(winners[1].participant, participant2);
        assertEq(winners[1].score, 50);
    }
} 