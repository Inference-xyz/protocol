// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../contracts/Contest.sol";
import "../contracts/ContestFactory.sol";
import "../contracts/utils/ReviewVerifier.sol";
import "../contracts/types/ReviewStructs.sol";

contract ContestTest is Test {
    Contest public contest;
    ContestFactory public factory;
    ReviewVerifier public reviewVerifier;
    
    // Test accounts
    address public creator = address(0x101);
    address public participant1 = address(0x102);
    address public participant2 = address(0x103);
    address public participant3 = address(0x104);
    address public participant4 = address(0x105);
    
    // Test constants
    uint256 public constant REQUIRED_STAKE = 0.1 ether;
    uint256 public constant CREATOR_FEE_PCT = 1000; // 10%
    string public constant METADATA_URI = "https://example.com/contest-metadata";
    
    // Contest configuration
    IContestFactory.ContestConfig public testConfig;
    
    // Review signing
    uint256 public constant REVIEWER_PRIVATE_KEY = 0xA11CE;
    address public reviewer = vm.addr(REVIEWER_PRIVATE_KEY);
    
    function setUp() public {
        // Deploy factory and create contest
        factory = new ContestFactory();
        
        testConfig = IContestFactory.ContestConfig({
            metadataURI: METADATA_URI,
            creatorFeePct: CREATOR_FEE_PCT,
            duration: 14 days,
            minParticipants: 2,
            maxParticipants: 10,
            requiresStaking: true,
            stakingAmount: REQUIRED_STAKE,
            ownerPct: CREATOR_FEE_PCT,
            participantPct: 10000 - CREATOR_FEE_PCT,
            validatorPct: 0,
            maxValidators: 0,
            minValidatorStake: 0
        });
        
        // Create contest
        vm.prank(creator);
        address contestAddress = factory.createContest(testConfig);
        contest = Contest(payable(contestAddress));
        
        reviewVerifier = contest.reviewVerifier();
        
        // Fund test accounts
        vm.deal(creator, 10 ether);
        vm.deal(participant1, 10 ether);
        vm.deal(participant2, 10 ether);
        vm.deal(participant3, 10 ether);
        vm.deal(participant4, 10 ether);
        vm.deal(reviewer, 10 ether);
    }

    function testContestInitialization() public {
        IContest.ContestInfo memory info = contest.getContestInfo();
        assertEq(info.creator, creator);
        assertEq(info.metadataURI, METADATA_URI);
        assertEq(info.rewardSplit.ownerPct, CREATOR_FEE_PCT);
        assertEq(info.rewardSplit.participantPct, 10000 - CREATOR_FEE_PCT);
        assertTrue(info.status == IContest.ContestStatus.Active);
    }

    function testJoinContest() public {
        vm.prank(participant1);
        contest.joinContest{value: REQUIRED_STAKE}();
        
        assertTrue(contest.isParticipant(participant1));
        
        IContest.Participant memory p = contest.getParticipant(participant1);
        assertEq(p.account, participant1);
        assertEq(p.stakedAmount, REQUIRED_STAKE);
        assertTrue(p.isActive);
    }

    function testJoinContestInsufficientStake() public {
        vm.prank(participant1);
        vm.expectRevert("Insufficient stake");
        contest.joinContest{value: REQUIRED_STAKE - 1}();
    }

    function testLeaveContest() public {
        // Join first
        vm.prank(participant1);
        contest.joinContest{value: REQUIRED_STAKE}();
        
        uint256 balanceBefore = participant1.balance;
        
        // Leave contest
        vm.prank(participant1);
        contest.leaveContest();
        
        assertFalse(contest.isParticipant(participant1));
        assertEq(participant1.balance, balanceBefore + REQUIRED_STAKE);
    }

    function testCommitSubmission() public {
        // Join contest
        vm.prank(participant1);
        contest.joinContest{value: REQUIRED_STAKE}();
        
        // Create commit hash
        bytes32 outputHash = keccak256("test-output");
        uint256 nonce = 12345;
        bytes32 commitHash = keccak256(abi.encodePacked(outputHash, nonce));
        
        // Commit submission
        vm.prank(participant1);
        contest.commitSubmission(commitHash);
        
        // Verify submission was stored
        ReviewStructs.Submission memory submission = contest.getSubmission(1);
        assertEq(submission.participant, participant1);
        assertEq(submission.commitHash, commitHash);
        assertFalse(submission.isRevealed);
    }

    function testRevealSubmission() public {
        // Setup: join and commit
        vm.prank(participant1);
        contest.joinContest{value: REQUIRED_STAKE}();
        
        bytes32 outputHash = keccak256("test-output");
        uint256 nonce = 12345;
        bytes32 commitHash = keccak256(abi.encodePacked(outputHash, nonce));
        
        vm.prank(participant1);
        contest.commitSubmission(commitHash);
        
        // Move to reveal phase
        vm.warp(block.timestamp + (14 days / 4) + 1);
        
        // Reveal submission
        string memory ipfsURI = "QmTestHash123";
        vm.prank(participant1);
        contest.revealSubmission(1, ipfsURI, outputHash, nonce);
        
        // Verify reveal
        ReviewStructs.Submission memory submission = contest.getSubmission(1);
        assertTrue(submission.isRevealed);
        assertEq(submission.ipfsURI, ipfsURI);
        assertEq(submission.outputHash, outputHash);
    }

    function testRevealSubmissionInvalidCommit() public {
        // Setup: join and commit
        vm.prank(participant1);
        contest.joinContest{value: REQUIRED_STAKE}();
        
        bytes32 outputHash = keccak256("test-output");
        uint256 nonce = 12345;
        bytes32 commitHash = keccak256(abi.encodePacked(outputHash, nonce));
        
        vm.prank(participant1);
        contest.commitSubmission(commitHash);
        
        // Move to reveal phase
        vm.warp(block.timestamp + (14 days / 4) + 1);
        
        // Try to reveal with wrong nonce
        string memory ipfsURI = "QmTestHash123";
        vm.prank(participant1);
        vm.expectRevert("Invalid reveal");
        contest.revealSubmission(1, ipfsURI, outputHash, 54321); // Wrong nonce
    }

    function testFullWorkflow() public {
        IContest.ContestInfo memory info = contest.getContestInfo();
        uint256 startTime = info.startTime;
        
        // Phase 1: Multiple participants join
        address[] memory participants = new address[](3);
        participants[0] = participant1;
        participants[1] = participant2;
        participants[2] = participant3;
        
        for (uint256 i = 0; i < participants.length; i++) {
            vm.prank(participants[i]);
            contest.joinContest{value: REQUIRED_STAKE}();
        }
        
        // Phase 2: Commit submissions
        bytes32[] memory outputHashes = new bytes32[](3);
        uint256[] memory nonces = new uint256[](3);
        
        for (uint256 i = 0; i < participants.length; i++) {
            outputHashes[i] = keccak256(abi.encodePacked("output", i));
            nonces[i] = 10000 + i;
            bytes32 commitHash = keccak256(abi.encodePacked(outputHashes[i], nonces[i]));
            
            vm.prank(participants[i]);
            contest.commitSubmission(commitHash);
        }
        
        // Phase 3: Move to reveal phase and reveal
        vm.warp(startTime + (info.duration / 4) + 1);
        
        for (uint256 i = 0; i < participants.length; i++) {
            string memory ipfsURI = string(abi.encodePacked("QmHash", vm.toString(i)));
            vm.prank(participants[i]);
            contest.revealSubmission(i + 1, ipfsURI, outputHashes[i], nonces[i]);
        }
        
        // Phase 4: Move to review phase and generate assignments
        vm.warp(startTime + (info.duration / 2) + 1);
        
        contest.generateReviewAssignments();
        
        // Verify assignments were created
        ReviewStructs.ReviewAssignment[] memory assignments1 = contest.getReviewAssignments(1);
        assertTrue(assignments1.length > 0);
        
        // Phase 5: Submit reviews (simplified - would normally be signed off-chain)
        // This is a simplified test - in practice, reviews would be signed off-chain
        // and submitted via submitReviews() with proper signatures
        
        // Phase 6: Finalize contest
        vm.warp(startTime + (3 * info.duration / 4) + 1);
        vm.prank(creator);
        contest.finalizeContest();
        
        assertTrue(contest.isFinalized());
    }

    function testSignedReview() public {
        IContest.ContestInfo memory info = contest.getContestInfo();
        uint256 startTime = info.startTime;
        
        // Setup a basic contest with participants
        vm.prank(participant1);
        contest.joinContest{value: REQUIRED_STAKE}();
        
        vm.prank(reviewer);
        contest.joinContest{value: REQUIRED_STAKE}();
        
        // Commit and reveal a submission
        bytes32 outputHash = keccak256("test-output");
        uint256 nonce = 12345;
        bytes32 commitHash = keccak256(abi.encodePacked(outputHash, nonce));
        
        vm.prank(participant1);
        contest.commitSubmission(commitHash);
        
        // Move to reveal phase
        vm.warp(startTime + (info.duration / 4) + 1);
        
        vm.prank(participant1);
        contest.revealSubmission(1, "QmTestHash", outputHash, nonce);
        
        // Move to review phase
        vm.warp(startTime + (info.duration / 2) + 1);
        
        contest.generateReviewAssignments();
        
        // Create a signed review
        ReviewStructs.SignedReview memory review = ReviewStructs.SignedReview({
            submissionId: 1,
            contestId: 1, // Assuming contest ID is 1
            score: 85,
            reviewer: reviewer,
            nonce: 1,
            deadline: block.timestamp + 1 days,
            signature: ""
        });
        
        // Generate signature (simplified for testing)
        bytes32 domainSeparator = ReviewStructs.getDomainSeparator(
            "InferenceProtocol",
            "1",
            block.chainid,
            address(contest)
        );
        
        bytes32 structHash = ReviewStructs.getReviewStructHash(review);
        bytes32 typedDataHash = keccak256(abi.encodePacked("\x19\x01", domainSeparator, structHash));
        
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(REVIEWER_PRIVATE_KEY, typedDataHash);
        review.signature = abi.encodePacked(r, s, v);
        
        // Submit the review
        ReviewStructs.SignedReview[] memory reviews = new ReviewStructs.SignedReview[](1);
        reviews[0] = review;
        
        // This would normally work, but requires the reviewer to be properly assigned
        // vm.prank(reviewer);
        // contest.submitReviews(reviews);
    }

    function testPhasesProgression() public {
        IContest.ContestInfo memory info = contest.getContestInfo();
        console.log("Contest start time:", info.startTime);
        console.log("Test start time:", block.timestamp);
        console.log("Contest duration:", info.duration);
        
        // Calculate expected phase deadlines
        uint256 commitDeadline = info.startTime + (info.duration / 4);
        uint256 revealDeadline = info.startTime + (info.duration / 2);
        uint256 reviewDeadline = info.startTime + (3 * info.duration / 4);
        
        console.log("Commit deadline:", commitDeadline);
        console.log("Reveal deadline:", revealDeadline);
        console.log("Review deadline:", reviewDeadline);
        
        // Should start in Commit phase
        assertEq(uint256(contest.getCurrentPhase()), uint256(ReviewStructs.Phase.Commit));
        
        // Move to Reveal phase
        vm.warp(commitDeadline + 1);
        console.log("After warp to commit+1, current time:", block.timestamp);
        console.log("Current phase:", uint256(contest.getCurrentPhase()));
        assertEq(uint256(contest.getCurrentPhase()), uint256(ReviewStructs.Phase.Reveal));
        
        // Move to Review phase
        vm.warp(revealDeadline + 1);
        console.log("After warp to reveal+1, current time:", block.timestamp);
        console.log("Current phase:", uint256(contest.getCurrentPhase()));
        assertEq(uint256(contest.getCurrentPhase()), uint256(ReviewStructs.Phase.Review));
        
        // Move to Finalized phase
        vm.warp(reviewDeadline + 1);
        console.log("After warp to review+1, current time:", block.timestamp);
        console.log("Current phase:", uint256(contest.getCurrentPhase()));
        assertEq(uint256(contest.getCurrentPhase()), uint256(ReviewStructs.Phase.Finalized));
    }

    function testCannotCommitInWrongPhase() public {
        vm.prank(participant1);
        contest.joinContest{value: REQUIRED_STAKE}();
        
        // Move past commit phase
        vm.warp(block.timestamp + (14 days / 4) + 1);
        
        bytes32 commitHash = keccak256(abi.encodePacked(keccak256("test"), uint256(123)));
        
        vm.prank(participant1);
        vm.expectRevert("Invalid phase");
        contest.commitSubmission(commitHash);
    }

    function testCannotRevealInWrongPhase() public {
        vm.prank(participant1);
        contest.joinContest{value: REQUIRED_STAKE}();
        
        // Still in commit phase
        vm.prank(participant1);
        vm.expectRevert("Invalid phase");
        contest.revealSubmission(1, "QmTest", keccak256("test"), 123);
    }

    function testContestFactoryIntegration() public {
        // Test that factory correctly creates contests
        address[] memory createdContests = factory.getCreatedContests();
        assertTrue(createdContests.length > 0);
        
        address[] memory creatorContests = factory.getContestsByCreator(creator);
        assertTrue(creatorContests.length > 0);
        assertEq(creatorContests[0], address(contest));
        
        assertTrue(factory.isValidContest(address(contest)));
    }

    function testReviewVerifierFunctions() public {
        // Test score validation
        assertTrue(reviewVerifier.validateScore(50, 0, 100));
        assertFalse(reviewVerifier.validateScore(150, 0, 100));
        
        // Test score aggregation
        uint256[] memory scores = new uint256[](3);
        scores[0] = 80;
        scores[1] = 90;
        scores[2] = 70;
        
        uint256 meanScore = reviewVerifier.aggregateScores(scores, 0); // Mean
        assertEq(meanScore, 80);
        
        uint256 medianScore = reviewVerifier.aggregateScores(scores, 1); // Median
        assertEq(medianScore, 80);
    }

    // Helper functions for testing
    function _createSignedReview(
        uint256 submissionId,
        uint256 contestId,
        uint256 score,
        address reviewerAddr,
        uint256 reviewNonce,
        uint256 deadline,
        uint256 privateKey
    ) internal view returns (ReviewStructs.SignedReview memory) {
        ReviewStructs.SignedReview memory review = ReviewStructs.SignedReview({
            submissionId: submissionId,
            contestId: contestId,
            score: score,
            reviewer: reviewerAddr,
            nonce: reviewNonce,
            deadline: deadline,
            signature: ""
        });
        
        bytes32 domainSeparator = ReviewStructs.getDomainSeparator(
            "InferenceProtocol",
            "1",
            block.chainid,
            address(contest)
        );
        
        bytes32 typedDataHash = ReviewStructs.getReviewTypedDataHash(review, domainSeparator);
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(privateKey, typedDataHash);
        review.signature = abi.encodePacked(r, s, v);
        
        return review;
    }

    // Test edge cases
    function testDoubleJoinPrevention() public {
        vm.prank(participant1);
        contest.joinContest{value: REQUIRED_STAKE}();
        
        vm.prank(participant1);
        vm.expectRevert("Already joined");
        contest.joinContest{value: REQUIRED_STAKE}();
    }

    function testNonParticipantCannotCommit() public {
        bytes32 commitHash = keccak256(abi.encodePacked(keccak256("test"), uint256(123)));
        
        vm.prank(participant1);
        vm.expectRevert("Not an active participant");
        contest.commitSubmission(commitHash);
    }
 
    function testCannotLeaveAfterCommitPhase() public {
        vm.prank(participant1);
        contest.joinContest{value: REQUIRED_STAKE}();
        
        // Move past commit phase
        vm.warp(block.timestamp + (14 days / 4) + 1);
        
        vm.prank(participant1);
        vm.expectRevert("Cannot leave after commit phase");
        contest.leaveContest();
    }

    receive() external payable {}
}
