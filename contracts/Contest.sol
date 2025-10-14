// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "./interfaces/IContest.sol";
import "./types/ReviewStructs.sol";

contract Contest is IContest, Ownable, ReentrancyGuard {
    ContestInfo public contestInfo;
    
    uint256 public nextSubmissionId = 1;
    bool public isFinalized;
    
    uint256 public constant MIN_SCORE = 0;
    uint256 public constant MAX_SCORE = 100;
    
    mapping(address => Participant) public participants;
    mapping(uint256 => ReviewStructs.Submission) public submissions;
    mapping(address => uint256[]) public participantSubmissions;
    mapping(address => mapping(uint256 => uint256)) public reviews; // reviewer => submissionId => score (deprecated, kept for compatibility)
    mapping(address => mapping(uint256 => ReviewStructs.Review)) public reviewCommits; // reviewer => submissionId => Review
    mapping(address => uint256[]) public reviewAssignments; // reviewer => list of submissionIds to review
    mapping(address => mapping(uint256 => bool)) public isAssignedReviewer; // reviewer => submissionId => isAssigned
    
    address[] public participantList;
    uint256[] public submissionIds;
    uint256 public constant REVIEWS_PER_PARTICIPANT = 5;
    bool public assignmentsGenerated;
    
    modifier onlyParticipant() {
        require(participants[msg.sender].isActive, "Not an active participant");
        _;
    }

    constructor() Ownable(msg.sender) {}

    function initialize(
        address creator,
        string calldata metadataURI,
        uint256 creatorFeePct,
        uint256 duration,
        bool isEverlasting
    ) external override {
        require(contestInfo.creator == address(0), "Already initialized");
        
        contestInfo = ContestInfo({
            creator: creator,
            metadataURI: metadataURI,
            rewardSplit: RewardSplit({
                ownerPct: creatorFeePct,
                participantPct: 10000 - creatorFeePct,
                validatorPct: 0,
                totalPct: 10000
            }),
            startTime: block.timestamp,
            duration: duration,
            currentEpoch: 1,
            status: ContestStatus.Active,
            isEverlasting: isEverlasting,
            validatorSet: address(0),
            scoringVerifier: address(0)
        });
        
        nextSubmissionId = 1;
        _transferOwnership(creator);
        
        emit ContestInitialized(creator, metadataURI, isEverlasting);
    }

    function joinContest() external payable override {
        require(!participants[msg.sender].isActive, "Already joined");
        
        participants[msg.sender] = Participant({
            account: msg.sender,
            joinedAt: block.timestamp,
            currentScore: 0,
            totalRewards: 0,
            lastSubmissionTime: 0,
            isActive: true,
            stakedAmount: msg.value
        });
        
        participantList.push(msg.sender);
        
        emit ParticipantJoined(msg.sender, block.timestamp);
    }

    function leaveContest() external {
        require(participants[msg.sender].isActive, "Not a participant");
        require(!isFinalized, "Contest already finalized");
        
        uint256 stakedAmount = participants[msg.sender].stakedAmount;
        
        // Remove from active participants
        participants[msg.sender].isActive = false;
        
        // Remove from participant list
        for (uint256 i = 0; i < participantList.length; i++) {
            if (participantList[i] == msg.sender) {
                participantList[i] = participantList[participantList.length - 1];
                participantList.pop();
                break;
            }
        }
        
        // Refund stake
        if (stakedAmount > 0) {
            (bool success, ) = msg.sender.call{value: stakedAmount}("");
            require(success, "Refund failed");
        }
    }

    /**
     * @dev Commit a submission (commit phase)
     */
    function commitSubmission(bytes32 commitHash) external onlyParticipant {
        uint256 submissionId = nextSubmissionId++;
        
        submissions[submissionId] = ReviewStructs.Submission({
            participant: msg.sender,
            commitHash: commitHash,
            ipfsURI: "",
            outputHash: bytes32(0),
            commitTime: block.timestamp,
            revealTime: 0,
            isRevealed: false,
            aggregatedScore: 0,
            reviewCount: 0
        });
        
        participantSubmissions[msg.sender].push(submissionId);
        submissionIds.push(submissionId);
        participants[msg.sender].lastSubmissionTime = block.timestamp;
        
        emit SubmissionSubmitted(msg.sender, submissionId, "", commitHash);
    }

    /**
     * @dev Reveal a submission (reveal phase)
     */
    function revealSubmission(
        uint256 submissionId,
        string calldata ipfsURI,
        bytes32 outputHash,
        uint256 nonce
    ) external onlyParticipant {
        require(submissionId < nextSubmissionId, "Invalid submission ID");
        ReviewStructs.Submission storage submission = submissions[submissionId];
        require(submission.participant == msg.sender, "Not your submission");
        require(!submission.isRevealed, "Already revealed");
        
        // Verify the reveal matches the commit
        bytes32 expectedCommit = keccak256(abi.encodePacked(ipfsURI, outputHash, nonce));
        require(submission.commitHash == expectedCommit, "Invalid reveal");
        
        submission.ipfsURI = ipfsURI;
        submission.outputHash = outputHash;
        submission.revealTime = block.timestamp;
        submission.isRevealed = true;
        
        emit SubmissionSubmitted(msg.sender, submissionId, ipfsURI, outputHash);
    }

    /**
     * @dev Commit a review for a submission (commit phase)
     */
    function commitReview(uint256 submissionId, bytes32 commitHash) external onlyParticipant {
        require(submissionId < nextSubmissionId, "Invalid submission ID");
        require(submissions[submissionId].isRevealed, "Submission not revealed yet");
        require(submissions[submissionId].participant != msg.sender, "Cannot review own submission");
        require(reviewCommits[msg.sender][submissionId].commitTime == 0, "Review already committed");
        require(isAssignedReviewer[msg.sender][submissionId], "Not assigned to review this submission");
        
        reviewCommits[msg.sender][submissionId] = ReviewStructs.Review({
            reviewer: msg.sender,
            commitHash: commitHash,
            score: 0,
            commitTime: block.timestamp,
            revealTime: 0,
            isRevealed: false
        });
        
        emit ReviewCommitted(msg.sender, submissionId, commitHash);
    }

    /**
     * @dev Reveal a review for a submission (reveal phase)
     */
    function revealReview(uint256 submissionId, uint256 score, uint256 nonce) external onlyParticipant {
        require(submissionId < nextSubmissionId, "Invalid submission ID");
        require(score >= MIN_SCORE && score <= MAX_SCORE, "Invalid score");
        
        ReviewStructs.Review storage review = reviewCommits[msg.sender][submissionId];
        require(review.commitTime > 0, "Review not committed");
        require(!review.isRevealed, "Review already revealed");
        
        // Verify the reveal matches the commit
        bytes32 expectedCommit = keccak256(abi.encodePacked(submissionId, score, nonce));
        require(review.commitHash == expectedCommit, "Invalid reveal");
        
        review.score = score;
        review.revealTime = block.timestamp;
        review.isRevealed = true;
        
        // Update submission aggregated score (simple average)
        ReviewStructs.Submission storage submission = submissions[submissionId];
        submission.reviewCount++;
        submission.aggregatedScore = (submission.aggregatedScore * (submission.reviewCount - 1) + score) / submission.reviewCount;
        
        emit ReviewRevealed(msg.sender, submissionId, score);
    }

    /**
     * @dev Finalize contest and distribute rewards
     */
    function finalizeContest() external onlyOwner {
        require(!isFinalized, "Already finalized");
        require(submissionIds.length > 0, "No submissions to finalize");
        
        // Find winner (highest score)
        uint256 bestSubmissionId = 0;
        uint256 bestScore = 0;
        
        for (uint256 i = 0; i < submissionIds.length; i++) {
            uint256 submissionId = submissionIds[i];
            if (submissions[submissionId].aggregatedScore > bestScore) {
                bestScore = submissions[submissionId].aggregatedScore;
                bestSubmissionId = submissionId;
            }
        }
        
        // Distribute all rewards to winner
        if (bestSubmissionId > 0) {
            address winner = submissions[bestSubmissionId].participant;
            uint256 totalRewards = address(this).balance;
            participants[winner].totalRewards = totalRewards;
            
            emit ContestFinalized(winner, totalRewards);
        }
        
        isFinalized = true;
        contestInfo.status = ContestStatus.Finalized;
    }

    /**
     * @dev Generate review assignments for all participants using pseudo-random assignment
     * @notice Each participant will be assigned REVIEWS_PER_PARTICIPANT submissions to review
     */
    function generateReviewAssignments() external onlyOwner {
        require(!isFinalized, "Contest already finalized");
        require(!assignmentsGenerated, "Assignments already generated");
        require(submissionIds.length > 0, "No submissions to review");
        require(participantList.length > 1, "Need at least 2 participants");
        
        // Use block.prevrandao for pseudo-randomness (more secure than block.timestamp)
        uint256 randomSeed = block.prevrandao + block.timestamp;
        
        // Create a list of eligible submissions for each participant
        for (uint256 i = 0; i < participantList.length; i++) {
            address reviewer = participantList[i];
            uint256[] memory eligibleSubmissions = new uint256[](submissionIds.length);
            uint256 eligibleCount = 0;
            
            // Find all submissions that this participant didn't create and that are revealed
            for (uint256 j = 0; j < submissionIds.length; j++) {
                uint256 submissionId = submissionIds[j];
                if (submissions[submissionId].participant != reviewer && submissions[submissionId].isRevealed) {
                    eligibleSubmissions[eligibleCount] = submissionId;
                    eligibleCount++;
                }
            }
            
            // Determine how many reviews to assign (min of REVIEWS_PER_PARTICIPANT and eligible count)
            uint256 reviewsToAssign = eligibleCount < REVIEWS_PER_PARTICIPANT ? eligibleCount : REVIEWS_PER_PARTICIPANT;
            
            // Pseudo-randomly select submissions using Fisher-Yates shuffle
            for (uint256 k = 0; k < reviewsToAssign; k++) {
                // Generate pseudo-random index
                uint256 randomIndex = uint256(keccak256(abi.encodePacked(randomSeed, reviewer, k))) % (eligibleCount - k);
                uint256 selectedSubmissionId = eligibleSubmissions[randomIndex];
                
                // Assign the review
                reviewAssignments[reviewer].push(selectedSubmissionId);
                isAssignedReviewer[reviewer][selectedSubmissionId] = true;
                
                // Swap selected with last unselected (Fisher-Yates)
                eligibleSubmissions[randomIndex] = eligibleSubmissions[eligibleCount - k - 1];
            }
        }
        
        assignmentsGenerated = true;
        
        // Count total assignments for event
        uint256 totalAssignments = 0;
        for (uint256 i = 0; i < participantList.length; i++) {
            totalAssignments += reviewAssignments[participantList[i]].length;
        }
        
        emit ReviewAssignmentsGenerated(totalAssignments);
    }

    /**
     * @dev Get review assignments for a specific reviewer
     */
    function getReviewAssignmentsForReviewer(address reviewer) external view returns (uint256[] memory) {
        return reviewAssignments[reviewer];
    }

    /**
     * @dev Check if a reviewer is assigned to a specific submission
     */
    function isReviewerAssigned(address reviewer, uint256 submissionId) external view returns (bool) {
        return isAssignedReviewer[reviewer][submissionId];
    }

    /**
     * @dev Claim rewards for participant
     */
    function claimReward() external override nonReentrant {
        require(isFinalized, "Contest not finalized");
        Participant storage participant = participants[msg.sender];
        require(participant.totalRewards > 0, "No rewards to claim");
        
        uint256 rewardAmount = participant.totalRewards;
        participant.totalRewards = 0;
        
        payable(msg.sender).transfer(rewardAmount);
        
        emit RewardClaimed(msg.sender, rewardAmount);
    }

    /**
     * @dev Get submission details
     */
    function getSubmission(uint256 submissionId) external view returns (ReviewStructs.Submission memory) {
        return submissions[submissionId];
    }

    /**
     * @dev Get review score for a submission by reviewer (deprecated)
     */
    function getReview(address reviewer, uint256 submissionId) external view returns (uint256) {
        return reviews[reviewer][submissionId];
    }

    /**
     * @dev Get review details (commit-reveal) for a submission by reviewer
     */
    function getReviewCommit(address reviewer, uint256 submissionId) external view returns (ReviewStructs.Review memory) {
        return reviewCommits[reviewer][submissionId];
    }

    // Essential view functions
    function getContestInfo() external view override returns (ContestInfo memory) {
        return contestInfo;
    }

    function getParticipant(address account) external view override returns (Participant memory) {
        return participants[account];
    }

    function getParticipants() external view override returns (address[] memory) {
        return participantList;
    }

    function getCurrentEpoch() external view override returns (uint256) {
        return contestInfo.currentEpoch;
    }

    function getClaimableRewards(address participant) external view override returns (uint256) {
        return participants[participant].totalRewards;
    }

    function isParticipant(address account) external view override returns (bool) {
        return participants[account].isActive;
    }

    function isActive() external view override returns (bool) {
        return contestInfo.status == ContestStatus.Active;
    }

    // Admin functions
    function updateMetadata(string calldata newMetadataURI) external override onlyOwner {
        contestInfo.metadataURI = newMetadataURI;
    }

    function getRewardSplit() external view override returns (RewardSplit memory) {
        return contestInfo.rewardSplit;
    }

    function getScoringVerifier() external view returns (address) {
        return contestInfo.scoringVerifier;
    }

    function getCurrentPhase() external view returns (ReviewStructs.Phase) {
        // Simplified phase logic - in practice this would be more sophisticated
        uint256 elapsed = block.timestamp - contestInfo.startTime;
        uint256 fifth = contestInfo.duration / 5;
        
        if (elapsed < fifth) {
            return ReviewStructs.Phase.Commit;
        } else if (elapsed < fifth * 2) {
            return ReviewStructs.Phase.Reveal;
        } else if (elapsed < fifth * 3) {
            return ReviewStructs.Phase.ReviewCommit;
        } else if (elapsed < fifth * 4) {
            return ReviewStructs.Phase.ReviewReveal;
        } else {
            return ReviewStructs.Phase.Finalized;
        }
    }

    // Receive function to accept ETH for rewards
    receive() external payable {}
}
