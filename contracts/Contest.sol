// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "./interfaces/IContest.sol";
import "./interfaces/IContestManager.sol";
import "./types/ReviewStructs.sol";

/**
 * @title Contest
 * @notice Manages decentralized ML model competitions with commit-reveal submission and peer review
 * @dev Implements a 4-phase lifecycle: Commit → Reveal → ReviewCommit → ReviewReveal → Finalized
 * Uses minimal proxy pattern via clone() - state is set through initialize()
 */
contract Contest is IContest, Ownable, ReentrancyGuard {
    ContestInfo public contestInfo;
    
    address public contestManager;
    address public infToken;
    
    uint256 public nextSubmissionId = 1;
    bool public isFinalized;
    
    // Contest configuration - set during initialization
    uint256 public minScore = 0;
    uint256 public maxScore = 1e6; // Max score for cosine similarity (1.0 * 1e6)
    uint256 public reviewCount; // Number of reviews required per output (k)
    
    // Reviewer incentive parameters (in 1e6 scale for precision)
    uint256 public epsilonReward; // Reward threshold for average error (e.g., 100000 = 10%)
    uint256 public epsilonSlash; // Slashing threshold for average error (e.g., 200000 = 20%)
    uint256 public alpha; // Reputation update coefficient (e.g., 1000 = 1.0)
    uint256 public beta; // Slashing coefficient (e.g., 100 = 10%)
    uint256 public gamma; // Reward coefficient (e.g., 50 = 5%)
    
    uint256 public maxParticipants; // Maximum number of participants allowed (0 = unlimited)
    uint256 public minStakeAmount; // Minimum stake required to join
    uint256 public joinPriceAdjustment; // Basis points (100 = 1%) - price increase rate per join
    uint256 public currentJoinPrice; // Current price to join (starts at minStakeAmount)
    
    mapping(address => Participant) public participants;
    mapping(uint256 => ReviewStructs.Submission) public submissions;
    mapping(address => uint256[]) public participantSubmissions;
    
    // Commit-reveal peer-review storage
    mapping(address => mapping(uint256 => ReviewStructs.Review)) public reviewCommits; // reviewer => outputId => Review
    mapping(uint256 => ReviewStructs.Review[]) public reviewsByOutput; // outputId => Review[]
    mapping(address => uint256[]) public reviewedOutputsByReviewer; // reviewer => outputIds[]
    
    // Reviewer trust and reputation system
    mapping(address => uint256) public trustScore; // trust score (weight) of reviewer [1e6 scale]
    mapping(address => uint256) public reputation; // reputation score of reviewer
    mapping(address => bool) public reviewerSlashed; // reviewer => is slashed
    mapping(address => uint256) public reviewerRewards; // accumulated rewards for honest reviewing
    
    mapping(address => uint256[]) public reviewAssignments;
    mapping(address => mapping(uint256 => bool)) public isAssignedReviewer;
    
    address[] public participantList;
    uint256[] public submissionIds;
    bool public assignmentsGenerated;
    bool public paused;
    
    modifier onlyParticipant() {
        require(participants[msg.sender].isActive, "Not an active participant");
        _;
    }
    
    modifier whenNotPaused() {
        require(!paused, "Contest is paused");
        _;
    }

    modifier onlyContestManager() {
        require(msg.sender == contestManager, "Only contest manager");
        _;
    }

    constructor() Ownable(msg.sender) {}

    /**
     * @notice Initialize contest (called once by factory after clone)
     * @dev Replaces constructor for minimal proxy pattern
     * @param creator Contest owner who can finalize and manage
     * @param metadataURI IPFS URI with contest rules and dataset info
     * @param duration Total contest duration in seconds (divided into 4 equal phases)
     * @param _contestManager Address of ContestManager for stake/reward handling
     * @param _infToken INF token address for staking and rewards
     * @param params Initialization parameters struct containing all configuration
     */
    function initialize(
        address creator,
        string calldata metadataURI,
        uint256 duration,
        address _contestManager,
        address _infToken,
        IContest.InitParams calldata params
    ) external override {
        require(contestInfo.creator == address(0), "Already initialized");
        require(_contestManager != address(0), "Invalid contest manager");
        require(_infToken != address(0), "Invalid INF token");
        require(duration > 0, "Invalid duration");
        require(params.reviewCount > 0, "Review count must be > 0");
        require(params.epsilonReward < params.epsilonSlash, "Reward threshold must be less than slash threshold");
        require(params.epsilonSlash <= 1e6, "Slash threshold too high");
        require(params.beta <= 1000, "Beta too high (max 100%)");
        require(params.gamma <= 1000, "Gamma too high (max 100%)");
        require(params.minStakeAmount > 0, "Min stake must be > 0");
        
        contestManager = _contestManager;
        infToken = _infToken;
        
        contestInfo = ContestInfo({
            creator: creator,
            metadataURI: metadataURI,
            startTime: block.timestamp,
            duration: duration,
            status: ContestStatus.Active
        });
        
        // Set peer-review configuration
        reviewCount = params.reviewCount;
        epsilonReward = params.epsilonReward;
        epsilonSlash = params.epsilonSlash;
        alpha = params.alpha;
        beta = params.beta;
        gamma = params.gamma;
        
        // Set contest participation parameters
        minStakeAmount = params.minStakeAmount;
        currentJoinPrice = params.minStakeAmount;
        joinPriceAdjustment = params.joinPriceAdjustment;
        maxParticipants = params.maxParticipants;
        minScore = 0;
        maxScore = 1e6;
        
        nextSubmissionId = 1;
        _transferOwnership(creator);
        
        emit ContestInitialized(creator, metadataURI);
    }

    /**
     * @notice Join contest by staking INF tokens
     * @dev Transfers stake to ContestManager, not this contract. Uses CEI pattern for reentrancy safety
     * Enforces participant limit and dynamic pricing that increases with each join
     * @param stakeAmount Amount of INF tokens to stake (must be >= currentJoinPrice)
     */
    function joinContest(uint256 stakeAmount) external override whenNotPaused {
        require(!participants[msg.sender].isActive, "Already joined");
        require(contestInfo.status == ContestStatus.Active, "Contest not active");
        
        // Check participant limit
        if (maxParticipants > 0) {
            require(participantList.length < maxParticipants, "Contest is full");
        }
        
        // Check minimum stake requirement (dynamic price)
        require(stakeAmount >= currentJoinPrice, "Stake amount below current join price");
        require(stakeAmount >= minStakeAmount, "Stake amount below minimum");
        
        // Check-Effects-Interactions pattern for reentrancy protection
        participants[msg.sender] = Participant({
            account: msg.sender,
            joinedAt: block.timestamp,
            totalRewards: 0,
            lastSubmissionTime: 0,
            isActive: true,
            stakedAmount: stakeAmount
        });
        
        participantList.push(msg.sender);
        
        // Initialize trust score and reputation for new participant
        trustScore[msg.sender] = 1e6; // Start with trust score of 1.0
        reputation[msg.sender] = 0; // Start with neutral reputation
        
        // Update join price for next participant
        // newPrice = curPrice + curPrice * joinPriceAdjustment / 10000
        if (joinPriceAdjustment > 0) {
            currentJoinPrice = currentJoinPrice + (currentJoinPrice * joinPriceAdjustment / 10000);
        }
        
        // External interactions last
        IERC20(infToken).transferFrom(msg.sender, contestManager, stakeAmount);
        IContestManager(contestManager).recordStake(msg.sender, stakeAmount);
        
        emit ParticipantJoined(msg.sender, block.timestamp);
    }

    /**
     * @notice Leave contest and get refunded the staked amount
     * @dev Can only leave during Commit phase and before making any submissions
     * Stake is returned to the participant from ContestManager
     */
    function leaveContest() external whenNotPaused {
        require(participants[msg.sender].isActive, "Not a participant");
        require(!isFinalized, "Contest already finalized");
        require(getCurrentPhase() == ReviewStructs.Phase.Commit, "Cannot leave after commit phase");
        require(participantSubmissions[msg.sender].length == 0, "Cannot leave after submitting");
        
        uint256 refundAmount = participants[msg.sender].stakedAmount;
        
        participants[msg.sender].isActive = false;
        participants[msg.sender].stakedAmount = 0;
        
        // Remove from participant list - optimized
        uint256 length = participantList.length;
        for (uint256 i = 0; i < length; i++) {
            if (participantList[i] == msg.sender) {
                participantList[i] = participantList[length - 1];
                participantList.pop();
                break;
            }
        }
        
        // Refund stake through ContestManager
        IContestManager(contestManager).refundStake(address(this), msg.sender, refundAmount);
        
        emit ParticipantLeft(msg.sender);
    }

    /**
     * @notice Submit model prediction commitment during Commit phase
     * @dev Hash should be keccak256(ipfsURI, outputHash, nonce) to prevent front-running
     * @param commitHash Commitment hash to be revealed in next phase
     */
    function commitSubmission(bytes32 commitHash) external onlyParticipant whenNotPaused {
        require(getCurrentPhase() == ReviewStructs.Phase.Commit, "Invalid phase");
        
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
     * @notice Reveal previously committed submission during Reveal phase
     * @dev Validates commitment matches keccak256(ipfsURI, outputHash, nonce)
     * @param submissionId ID of the committed submission
     * @param ipfsURI IPFS hash pointing to model predictions
     * @param outputHash Hash of the prediction outputs
     * @param nonce Random value used in commitment
     */
    function revealSubmission(
        uint256 submissionId,
        string calldata ipfsURI,
        bytes32 outputHash,
        uint256 nonce
    ) external onlyParticipant whenNotPaused {
        require(getCurrentPhase() == ReviewStructs.Phase.Reveal, "Invalid phase");
        
        ReviewStructs.Submission storage submission = submissions[submissionId];
        require(submission.participant != address(0), "Submission does not exist");
        require(submission.participant == msg.sender, "Not your submission");
        require(!submission.isRevealed, "Already revealed");
        
        bytes32 expectedCommit = keccak256(abi.encodePacked(ipfsURI, outputHash, nonce));
        require(submission.commitHash == expectedCommit, "Invalid reveal");
        
        submission.ipfsURI = ipfsURI;
        submission.outputHash = outputHash;
        submission.revealTime = block.timestamp;
        submission.isRevealed = true;
        
        emit SubmissionSubmitted(msg.sender, submissionId, ipfsURI, outputHash);
    }

    /**
     * @notice Commit a review for an output during ReviewCommit phase
     * @dev Commitment hash should be keccak256(abi.encodePacked(outputId, score, nonce))
     * This prevents front-running and ensures reviewers commit to their scores before seeing others
     * @param outputId ID of output/submission to review
     * @param commitHash Hash of (outputId, score, nonce) - must match in reveal phase
     */
    function commitReview(uint256 outputId, bytes32 commitHash) external onlyParticipant whenNotPaused {
        require(getCurrentPhase() == ReviewStructs.Phase.ReviewCommit, "Invalid phase");
        require(submissions[outputId].participant != address(0), "Output does not exist");
        require(submissions[outputId].isRevealed, "Output not revealed yet");
        require(submissions[outputId].participant != msg.sender, "Cannot review own output");
        require(reviewCommits[msg.sender][outputId].commitTime == 0, "Review already committed");
        require(isAssignedReviewer[msg.sender][outputId], "Not assigned to review this output");
        
        reviewCommits[msg.sender][outputId] = ReviewStructs.Review({
            reviewer: msg.sender,
            outputId: outputId,
            commitHash: commitHash,
            score: 0,
            commitTime: block.timestamp,
            revealTime: 0,
            isRevealed: false
        });
        
        emit ReviewCommitted(msg.sender, outputId, commitHash);
    }

    /**
     * @notice Reveal previously committed review score during ReviewReveal phase
     * @dev Updates output's running average and stores score for median calculation
     * @param outputId ID of output being reviewed
     * @param score Review score [0, 1e6] (cosine similarity * 1e6)
     * @param nonce Random value used in commitment
     */
    function revealReview(uint256 outputId, uint256 score, uint256 nonce) external onlyParticipant whenNotPaused {
        require(getCurrentPhase() == ReviewStructs.Phase.ReviewReveal, "Invalid phase");
        require(submissions[outputId].participant != address(0), "Output does not exist");
        require(score >= minScore && score <= maxScore, "Invalid score");
        
        ReviewStructs.Review storage review = reviewCommits[msg.sender][outputId];
        require(review.commitTime > 0, "Review not committed");
        require(!review.isRevealed, "Review already revealed");
        
        bytes32 expectedCommit = keccak256(abi.encodePacked(outputId, score, nonce));
        require(review.commitHash == expectedCommit, "Invalid reveal");
        
        review.score = score;
        review.revealTime = block.timestamp;
        review.isRevealed = true;
        
        // Store revealed review for aggregation
        reviewsByOutput[outputId].push(review);
        reviewedOutputsByReviewer[msg.sender].push(outputId);
        
        // Update submission review count
        submissions[outputId].reviewCount++;
        
        emit ReviewRevealed(msg.sender, outputId, score);
    }

    /**
     * @notice Finalize contest and distribute entire reward pool to winner
     * @dev Winner is determined by highest aggregated review score. Can only be called once
     */
    function finalizeContest() external onlyOwner {
        require(!isFinalized, "Already finalized");
        require(submissionIds.length > 0, "No submissions to finalize");
        
        // Find submission with highest aggregated score
        uint256 bestSubmissionId = 0;
        uint256 bestScore = 0;
        
        for (uint256 i = 0; i < submissionIds.length; i++) {
            uint256 submissionId = submissionIds[i];
            if (submissions[submissionId].aggregatedScore > bestScore) {
                bestScore = submissions[submissionId].aggregatedScore;
                bestSubmissionId = submissionId;
            }
        }
        
        if (bestSubmissionId > 0) {
            address winner = submissions[bestSubmissionId].participant;
            
            address[] memory winners = new address[](1);
            winners[0] = winner;
            uint256[] memory amounts = new uint256[](1);
            amounts[0] = IContestManager(contestManager).getRewardPool(address(this));
            
            participants[winner].totalRewards = amounts[0];
            
            IContestManager(contestManager).distributeRewards(address(this), winners, amounts);
            
            emit ContestFinalized(winner, amounts[0]);
        }
        
        isFinalized = true;
        contestInfo.status = ContestStatus.Finalized;
    }

    /**
     * @notice Pseudo-randomly assign peer reviews to participants
     * @dev Uses blockhash + prevrandao for randomness (not perfect, consider Chainlink VRF for production)
     * Each participant reviews up to reviewsPerParticipant submissions (excluding their own)
     * QC address gets same number of assignments if set
     */
    function generateReviewAssignments() external onlyOwner {
        require(!isFinalized, "Contest already finalized");
        require(!assignmentsGenerated, "Assignments already generated");
        require(submissionIds.length > 0, "No submissions to review");
        require(participantList.length > 1, "Need at least 2 participants");
        require(getCurrentPhase() == ReviewStructs.Phase.ReviewCommit || getCurrentPhase() == ReviewStructs.Phase.Reveal, "Invalid phase for assignment generation");
        
        // Generate pseudo-random seed from multiple sources
        uint256 randomSeed = uint256(keccak256(abi.encodePacked(
            block.prevrandao,
            block.timestamp,
            blockhash(block.number - 1),
            participantList.length,
            submissionIds.length
        )));
        
        // Assign reviews to regular participants
        for (uint256 i = 0; i < participantList.length; i++) {
            address reviewer = participantList[i];
            uint256[] memory eligibleSubmissions = new uint256[](submissionIds.length);
            uint256 eligibleCount = 0;
            
            for (uint256 j = 0; j < submissionIds.length; j++) {
                uint256 submissionId = submissionIds[j];
                if (submissions[submissionId].participant != reviewer && submissions[submissionId].isRevealed) {
                    eligibleSubmissions[eligibleCount] = submissionId;
                    eligibleCount++;
                }
            }
            
            uint256 reviewsToAssign = eligibleCount < reviewsPerParticipant ? eligibleCount : reviewsPerParticipant;
            
            for (uint256 k = 0; k < reviewsToAssign; k++) {
                uint256 randomIndex = uint256(keccak256(abi.encodePacked(randomSeed, reviewer, k))) % (eligibleCount - k);
                uint256 selectedSubmissionId = eligibleSubmissions[randomIndex];
                
                reviewAssignments[reviewer].push(selectedSubmissionId);
                isAssignedReviewer[reviewer][selectedSubmissionId] = true;
                
                eligibleSubmissions[randomIndex] = eligibleSubmissions[eligibleCount - k - 1];
            }
        }
        
        assignmentsGenerated = true;
        
        uint256 totalAssignments = 0;
        for (uint256 i = 0; i < participantList.length; i++) {
            totalAssignments += reviewAssignments[participantList[i]].length;
        }
        
        emit ReviewAssignmentsGenerated(totalAssignments);
    }


    /**
     * @notice Aggregate reviews for a specific output by calculating weighted consensus score
     * @dev Calculates weighted average using trust scores as weights
     * @param outputId Output/submission ID to aggregate reviews for
     */
    function aggregateReviews(uint256 outputId) public {
        require(submissions[outputId].participant != address(0), "Output does not exist");
        require(reviewsByOutput[outputId].length >= reviewCount, "Not enough reviews yet");
        
        ReviewStructs.Review[] memory reviews = reviewsByOutput[outputId];
        
        uint256 weightedSum = 0;
        uint256 totalWeight = 0;
        
        // Calculate weighted average using trust scores
        for (uint256 i = 0; i < reviews.length; i++) {
            address reviewer = reviews[i].reviewer;
            uint256 score = reviews[i].score;
            uint256 weight = trustScore[reviewer];
            
            weightedSum += weight * score;
            totalWeight += weight;
        }
        
        require(totalWeight > 0, "No valid trust weights");
        
        uint256 consensus = weightedSum / totalWeight;
        submissions[outputId].aggregatedScore = consensus;
        
        emit ReviewsAggregated(outputId, consensus, reviews.length);
    }
    
    /**
     * @notice Aggregate reviews for all outputs in the contest
     * @dev Batch operation to calculate median for all outputs that have enough reviews
     * @param contestId Contest ID (should match current contest)
     */
    function aggregateReviews(uint256 contestId) external {
        require(contestId == uint256(uint160(address(this))), "Invalid contest ID");
        
        for (uint256 i = 1; i < nextSubmissionId; i++) {
            if (scoresByOutput[i].length >= reviewCount) {
                aggregateReviews(i);
            }
        }
    }
    
    /**
     * @notice Evaluate a reviewer's performance and apply slashing/rewards
     * @dev Implements trust-reputation system with following steps:
     * - Calculate error per review as absolute difference from consensus
     * - Update accumulated error and review count
     * - Calculate average error across all reviews
     * - Update trust score based on average error
     * - Update reputation with threshold-based rewards/penalties
     * - Apply slashing if error exceeds threshold
     * - Apply rewards if error is below reward threshold
     * @param reviewer Address of reviewer to evaluate
     */
    function evaluateReviewer(address reviewer) external {
        require(participants[reviewer].isActive || reviewedOutputsByReviewer[reviewer].length > 0, "Not a reviewer");
        
        uint256[] memory outputIds = reviewedOutputsByReviewer[reviewer];
        require(outputIds.length > 0, "No reviews submitted");
        
        uint256 totalError = 0;
        uint256 reviewsCompleted = 0;
        
        // Process all reviews by this reviewer to calculate errors
        for (uint256 i = 0; i < outputIds.length; i++) {
            uint256 outputId = outputIds[i];
            ReviewStructs.Review storage review = reviewCommits[reviewer][outputId];
            
            // Only count revealed reviews for outputs that have been aggregated
            if (review.isRevealed) {
                // Ensure consensus is calculated
                uint256 consensus = submissions[outputId].aggregatedScore;
                if (consensus == 0 && reviewsByOutput[outputId].length >= reviewCount) {
                    aggregateReviews(outputId);
                    consensus = submissions[outputId].aggregatedScore;
                }
                
                if (consensus > 0) {
                    // Calculate absolute error from consensus
                    uint256 error;
                    if (review.score > consensus) {
                        error = review.score - consensus;
                    } else {
                        error = consensus - review.score;
                    }
                    
                    totalError += error;
                    reviewsCompleted++;
                }
            }
        }
        
        require(reviewsCompleted > 0, "No valid reviews to evaluate");
        
        // Calculate average error
        uint256 avgError = totalError / reviewsCompleted;
        
        // Update trust score using inverse relationship with error
        trustScore[reviewer] = (1e12) / (1e6 + avgError);
        
        // Update reputation based on performance thresholds
        int256 reputationDelta = 0;
        if (avgError < epsilonReward) {
            reputationDelta = int256(alpha);
        } else if (avgError > epsilonSlash) {
            reputationDelta = -int256(alpha);
        }
        
        // Apply reputation update (careful with underflow)
        if (reputationDelta >= 0) {
            reputation[reviewer] += uint256(reputationDelta);
        } else {
            uint256 decrease = uint256(-reputationDelta);
            if (reputation[reviewer] >= decrease) {
                reputation[reviewer] -= decrease;
            } else {
                reputation[reviewer] = 0;
            }
        }
        
        // Apply slashing if average error exceeds threshold
        if (avgError > epsilonSlash && !reviewerSlashed[reviewer]) {
            uint256 stakeAmount = participants[reviewer].stakedAmount;
            uint256 slashAmount = (stakeAmount * beta) / 1000;
            
            if (slashAmount > 0 && stakeAmount >= slashAmount) {
                participants[reviewer].stakedAmount -= slashAmount;
                reviewerSlashed[reviewer] = true;
                
                IContestManager(contestManager).slashParticipant(
                    address(this),
                    reviewer,
                    slashAmount,
                    "Average error exceeds slash threshold"
                );
                
                emit ReviewerSlashed(reviewer, 0, slashAmount, avgError);
            }
        }
        
        // Apply reward if average error is below reward threshold
        if (avgError < epsilonReward) {
            uint256 stakeAmount = participants[reviewer].stakedAmount;
            uint256 rewardAmount = (stakeAmount * gamma) / 1000;
            
            if (rewardAmount > 0) {
                reviewerRewards[reviewer] += rewardAmount;
                emit ReviewerRewarded(reviewer, rewardAmount, avgError);
            }
        }
        
        emit ReviewerEvaluated(reviewer, avgError, reviewsCompleted);
        emit TrustScoreUpdated(reviewer, trustScore[reviewer]);
        emit ReputationUpdated(reviewer, reputation[reviewer]);
    }

    function getReviewAssignmentsForReviewer(address reviewer) external view override returns (uint256[] memory) {
        return reviewAssignments[reviewer];
    }

    function isReviewerAssigned(address reviewer, uint256 submissionId) external view override returns (bool) {
        return isAssignedReviewer[reviewer][submissionId];
    }
    
    /**
     * @notice Claim accumulated rewards for honest reviewing
     * @dev Transfers accumulated rewards from contest reward pool to reviewer
     */
    function claimReviewerRewards() external nonReentrant {
        require(reviewerRewards[msg.sender] > 0, "No rewards to claim");
        
        uint256 rewardAmount = reviewerRewards[msg.sender];
        reviewerRewards[msg.sender] = 0;
        
        // Transfer rewards from contest reward pool
        IERC20(infToken).transfer(msg.sender, rewardAmount);
        
        emit ReviewerRewardsClaimed(msg.sender, rewardAmount);
    }

    /**
     * @notice Claim rewards after contest finalization
     * @dev Uses reentrancy guard and CEI pattern
     */
    function claimReward() external override nonReentrant {
        require(isFinalized, "Contest not finalized");
        Participant storage participant = participants[msg.sender];
        require(participant.totalRewards > 0, "No rewards to claim");
        
        uint256 rewardAmount = participant.totalRewards;
        participant.totalRewards = 0;
        
        IERC20(infToken).transfer(msg.sender, rewardAmount);
        
        emit RewardClaimed(msg.sender, rewardAmount);
    }

    function getSubmission(uint256 submissionId) external view returns (ReviewStructs.Submission memory) {
        return submissions[submissionId];
    }

    function getReviewCommit(address reviewer, uint256 submissionId) 
        external 
        view 
        returns (ReviewStructs.Review memory) 
    {
        return reviewCommits[reviewer][submissionId];
    }

    function getContestInfo() external view override returns (ContestInfo memory) {
        return contestInfo;
    }

    function getParticipant(address account) external view override returns (Participant memory) {
        return participants[account];
    }

    function getParticipants() external view override returns (address[] memory) {
        return participantList;
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

    function updateMetadata(string calldata newMetadataURI) external override onlyOwner {
        string memory oldMetadataURI = contestInfo.metadataURI;
        contestInfo.metadataURI = newMetadataURI;
        
        emit MetadataUpdated(oldMetadataURI, newMetadataURI);
    }
    
    /**
     * @notice Emergency pause to stop all participant actions
     * @dev Blocks join, submit, reveal, and review operations. Required before emergencyWithdraw
     */
    function pauseContest() external onlyOwner {
        require(!paused, "Already paused");
        require(!isFinalized, "Contest already finalized");
        
        paused = true;
        contestInfo.status = ContestStatus.Paused;
        
        emit ContestPaused();
    }
    
    function unpauseContest() external onlyOwner {
        require(paused, "Not paused");
        
        paused = false;
        contestInfo.status = ContestStatus.Active;
        
        emit ContestUnpaused();
    }
    
    /**
     * @notice Withdraw any tokens stuck in this contract (emergency only)
     * @dev Requires contest to be paused first. Use if tokens are accidentally sent here
     * Normal flow sends stakes to ContestManager, not here
     */
    function emergencyWithdraw() external onlyOwner {
        require(paused, "Contest must be paused first");
        
        uint256 balance = IERC20(infToken).balanceOf(address(this));
        if (balance > 0) {
            IERC20(infToken).transfer(owner(), balance);
        }
        
        emit EmergencyWithdraw(owner(), balance);
    }

    
    event ReviewsAggregated(uint256 indexed outputId, uint256 consensusScore, uint256 reviewCount);
    event ReviewerEvaluated(address indexed reviewer, uint256 averageError, uint256 reviewCount);
    event ReviewerSlashed(address indexed reviewer, uint256 indexed submissionId, uint256 amount, uint256 averageError);
    event ReviewerRewarded(address indexed reviewer, uint256 rewardAmount, uint256 averageError);
    event ReviewerRewardsClaimed(address indexed reviewer, uint256 amount);
    event TrustScoreUpdated(address indexed reviewer, uint256 newTrustScore);
    event ReputationUpdated(address indexed reviewer, uint256 newReputation);
    event ParticipantLeft(address indexed participant);
    event MetadataUpdated(string oldMetadataURI, string newMetadataURI);
    event ContestPaused();
    event ContestUnpaused();
    event EmergencyWithdraw(address indexed owner, uint256 amount);

    /**
     * @notice Update contest parameters (owner only)
     * @dev Can only be called before contest starts accepting submissions
     * @param _maxParticipants Maximum participants (0 = unlimited)
     * @param _minStakeAmount Minimum stake amount required
     * @param _joinPriceAdjustment Join price increase rate in basis points (100 = 1%)
     * @param _reviewCount Number of reviews required per output
     * @param _epsilonReward Reward threshold for average error
     * @param _epsilonSlash Slashing threshold for average error
     * @param _alpha Reputation update coefficient
     * @param _beta Slashing coefficient
     * @param _gamma Reward coefficient
     */
    function updateContestParameters(
        uint256 _maxParticipants,
        uint256 _minStakeAmount,
        uint256 _joinPriceAdjustment,
        uint256 _reviewCount,
        uint256 _epsilonReward,
        uint256 _epsilonSlash,
        uint256 _alpha,
        uint256 _beta,
        uint256 _gamma
    ) external onlyOwner {
        require(submissionIds.length == 0, "Cannot update after submissions started");
        require(_reviewCount > 0, "Review count must be > 0");
        require(_epsilonReward < _epsilonSlash, "Reward threshold must be less than slash threshold");
        require(_epsilonSlash <= 1e6, "Slash threshold too high");
        require(_beta <= 1000, "Beta too high (max 100%)");
        require(_gamma <= 1000, "Gamma too high (max 100%)");
        
        maxParticipants = _maxParticipants;
        minStakeAmount = _minStakeAmount;
        joinPriceAdjustment = _joinPriceAdjustment;
        reviewCount = _reviewCount;
        epsilonReward = _epsilonReward;
        epsilonSlash = _epsilonSlash;
        alpha = _alpha;
        beta = _beta;
        gamma = _gamma;
        
        // Reset current join price if min stake changed
        if (participantList.length == 0) {
            currentJoinPrice = _minStakeAmount;
        }
        
        emit ContestParametersUpdated(_maxParticipants, _minStakeAmount, _reviewCount, _epsilonReward, _epsilonSlash);
    }
    
    event ContestParametersUpdated(
        uint256 maxParticipants, 
        uint256 minStakeAmount, 
        uint256 reviewCount,
        uint256 epsilonReward,
        uint256 epsilonSlash
    );

    /**
     * @notice Calculate current contest phase based on elapsed time
     * @dev Divides duration into 4 equal quarters: Commit → Reveal → ReviewCommit → ReviewReveal
     * After contest duration expires, it enters Finalized phase automatically
     * Note: Contests are currently single-cycle (non-recurring)
     * @return Current phase enum
     */
    function getCurrentPhase() public view returns (ReviewStructs.Phase) {
        if (isFinalized) {
            return ReviewStructs.Phase.Finalized;
        }
        
        uint256 elapsed = block.timestamp - contestInfo.startTime;
        uint256 quarter = contestInfo.duration / 4;
        
        if (elapsed < quarter) {
            return ReviewStructs.Phase.Commit;
        } else if (elapsed < quarter * 2) {
            return ReviewStructs.Phase.Reveal;
        } else if (elapsed < quarter * 3) {
            return ReviewStructs.Phase.ReviewCommit;
        } else if (elapsed < contestInfo.duration) {
            return ReviewStructs.Phase.ReviewReveal;
        } else {
            return ReviewStructs.Phase.Finalized;
        }
    }
}
