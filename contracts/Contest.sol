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
    
    uint256 public constant MIN_SCORE = 0;
    uint256 public constant MAX_SCORE = 100;
    uint256 public constant REVIEWS_PER_PARTICIPANT = 5; // Max reviews assigned per participant
    
    mapping(address => Participant) public participants;
    mapping(uint256 => ReviewStructs.Submission) public submissions;
    mapping(address => uint256[]) public participantSubmissions;
    mapping(address => mapping(uint256 => ReviewStructs.Review)) public reviewCommits;
    mapping(address => uint256[]) public reviewAssignments;
    mapping(address => mapping(uint256 => bool)) public isAssignedReviewer;
    
    address[] public participantList;
    uint256[] public submissionIds;
    bool public assignmentsGenerated;
    bool public paused;
    
    modifier onlyParticipant() {
        require(participants[msg.sender].isActive || msg.sender == contestInfo.qcAddress, "Not an active participant or QC");
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

    modifier onlyQC() {
        require(msg.sender == contestInfo.qcAddress, "Only QC");
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
     * @param _qcAddress Optional quality control address with special privileges
     */
    function initialize(
        address creator,
        string calldata metadataURI,
        uint256 duration,
        address _contestManager,
        address _infToken,
        address _qcAddress
    ) external override {
        require(contestInfo.creator == address(0), "Already initialized");
        require(_contestManager != address(0), "Invalid contest manager");
        require(_infToken != address(0), "Invalid INF token");
        require(duration > 0, "Invalid duration");
        
        contestManager = _contestManager;
        infToken = _infToken;
        
        contestInfo = ContestInfo({
            creator: creator,
            metadataURI: metadataURI,
            startTime: block.timestamp,
            duration: duration,
            status: ContestStatus.Active,
            qcAddress: _qcAddress
        });
        
        nextSubmissionId = 1;
        _transferOwnership(creator);
        
        emit ContestInitialized(creator, metadataURI);
    }

    /**
     * @notice Join contest by staking INF tokens
     * @dev Transfers stake to ContestManager, not this contract. Uses CEI pattern for reentrancy safety
     * @param stakeAmount Amount of INF tokens to stake
     */
    function joinContest(uint256 stakeAmount) external override whenNotPaused {
        require(!participants[msg.sender].isActive, "Already joined");
        require(stakeAmount > 0, "Stake amount must be greater than 0");
        require(contestInfo.status == ContestStatus.Active, "Contest not active");
        
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
        
        // External interactions last
        IERC20(infToken).transferFrom(msg.sender, contestManager, stakeAmount);
        IContestManager(contestManager).recordStake(msg.sender, stakeAmount);
        
        emit ParticipantJoined(msg.sender, block.timestamp);
    }

    function leaveContest() external whenNotPaused {
        require(participants[msg.sender].isActive, "Not a participant");
        require(!isFinalized, "Contest already finalized");
        require(getCurrentPhase() == ReviewStructs.Phase.Commit, "Cannot leave after commit phase");
        require(participantSubmissions[msg.sender].length == 0, "Cannot leave after submitting");
        
        participants[msg.sender].isActive = false;
        
        // Remove from participant list - optimized
        uint256 length = participantList.length;
        for (uint256 i = 0; i < length; i++) {
            if (participantList[i] == msg.sender) {
                participantList[i] = participantList[length - 1];
                participantList.pop();
                break;
            }
        }
        
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
        require(submissionId < nextSubmissionId, "Invalid submission ID");
        
        ReviewStructs.Submission storage submission = submissions[submissionId];
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

    function commitReview(uint256 submissionId, bytes32 commitHash) external onlyParticipant whenNotPaused {
        require(getCurrentPhase() == ReviewStructs.Phase.ReviewCommit, "Invalid phase");
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
     * @notice Reveal previously committed review score
     * @dev Updates submission's running average score using safe math to prevent overflow
     * @param submissionId ID of submission being reviewed
     * @param score Review score (0-100)
     * @param nonce Random value used in commitment
     */
    function revealReview(uint256 submissionId, uint256 score, uint256 nonce) external onlyParticipant whenNotPaused {
        require(getCurrentPhase() == ReviewStructs.Phase.ReviewReveal, "Invalid phase");
        require(submissionId < nextSubmissionId, "Invalid submission ID");
        require(score >= MIN_SCORE && score <= MAX_SCORE, "Invalid score");
        
        ReviewStructs.Review storage review = reviewCommits[msg.sender][submissionId];
        require(review.commitTime > 0, "Review not committed");
        require(!review.isRevealed, "Review already revealed");
        
        bytes32 expectedCommit = keccak256(abi.encodePacked(submissionId, score, nonce));
        require(review.commitHash == expectedCommit, "Invalid reveal");
        
        review.score = score;
        review.revealTime = block.timestamp;
        review.isRevealed = true;
        
        ReviewStructs.Submission storage submission = submissions[submissionId];
        
        // Calculate running average safely to prevent overflow
        // Formula: new_avg = (old_avg * count + new_score) / (count + 1)
        if (submission.reviewCount == 0) {
            submission.aggregatedScore = score;
        } else {
            uint256 totalScore = submission.aggregatedScore * submission.reviewCount;
            submission.aggregatedScore = (totalScore + score) / (submission.reviewCount + 1);
        }
        submission.reviewCount++;
        
        emit ReviewRevealed(msg.sender, submissionId, score);
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
     * Each participant reviews up to REVIEWS_PER_PARTICIPANT submissions (excluding their own)
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
            
            uint256 reviewsToAssign = eligibleCount < REVIEWS_PER_PARTICIPANT ? eligibleCount : REVIEWS_PER_PARTICIPANT;
            
            for (uint256 k = 0; k < reviewsToAssign; k++) {
                uint256 randomIndex = uint256(keccak256(abi.encodePacked(randomSeed, reviewer, k))) % (eligibleCount - k);
                uint256 selectedSubmissionId = eligibleSubmissions[randomIndex];
                
                reviewAssignments[reviewer].push(selectedSubmissionId);
                isAssignedReviewer[reviewer][selectedSubmissionId] = true;
                
                eligibleSubmissions[randomIndex] = eligibleSubmissions[eligibleCount - k - 1];
            }
        }
        
        // Assign reviews to QC if set
        if (contestInfo.qcAddress != address(0)) {
            uint256[] memory qcEligibleSubmissions = new uint256[](submissionIds.length);
            uint256 qcEligibleCount = 0;
            
            for (uint256 j = 0; j < submissionIds.length; j++) {
                if (submissions[submissionIds[j]].isRevealed) {
                    qcEligibleSubmissions[qcEligibleCount] = submissionIds[j];
                    qcEligibleCount++;
                }
            }
            
            uint256 qcReviewsToAssign = qcEligibleCount < REVIEWS_PER_PARTICIPANT ? qcEligibleCount : REVIEWS_PER_PARTICIPANT;
            
            for (uint256 k = 0; k < qcReviewsToAssign; k++) {
                uint256 randomIndex = uint256(keccak256(abi.encodePacked(randomSeed, contestInfo.qcAddress, k))) % (qcEligibleCount - k);
                uint256 selectedSubmissionId = qcEligibleSubmissions[randomIndex];
                
                reviewAssignments[contestInfo.qcAddress].push(selectedSubmissionId);
                isAssignedReviewer[contestInfo.qcAddress][selectedSubmissionId] = true;
                
                qcEligibleSubmissions[randomIndex] = qcEligibleSubmissions[qcEligibleCount - k - 1];
            }
        }
        
        assignmentsGenerated = true;
        
        uint256 totalAssignments = 0;
        for (uint256 i = 0; i < participantList.length; i++) {
            totalAssignments += reviewAssignments[participantList[i]].length;
        }
        if (contestInfo.qcAddress != address(0)) {
            totalAssignments += reviewAssignments[contestInfo.qcAddress].length;
        }
        
        emit ReviewAssignmentsGenerated(totalAssignments);
    }

    /**
     * @notice Slash participant's stake for misbehavior
     * @dev Slashed amount is added to contest reward pool
     * @param participant Address to slash
     * @param amount Amount of stake to slash
     * @param reason Human-readable reason for slashing
     */
    function slashParticipantForMisbehavior(address participant, uint256 amount, string calldata reason) 
        external 
        onlyOwner 
    {
        require(participants[participant].isActive, "Not an active participant");
        require(amount > 0, "Amount must be greater than 0");
        require(participants[participant].stakedAmount >= amount, "Insufficient stake to slash");
        
        participants[participant].stakedAmount -= amount;
        
        IContestManager(contestManager).slashParticipant(address(this), participant, amount, reason);
        
        emit ParticipantSlashedInContest(participant, amount, reason);
    }

    function getReviewAssignmentsForReviewer(address reviewer) external view override returns (uint256[] memory) {
        return reviewAssignments[reviewer];
    }

    function isReviewerAssigned(address reviewer, uint256 submissionId) external view override returns (bool) {
        return isAssignedReviewer[reviewer][submissionId];
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

    /**
     * @notice QC can override aggregated score for quality control
     * @dev Only works for submissions assigned to QC during review assignment
     * @param submissionId Submission to override
     * @param qcScore New score (0-100) replacing peer review average
     */
    function overrideSubmissionScore(uint256 submissionId, uint256 qcScore) external onlyQC {
        require(submissionId < nextSubmissionId, "Invalid submission ID");
        require(qcScore >= MIN_SCORE && qcScore <= MAX_SCORE, "Invalid score");
        require(submissions[submissionId].isRevealed, "Submission not revealed");
        require(isAssignedReviewer[msg.sender][submissionId], "QC not assigned to this submission");
        
        ReviewStructs.Submission storage submission = submissions[submissionId];
        submission.aggregatedScore = qcScore;
        
        emit QCScoreOverride(submissionId, qcScore);
    }
    
    event QCScoreOverride(uint256 indexed submissionId, uint256 qcScore);
    event ParticipantLeft(address indexed participant);
    event MetadataUpdated(string oldMetadataURI, string newMetadataURI);
    event ContestPaused();
    event ContestUnpaused();
    event EmergencyWithdraw(address indexed owner, uint256 amount);
    event ParticipantSlashedInContest(address indexed participant, uint256 amount, string reason);

    /**
     * @notice Calculate current contest phase based on elapsed time
     * @dev Divides duration into 4 equal quarters: Commit → Reveal → ReviewCommit → ReviewReveal
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
