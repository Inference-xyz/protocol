// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "./interfaces/IContest.sol";
import "./types/ReviewStructs.sol";

/**
 * @title Contest
 * @dev Simplified peer-reviewed AI inference contest
 */
contract Contest is IContest, Ownable, ReentrancyGuard {
    // Contest configuration
    ContestInfo public contestInfo;
    
    // Contest state
    uint256 public nextSubmissionId = 1;
    bool public isFinalized;
    
    // Simple configuration
    uint256 public constant MIN_SCORE = 0;
    uint256 public constant MAX_SCORE = 100;
    
    // Storage mappings
    mapping(address => Participant) public participants;
    mapping(uint256 => ReviewStructs.Submission) public submissions;
    mapping(address => uint256[]) public participantSubmissions;
    mapping(address => mapping(uint256 => uint256)) public reviews; // reviewer => submissionId => score
    
    // Arrays for iteration
    address[] public participantList;
    uint256[] public submissionIds;
    
    // Events
    event SubmissionSubmitted(
        address indexed participant,
        uint256 indexed submissionId,
        string ipfsURI,
        bytes32 outputHash
    );
    
    event ReviewSubmitted(
        address indexed reviewer,
        uint256 indexed submissionId,
        uint256 score
    );
    
    event ContestFinalized(
        address indexed winner,
        uint256 rewardAmount
    );

    modifier onlyParticipant() {
        require(participants[msg.sender].isActive, "Not an active participant");
        _;
    }

    constructor() Ownable(msg.sender) {}

    /**
     * @dev Initialize the contest
     */
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

    /**
     * @dev Join contest
     */
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

    /**
     * @dev Submit entry directly (simplified from commit-reveal)
     */
    function submitEntry(bytes32 outputHash, string calldata ipfsURI) external onlyParticipant {
        uint256 submissionId = nextSubmissionId++;
        
        submissions[submissionId] = ReviewStructs.Submission({
            participant: msg.sender,
            commitHash: outputHash,
            ipfsURI: ipfsURI,
            outputHash: outputHash,
            commitTime: block.timestamp,
            revealTime: block.timestamp,
            isRevealed: true,
            aggregatedScore: 0,
            reviewCount: 0
        });
        
        participantSubmissions[msg.sender].push(submissionId);
        submissionIds.push(submissionId);
        participants[msg.sender].lastSubmissionTime = block.timestamp;
        
        emit SubmissionSubmitted(msg.sender, submissionId, ipfsURI, outputHash);
    }

    /**
     * @dev Submit a review for a submission
     */
    function submitReview(uint256 submissionId, uint256 score) external onlyParticipant {
        require(submissionId < nextSubmissionId, "Invalid submission ID");
        require(score >= MIN_SCORE && score <= MAX_SCORE, "Invalid score");
        require(reviews[msg.sender][submissionId] == 0, "Already reviewed");
        require(submissions[submissionId].participant != msg.sender, "Cannot review own submission");
        
        reviews[msg.sender][submissionId] = score;
        
        // Update submission aggregated score (simple average)
        ReviewStructs.Submission storage submission = submissions[submissionId];
        submission.reviewCount++;
        submission.aggregatedScore = (submission.aggregatedScore * (submission.reviewCount - 1) + score) / submission.reviewCount;
        
        emit ReviewSubmitted(msg.sender, submissionId, score);
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
     * @dev Get review score for a submission by reviewer
     */
    function getReview(address reviewer, uint256 submissionId) external view returns (uint256) {
        return reviews[reviewer][submissionId];
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

    // Receive function to accept ETH for rewards
    receive() external payable {}
}



