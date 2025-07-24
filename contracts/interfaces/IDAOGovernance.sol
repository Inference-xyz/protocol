// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IDAOGovernance {
    struct Proposal {
        uint256 id;
        address proposer;
        string description;
        bytes data;
        uint256 forVotes;
        uint256 againstVotes;
        uint256 startTime;
        uint256 endTime;
        bool executed;
        bool canceled;
        ProposalType proposalType;
    }

    enum ProposalType {
        EmissionSplit,
        EmissionRate,
        ContestWeight,
        ProtocolUpgrade,
        Emergency
    }

    enum VoteType {
        Against,
        For,
        Abstain
    }

    struct Receipt {
        bool hasVoted;
        VoteType support;
        uint256 votes;
    }

    // Core Functions
    function proposeEmissionSplit(uint256[] calldata contestWeights, string calldata description)
        external
        returns (uint256 proposalId);
    function proposeEmissionRate(uint256 newRate, string calldata description) external returns (uint256 proposalId);
    function proposeContestWeight(address contest, uint256 weight, string calldata description)
        external
        returns (uint256 proposalId);

    function vote(uint256 proposalId, VoteType support) external;
    function executeProposal(uint256 proposalId) external;
    function cancelProposal(uint256 proposalId) external;

    // View Functions
    function getProposal(uint256 proposalId) external view returns (Proposal memory);
    function getActiveProposals() external view returns (uint256[] memory);
    function getProposalReceipt(uint256 proposalId, address voter) external view returns (Receipt memory);
    function hasVoted(uint256 proposalId, address account) external view returns (bool);
    function getVotingPower(address account) external view returns (uint256);

    // Configuration Functions
    function setVotingDelay(uint256 delay) external;
    function setVotingPeriod(uint256 period) external;
    function setProposalThreshold(uint256 threshold) external;
    function setQuorumVotes(uint256 quorum) external;
    function setRewardDistributor(address distributor) external;

    // View Configuration
    function getVotingDelay() external view returns (uint256);
    function getVotingPeriod() external view returns (uint256);
    function getProposalThreshold() external view returns (uint256);
    function getQuorumVotes() external view returns (uint256);
    function getRewardDistributor() external view returns (address);

    // Events
    event ProposalCreated(
        uint256 indexed proposalId, address indexed proposer, string description, ProposalType proposalType
    );
    event VoteCast(address indexed voter, uint256 indexed proposalId, VoteType support, uint256 votes);
    event ProposalExecuted(uint256 indexed proposalId);
    event ProposalCanceled(uint256 indexed proposalId);
    event VotingDelayUpdated(uint256 newDelay);
    event VotingPeriodUpdated(uint256 newPeriod);
    event ProposalThresholdUpdated(uint256 newThreshold);
    event QuorumVotesUpdated(uint256 newQuorum);
    event RewardDistributorUpdated(address indexed newDistributor);
}
