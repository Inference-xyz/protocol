// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "./InfToken.sol";

contract InferenceDAO is Ownable {
    struct Proposal {
        uint256 id;
        address proposer;
        address contest;
        uint256 emissionPercentage;
        uint256 forVotes;
        uint256 againstVotes;
        uint256 endTime;
        bool executed;
        string description;
    }

    InfToken public immutable infToken;
    
    uint256 public proposalCount;
    uint256 public votingPeriod = 7 days;
    uint256 public quorumVotes = 40000 * 10**18;
    uint256 public proposalThreshold = 1000 * 10**18;

    mapping(uint256 => Proposal) public proposals;
    mapping(address => uint256) public contestEmissionPercentages;
    mapping(uint256 => mapping(address => bool)) public hasVoted;

    event ProposalCreated(uint256 indexed proposalId, address indexed contest, uint256 emissionPercentage);
    event VoteCast(address indexed voter, uint256 indexed proposalId, bool support, uint256 votes);
    event ProposalExecuted(uint256 indexed proposalId, address indexed contest, uint256 emissionPercentage);

    constructor(InfToken _infToken) Ownable(msg.sender) {
        infToken = _infToken;
    }

    function proposeEmissionChange(
        address contest,
        uint256 emissionPercentage,
        string memory description
    ) external returns (uint256) {
        require(emissionPercentage <= 100, "Emission percentage cannot exceed 100%");
        require(contest != address(0), "Invalid contest address");
        require(
            infToken.balanceOf(msg.sender) >= proposalThreshold,
            "Insufficient tokens to create proposal"
        );

        proposalCount++;
        uint256 proposalId = proposalCount;

        proposals[proposalId] = Proposal({
            id: proposalId,
            proposer: msg.sender,
            contest: contest,
            emissionPercentage: emissionPercentage,
            forVotes: 0,
            againstVotes: 0,
            endTime: block.timestamp + votingPeriod,
            executed: false,
            description: description
        });

        emit ProposalCreated(proposalId, contest, emissionPercentage);
        return proposalId;
    }

    function castVote(uint256 proposalId, bool support) external {
        Proposal storage proposal = proposals[proposalId];
        require(proposal.id != 0, "Proposal does not exist");
        require(block.timestamp < proposal.endTime, "Voting ended");
        require(!proposal.executed, "Proposal already executed");
        require(!hasVoted[proposalId][msg.sender], "Already voted");

        uint256 votes = infToken.balanceOf(msg.sender);
        require(votes > 0, "No voting power");

        hasVoted[proposalId][msg.sender] = true;

        if (support) {
            proposal.forVotes += votes;
        } else {
            proposal.againstVotes += votes;
        }

        emit VoteCast(msg.sender, proposalId, support, votes);
    }

    function executeProposal(uint256 proposalId) external {
        Proposal storage proposal = proposals[proposalId];
        require(proposal.id != 0, "Proposal does not exist");
        require(block.timestamp >= proposal.endTime, "Voting not ended");
        require(!proposal.executed, "Already executed");
        require(
            proposal.forVotes > proposal.againstVotes,
            "Proposal not passed"
        );
        require(
            proposal.forVotes + proposal.againstVotes >= quorumVotes,
            "Quorum not reached"
        );

        proposal.executed = true;
        contestEmissionPercentages[proposal.contest] = proposal.emissionPercentage;

        emit ProposalExecuted(proposalId, proposal.contest, proposal.emissionPercentage);
    }

    function getContestEmissionPercentage(address contest) external view returns (uint256) {
        return contestEmissionPercentages[contest];
    }

    function getProposal(uint256 proposalId) external view returns (
        uint256 id,
        address proposer,
        address contest,
        uint256 emissionPercentage,
        uint256 forVotes,
        uint256 againstVotes,
        uint256 endTime,
        bool executed,
        string memory description
    ) {
        Proposal storage proposal = proposals[proposalId];
        return (
            proposal.id,
            proposal.proposer,
            proposal.contest,
            proposal.emissionPercentage,
            proposal.forVotes,
            proposal.againstVotes,
            proposal.endTime,
            proposal.executed,
            proposal.description
        );
    }

    // Admin functions
    function setVotingPeriod(uint256 _votingPeriod) external onlyOwner {
        votingPeriod = _votingPeriod;
    }

    function setQuorumVotes(uint256 _quorumVotes) external onlyOwner {
        quorumVotes = _quorumVotes;
    }

    function setProposalThreshold(uint256 _proposalThreshold) external onlyOwner {
        proposalThreshold = _proposalThreshold;
    }
}
