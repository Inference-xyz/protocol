// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IInferenceDAO {
    struct Proposal {
        uint256 id;
        address proposer;
        address contest;
        uint256 emissionPercentage;
        uint256 forVotes;
        uint256 againstVotes;
        uint256 startTime;
        uint256 endTime;
        bool executed;
        bool canceled;
        string description;
    }

    struct Receipt {
        bool hasVoted;
        bool support;
        uint256 votes;
    }

    function proposeEmissionChange(
        address contest,
        uint256 emissionPercentage,
        string memory description
    ) external returns (uint256);

    function castVote(uint256 proposalId, bool support) external;
    function executeProposal(uint256 proposalId) external;
    function cancelProposal(uint256 proposalId) external;

    function getContestEmissionPercentage(address contest) external view returns (uint256);
    function getRegisteredContests() external view returns (address[] memory);
    function getProposal(uint256 proposalId) external view returns (
        uint256 id,
        address proposer,
        address contest,
        uint256 emissionPercentage,
        uint256 forVotes,
        uint256 againstVotes,
        uint256 startTime,
        uint256 endTime,
        bool executed,
        bool canceled,
        string memory description
    );
    function getReceipt(uint256 proposalId, address voter) external view returns (
        bool hasVoted,
        bool support,
        uint256 votes
    );

    // Events
    event ProposalCreated(
        uint256 indexed proposalId,
        address indexed proposer,
        address indexed contest,
        uint256 emissionPercentage,
        string description
    );
    event VoteCast(
        address indexed voter,
        uint256 indexed proposalId,
        bool support,
        uint256 votes
    );
    event ProposalExecuted(uint256 indexed proposalId);
    event ProposalCanceled(uint256 indexed proposalId);
    event EmissionPercentageSet(address indexed contest, uint256 emissionPercentage);
}
