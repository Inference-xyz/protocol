// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/proxy/Clones.sol";
import "./interfaces/IContestFactory.sol";
import "./Contest.sol";

contract ContestFactory is IContestFactory, Ownable {
    using Clones for address;

    address public contestTemplate;
    
    address[] public allContests;
    mapping(address => address[]) public contestsByCreator;
    mapping(address => bool) public validContests;

    // Events are defined in IContestFactory interface

    constructor() Ownable(msg.sender) {
        Contest contest = new Contest();
        contestTemplate = address(contest);
    }

    function createContest(ContestConfig calldata config) public payable override returns (address contestAddress) {
        require(contestTemplate != address(0), "Contest template not set");
        require(config.creatorFeePct <= 5000, "Creator fee too high"); // Max 50%
        
        contestAddress = contestTemplate.clone();
        
        Contest(payable(contestAddress)).initialize(
            msg.sender,
            config.metadataURI,
            config.creatorFeePct,
            config.duration,
            false
        );
        
        // Transfer any ETH sent to the contest as initial reward pool
        if (msg.value > 0) {
            (bool success, ) = contestAddress.call{value: msg.value}("");
            require(success, "Failed to transfer reward pool");
        }
        
        allContests.push(contestAddress);
        contestsByCreator[msg.sender].push(contestAddress);
        validContests[contestAddress] = true;
        
        emit ContestCreated(contestAddress, msg.sender, config.metadataURI, config.creatorFeePct);
        
        return contestAddress;
    }

    /**
     * @dev Create contest with initial emission weight (for compatibility)
     */
    function createContestWithInitialWeight(
        ContestConfig calldata config,
        uint256 /* initialWeight */
    ) external override returns (address contestAddress) {
        return createContest(config);
    }

    /**
     * @dev Create contest with validator set (not applicable for peer review)
     */
    function createContestWithValidatorSet(
        ContestConfig calldata config,
        address /* validatorSet */
    ) external override returns (address contestAddress) {
        return createContest(config);
    }

    /**
     * @dev Create contest with scoring verifier (not applicable for peer review)
     */
    function createContestWithScoringVerifier(
        ContestConfig calldata config,
        address /* scoringVerifier */
    ) external override returns (address contestAddress) {
        return createContest(config);
    }

    /**
     * @dev Set the contest template (only owner)
     */
    function setContestTemplate(address template) external override onlyOwner {
        require(template != address(0), "Invalid template");
        address oldTemplate = contestTemplate;
        contestTemplate = template;
        emit ContestTemplateUpdated(oldTemplate, template);
    }

    /**
     * @dev Set the reward distributor contract
     */
    function setRewardDistributor(address distributor) external override onlyOwner {
        // Not used in simplified version
    }

    /**
     * @dev Set the contest registry contract
     */
    function setContestRegistry(address registry) external override onlyOwner {
        // Not used in simplified version
    }

    // View functions
    function getCreatedContests() external view override returns (address[] memory) {
        return allContests;
    }

    function getContestsByCreator(address creator) external view override returns (address[] memory) {
        return contestsByCreator[creator];
    }

    function isValidContest(address contest) external view override returns (bool) {
        return validContests[contest];
    }

    function getContestTemplate() external view override returns (address) {
        return contestTemplate;
    }

    function getTotalContestsCreated() external view override returns (uint256) {
        return allContests.length;
    }

    // Functions for compatibility with existing interface (not used in peer review)
    function setValidatorSetTemplate(address) external pure override {
        revert("Not applicable for peer review contests");
    }

    function setScoringVerifierTemplate(address) external pure override {
        revert("Not applicable for peer review contests");
    }

    function getValidatorSetTemplate() external pure override returns (address) {
        return address(0);
    }

    function getScoringVerifierTemplate() external pure override returns (address) {
        return address(0);
    }

}
