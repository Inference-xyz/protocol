// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/proxy/Clones.sol";
import "./interfaces/IContestFactory.sol";
import "./Contest.sol";

/**
 * @title ContestFactory
 * @dev Simplified factory contract for creating peer-reviewed inference contests
 */
contract ContestFactory is IContestFactory, Ownable {
    using Clones for address;

    // Template contract
    address public contestTemplate;
    
    // Contest tracking
    address[] public allContests;
    mapping(address => address[]) public contestsByCreator;
    mapping(address => bool) public validContests;

    event ContestTemplateUpdated(address indexed oldTemplate, address indexed newTemplate);

    constructor() Ownable(msg.sender) {
        // Deploy template contract
        Contest contest = new Contest();
        contestTemplate = address(contest);
    }

    /**
     * @dev Create a new peer-reviewed contest
     */
    function createContest(ContestConfig calldata config) public override returns (address contestAddress) {
        require(contestTemplate != address(0), "Contest template not set");
        require(config.creatorFeePct <= 5000, "Creator fee too high"); // Max 50%
        
        // Clone the contest template
        contestAddress = contestTemplate.clone();
        
        // Initialize the contest with 14 day duration
        Contest(payable(contestAddress)).initialize(
            msg.sender,
            config.metadataURI,
            config.creatorFeePct,
            14 days,
            false // Not everlasting for peer review contests
        );
        
        // Track the contest
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

    function setEZKLFactory(address) external pure override {
        revert("Not applicable for peer review contests");
    }

    function getValidatorSetTemplate() external pure override returns (address) {
        return address(0);
    }

    function getScoringVerifierTemplate() external pure override returns (address) {
        return address(0);
    }

    function getEZKLFactory() external pure override returns (address) {
        return address(0);
    }
}
