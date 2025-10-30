// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/proxy/Clones.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "./interfaces/IContestManager.sol";
import "./Contest.sol";

/**
 * @title ContestManager
 * @notice Factory and manager for Contest instances. Handles staking, slashing, and reward distribution
 * @dev Uses minimal proxy (EIP-1167) pattern to deploy cheap Contest clones
 * Holds all staked tokens and reward pools for all contests
 */
contract ContestManager is IContestManager, Ownable, ReentrancyGuard {
    using Clones for address;

    address public contestTemplate;
    address public infToken;
    
    address[] public allContests;
    mapping(address => address[]) public contestsByCreator;
    mapping(address => bool) public validContests;
    mapping(address => mapping(address => StakeInfo)) public stakes;
    mapping(address => uint256) public contestRewardPools;
    mapping(address => uint256) public contestEndTime;
    
    uint256 public constant CHALLENGE_PERIOD = 7 days; // Lock period after contest ends for disputes

    constructor(address _infToken) Ownable(msg.sender) {
        require(_infToken != address(0), "Invalid INF token address");
        infToken = _infToken;
        
        Contest contest = new Contest();
        contestTemplate = address(contest);
    }

    /**
     * @notice Create new contest using minimal proxy pattern
     * @dev Clones contestTemplate and initializes it. Creator becomes contest owner
     * @param config Contest configuration (metadata, duration, QC address, etc.)
     * @return contestAddress Address of newly created contest clone
     */
    function createContest(ContestConfig calldata config) external override returns (address contestAddress) {
        contestAddress = contestTemplate.clone();
        
        Contest(contestAddress).initialize(
            msg.sender,
            config.metadataURI,
            config.duration,
            address(this),
            infToken,
            config.reviewCount,
            config.outlierThreshold,
            config.slashRatio
        );
        
        allContests.push(contestAddress);
        contestsByCreator[msg.sender].push(contestAddress);
        validContests[contestAddress] = true;
        contestEndTime[contestAddress] = block.timestamp + config.duration + CHALLENGE_PERIOD;
        
        emit ContestCreated(contestAddress, msg.sender, config.metadataURI, config.stakingAmount);
        
        return contestAddress;
    }
    
    /**
     * @notice Record participant stake (called by Contest during joinContest)
     * @dev Only callable by valid Contest contracts. Tokens should be transferred before this call
     * @param participant Address of participant
     * @param amount Stake amount to record
     */
    function recordStake(address participant, uint256 amount) external {
        require(validContests[msg.sender], "Only valid contests");
        require(amount > 0, "Amount must be greater than 0");
        require(participant != address(0), "Invalid participant address");
        
        StakeInfo storage stakeInfo = stakes[msg.sender][participant];
        require(!stakeInfo.isSlashed, "Participant is slashed");
        
        stakeInfo.amount += amount;
        stakeInfo.lockedUntil = contestEndTime[msg.sender];
        
        emit Staked(msg.sender, participant, amount);
    }

    /**
     * @notice Refund stake to participant (called by Contest during leaveContest)
     * @dev Only callable by valid Contest contracts. Used when participant leaves before contest starts
     * @param contestAddress Contest from which participant is leaving
     * @param participant Address of participant to refund
     * @param amount Stake amount to refund
     */
    function refundStake(address contestAddress, address participant, uint256 amount) external override nonReentrant {
        require(validContests[msg.sender], "Only valid contests");
        require(msg.sender == contestAddress, "Contest address mismatch");
        require(participant != address(0), "Invalid participant address");
        require(amount > 0, "Amount must be greater than 0");
        
        StakeInfo storage stakeInfo = stakes[contestAddress][participant];
        require(stakeInfo.amount >= amount, "Insufficient stake to refund");
        
        // Update state before external call
        stakeInfo.amount -= amount;
        
        require(IERC20(infToken).transfer(participant, amount), "Transfer failed");
        
        emit Unstaked(contestAddress, participant, amount);
    }

    /**
     * @notice Withdraw stake after contest ends + challenge period
     * @dev Locked until contestEndTime (contest duration + CHALLENGE_PERIOD). Cannot unstake if slashed
     * @param contestAddress Contest to withdraw stake from
     */
    function unstakeFromContest(address contestAddress) external override nonReentrant {
        require(validContests[contestAddress], "Invalid contest");
        
        StakeInfo storage stakeInfo = stakes[contestAddress][msg.sender];
        require(stakeInfo.amount > 0, "No stake to withdraw");
        require(block.timestamp >= stakeInfo.lockedUntil, "Stake is still locked");
        require(!stakeInfo.isSlashed, "Stake was slashed");
        
        uint256 amount = stakeInfo.amount;
        
        // Update state before external call
        stakeInfo.amount = 0;
        
        require(IERC20(infToken).transfer(msg.sender, amount), "Transfer failed");
        
        emit Unstaked(contestAddress, msg.sender, amount);
    }

    /**
     * @notice Slash participant's stake and add to contest reward pool
     * @dev Can only be called by contest contract or manager owner. Marks participant as slashed (cannot unstake)
     * @param contestAddress Contest where slashing occurs
     * @param participant Address to slash
     * @param amount Amount to slash
     * @param reason Human-readable reason for audit trail
     */
    function slashParticipant(
        address contestAddress,
        address participant,
        uint256 amount,
        string calldata reason
    ) external override {
        require(validContests[contestAddress], "Invalid contest");
        require(msg.sender == contestAddress || msg.sender == owner(), "Not authorized");
        
        StakeInfo storage stakeInfo = stakes[contestAddress][participant];
        require(stakeInfo.amount >= amount, "Insufficient stake to slash");
        
        stakeInfo.amount -= amount;
        stakeInfo.isSlashed = true;
        
        contestRewardPools[contestAddress] += amount;
        
        emit ParticipantSlashed(contestAddress, participant, amount, reason);
    }

    /**
     * @notice Distribute rewards from contest pool to winners
     * @dev Called by Contest.finalizeContest(). Uses CEI pattern - updates state before transfers
     * @param contestAddress Contest distributing rewards
     * @param participants Array of winner addresses
     * @param amounts Corresponding reward amounts
     */
    function distributeRewards(
        address contestAddress,
        address[] calldata participants,
        uint256[] calldata amounts
    ) external override nonReentrant {
        require(validContests[contestAddress], "Invalid contest");
        require(msg.sender == contestAddress || msg.sender == owner(), "Not authorized");
        require(participants.length == amounts.length, "Arrays length mismatch");
        require(participants.length > 0, "No participants");
        
        uint256 totalAmount = 0;
        for (uint256 i = 0; i < amounts.length; i++) {
            require(participants[i] != address(0), "Invalid participant address");
            totalAmount += amounts[i];
        }
        
        require(contestRewardPools[contestAddress] >= totalAmount, "Insufficient reward pool");
        
        // CEI: Update state before external calls to prevent reentrancy
        contestRewardPools[contestAddress] -= totalAmount;
        
        for (uint256 i = 0; i < participants.length; i++) {
            if (amounts[i] > 0) {
                require(IERC20(infToken).transfer(participants[i], amounts[i]), "Transfer failed");
            }
        }
        
        emit RewardsDistributed(contestAddress, totalAmount);
    }

    /**
     * @notice Add funds to contest reward pool (anyone can fund)
     * @param contestAddress Contest to fund
     * @param amount Amount of INF tokens to add
     */
    function fundRewardPool(address contestAddress, uint256 amount) external nonReentrant {
        require(validContests[contestAddress], "Invalid contest");
        require(amount > 0, "Amount must be greater than 0");
        
        require(IERC20(infToken).transferFrom(msg.sender, address(this), amount), "Transfer failed");
        contestRewardPools[contestAddress] += amount;
    }

    function setContestTemplate(address template) external override onlyOwner {
        require(template != address(0), "Invalid template");
        address oldTemplate = contestTemplate;
        contestTemplate = template;
        emit ContestTemplateUpdated(oldTemplate, template);
    }

    function setInfToken(address token) external override onlyOwner {
        require(token != address(0), "Invalid token");
        address oldToken = infToken;
        infToken = token;
        emit InfTokenUpdated(oldToken, token);
    }

    function getStakeInfo(address contestAddress, address participant) 
        external 
        view 
        override 
        returns (StakeInfo memory) 
    {
        return stakes[contestAddress][participant];
    }

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

    function getInfToken() external view override returns (address) {
        return infToken;
    }

    function getRewardPool(address contestAddress) external view returns (uint256) {
        return contestRewardPools[contestAddress];
    }
    
    /**
     * @notice Emergency withdraw all tokens from manager (owner only)
     * @dev Use only in critical situations. Will withdraw ALL tokens including stakes and reward pools
     * @custom:security This breaks normal operation - use with extreme caution
     */
    function emergencyWithdrawFromManager() external onlyOwner {
        uint256 balance = IERC20(infToken).balanceOf(address(this));
        require(balance > 0, "No balance to withdraw");
        
        require(IERC20(infToken).transfer(owner(), balance), "Transfer failed");
        
        emit EmergencyWithdrawFromManager(owner(), balance);
    }
    
    event EmergencyWithdrawFromManager(address indexed owner, uint256 amount);
}

