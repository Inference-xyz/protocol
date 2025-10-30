// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title IContestManager
 * @notice Interface for ContestManager - factory and manager for Contest instances
 */
interface IContestManager {
    struct ContestConfig {
        string metadataURI;
        uint256 duration;
        uint256 stakingAmount;
        uint256 reviewCount;
        uint256 epsilonReward;
        uint256 epsilonSlash;
        uint256 alpha;
        uint256 beta;
        uint256 gamma;
        uint256 minStakeAmount;
        uint256 maxParticipants;
        uint256 joinPriceAdjustment;
    }

    struct StakeInfo {
        uint256 amount;
        uint256 lockedUntil;
        bool isSlashed;
    }

    function createContest(ContestConfig calldata config) external returns (address contestAddress);
    
    function recordStake(address participant, uint256 amount) external;
    
    function refundStake(address contestAddress, address participant, uint256 amount) external;
    
    function unstakeFromContest(address contestAddress) external;
    
    function slashParticipant(address contestAddress, address participant, uint256 amount, string calldata reason) external;
    
    function distributeRewards(
        address contestAddress,
        address[] calldata participants,
        uint256[] calldata amounts
    ) external;
    
    function setContestTemplate(address template) external;
    
    function setInfToken(address token) external;
    
    function getStakeInfo(address contestAddress, address participant) external view returns (StakeInfo memory);
    
    function getCreatedContests() external view returns (address[] memory);
    
    function getContestsByCreator(address creator) external view returns (address[] memory);
    
    function isValidContest(address contest) external view returns (bool);
    
    function getContestTemplate() external view returns (address);
    
    function getInfToken() external view returns (address);
    
    function getRewardPool(address contestAddress) external view returns (uint256);
    
    event ContestCreated(
        address indexed contestAddress,
        address indexed creator,
        string metadataURI,
        uint256 stakingAmount
    );
    
    event Staked(
        address indexed contestAddress,
        address indexed participant,
        uint256 amount
    );
    
    event Unstaked(
        address indexed contestAddress,
        address indexed participant,
        uint256 amount
    );
    
    event ParticipantSlashed(
        address indexed contestAddress,
        address indexed participant,
        uint256 amount,
        string reason
    );
    
    event RewardsDistributed(
        address indexed contestAddress,
        uint256 totalAmount
    );
    
    event ContestTemplateUpdated(address indexed oldTemplate, address indexed newTemplate);
    
    event InfTokenUpdated(address indexed oldToken, address indexed newToken);
}

