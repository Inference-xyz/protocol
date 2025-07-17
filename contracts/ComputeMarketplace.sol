// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";
import "./interfaces/IComputeMarketplace.sol";
import "./interfaces/IZKVerifier.sol";

contract ComputeMarketplace is IComputeMarketplace, Ownable, ReentrancyGuard, Pausable {
    using SafeERC20 for IERC20;
    
    // State variables
    IZKVerifier public zkVerifier;
    uint256 public nextJobId;
    uint256 public minBounty;
    uint256 public defaultTimeout;
    
    // Mappings
    mapping(uint256 => Job) public jobs;
    mapping(address => ProviderStats) public providerStats;
    mapping(address => bool) public supportedTokens;
    mapping(address => uint256[]) public clientJobs;
    mapping(address => uint256[]) public providerJobs;
    mapping(bytes32 => bool) public registeredModelHashes;
    mapping(bytes32 => address) public modelHashRegistrants;
    
    // Array to track all jobs for enumeration
    uint256[] public allJobIds;
    
    constructor(
        address _owner,
        address _zkVerifier
    ) Ownable(_owner) {
        zkVerifier = IZKVerifier(_zkVerifier);
        minBounty = 0.01 ether;
        defaultTimeout = 24 hours;
        nextJobId = 1;
    }
    
    // Core Functions
    function postJob(
        bytes32 modelHash,
        bytes32 inputHash1,
        bytes32 inputHash2,
        bytes calldata encryptedInputs,
        uint256 bounty,
        address bountyToken,
        uint256 timeout
    ) external payable override nonReentrant whenNotPaused returns (uint256 jobId) {
        require(registeredModelHashes[modelHash], "Model hash not registered");
        require(bounty >= minBounty, "Bounty below minimum");
        require(timeout > 0, "Invalid timeout");
        require(encryptedInputs.length > 0, "Encrypted inputs required");
        
        // Handle payment
        if (bountyToken == address(0)) {
            require(msg.value == bounty, "Incorrect ETH amount");
        } else {
            require(supportedTokens[bountyToken], "Token not supported");
            require(msg.value == 0, "No ETH for ERC20 payment");
            IERC20(bountyToken).safeTransferFrom(msg.sender, address(this), bounty);
        }
        
        jobId = nextJobId++;
        
        jobs[jobId] = Job({
            id: jobId,
            client: msg.sender,
            modelHash: modelHash,
            inputHash1: inputHash1,
            inputHash2: inputHash2,
            encryptedInputs: encryptedInputs,
            bounty: bounty,
            bountyToken: bountyToken,
            timeout: timeout,
            createdAt: block.timestamp,
            claimedAt: 0,
            provider: address(0),
            completed: false,
            timedOut: false
        });
        
        allJobIds.push(jobId);
        clientJobs[msg.sender].push(jobId);
        
        emit JobPosted(
            jobId,
            msg.sender,
            modelHash,
            inputHash1,
            inputHash2,
            encryptedInputs,
            bounty,
            bountyToken,
            timeout
        );
    }
    
    function claimJob(uint256 jobId) external override nonReentrant whenNotPaused {
        Job storage job = jobs[jobId];
        require(job.id != 0, "Job does not exist");
        require(job.provider == address(0), "Job already claimed");
        require(!job.completed, "Job already completed");
        require(!job.timedOut, "Job timed out");
        require(msg.sender != job.client, "Client cannot claim own job");
        
        job.provider = msg.sender;
        job.claimedAt = block.timestamp;
        providerJobs[msg.sender].push(jobId);
        
        emit JobClaimed(jobId, msg.sender);
    }
    
    function completeJob(
        uint256 jobId,
        bytes calldata encryptedOutput,
        bytes calldata zkProof
    ) external override nonReentrant whenNotPaused {
        Job storage job = jobs[jobId];
        require(job.id != 0, "Job does not exist");
        require(job.provider == msg.sender, "Only provider can complete");
        require(!job.completed, "Job already completed");
        require(!job.timedOut, "Job timed out");
        require(encryptedOutput.length > 0, "Encrypted output required");
        require(zkProof.length > 0, "ZK proof required");
        
        // Verify ZK proof
        bytes32[] memory inputs = new bytes32[](2);
        inputs[0] = job.inputHash1;
        inputs[1] = job.inputHash2;
        
        require(
            zkVerifier.verifyProof(zkProof, inputs),
            "Invalid ZK proof"
        );
        
        // Mark job as completed
        job.completed = true;
        
        // Update provider stats
        ProviderStats storage stats = providerStats[msg.sender];
        stats.totalJobs++;
        stats.completedJobs++;
        stats.totalEarnings += job.bounty;
        
        if (job.claimedAt > 0) {
            uint256 responseTime = block.timestamp - job.claimedAt;
            stats.averageResponseTime = (stats.averageResponseTime * (stats.completedJobs - 1) + responseTime) / stats.completedJobs;
        }
        
        // Calculate reputation score (simple implementation)
        stats.reputationScore = (stats.completedJobs * 100) / stats.totalJobs;
        
        // Transfer payment to provider
        if (job.bountyToken == address(0)) {
            payable(msg.sender).transfer(job.bounty);
        } else {
            IERC20(job.bountyToken).safeTransfer(msg.sender, job.bounty);
        }
        
        emit JobCompleted(jobId, msg.sender, encryptedOutput, zkProof);
    }
    
    function handleJobTimeout(uint256 jobId) external override nonReentrant whenNotPaused {
        Job storage job = jobs[jobId];
        require(job.id != 0, "Job does not exist");
        require(!job.completed, "Job already completed");
        require(!job.timedOut, "Job already timed out");
        
        bool isTimedOut = false;
        if (job.provider == address(0)) {
            // Unclaimed job - timeout from creation
            isTimedOut = block.timestamp >= job.createdAt + job.timeout;
        } else {
            // Claimed job - timeout from claim
            isTimedOut = block.timestamp >= job.claimedAt + job.timeout;
        }
        
        require(isTimedOut, "Job has not timed out yet");
        
        job.timedOut = true;
        
        // Refund bounty to client
        if (job.bountyToken == address(0)) {
            payable(job.client).transfer(job.bounty);
        } else {
            IERC20(job.bountyToken).safeTransfer(job.client, job.bounty);
        }
        
        emit JobTimedOut(jobId, job.client);
    }
    
    // Model Management
    function registerModelHash(bytes32 modelHash) external override {
        require(!registeredModelHashes[modelHash], "Model hash already registered");
        
        registeredModelHashes[modelHash] = true;
        modelHashRegistrants[modelHash] = msg.sender;
        
        emit ModelHashRegistered(modelHash, msg.sender);
    }
    
    function isModelHashRegistered(bytes32 modelHash) external view override returns (bool) {
        return registeredModelHashes[modelHash];
    }
    
    function getModelHashRegistrant(bytes32 modelHash) external view override returns (address) {
        return modelHashRegistrants[modelHash];
    }
    
    // View Functions
    function getJob(uint256 jobId) external view override returns (Job memory) {
        return jobs[jobId];
    }
    
    function getJobsByClient(address client) external view override returns (uint256[] memory) {
        return clientJobs[client];
    }
    
    function getJobsByProvider(address provider) external view override returns (uint256[] memory) {
        return providerJobs[provider];
    }
    
    function getProviderStats(address provider) external view override returns (ProviderStats memory) {
        return providerStats[provider];
    }
    
    function getAllJobs() external view override returns (Job[] memory) {
        Job[] memory allJobs = new Job[](allJobIds.length);
        for (uint256 i = 0; i < allJobIds.length; i++) {
            allJobs[i] = jobs[allJobIds[i]];
        }
        return allJobs;
    }
    
    function getAvailableJobs() external view override returns (Job[] memory) {
        uint256 availableCount = 0;
        
        // Count available jobs
        for (uint256 i = 0; i < allJobIds.length; i++) {
            Job memory job = jobs[allJobIds[i]];
            if (job.provider == address(0) && !job.completed && !job.timedOut) {
                availableCount++;
            }
        }
        
        // Create array of available jobs
        Job[] memory availableJobs = new Job[](availableCount);
        uint256 index = 0;
        
        for (uint256 i = 0; i < allJobIds.length; i++) {
            Job memory job = jobs[allJobIds[i]];
            if (job.provider == address(0) && !job.completed && !job.timedOut) {
                availableJobs[index] = job;
                index++;
            }
        }
        
        return availableJobs;
    }
    
    // Admin Functions
    function setSupportedTokens(address[] calldata tokens) external override onlyOwner {
        // Clear existing supported tokens
        for (uint256 i = 0; i < allJobIds.length; i++) {
            Job memory job = jobs[allJobIds[i]];
            if (job.bountyToken != address(0)) {
                supportedTokens[job.bountyToken] = false;
            }
        }
        
        // Set new supported tokens
        for (uint256 i = 0; i < tokens.length; i++) {
            supportedTokens[tokens[i]] = true;
        }
    }
    
    function isSupportedToken(address token) external view override returns (bool) {
        return supportedTokens[token];
    }
    
    function setMinBounty(uint256 newMinBounty) external override onlyOwner {
        minBounty = newMinBounty;
    }
    
    function getMinBounty() external view override returns (uint256) {
        return minBounty;
    }
    
    function setJobTimeout(uint256 newTimeout) external override onlyOwner {
        require(newTimeout > 0, "Invalid timeout");
        defaultTimeout = newTimeout;
    }
    
    function getJobTimeout() external view override returns (uint256) {
        return defaultTimeout;
    }
    
    function pause() external override onlyOwner {
        _pause();
    }
    
    function unpause() external override onlyOwner {
        _unpause();
    }
    
    function setZKVerifier(address _zkVerifier) external onlyOwner {
        require(_zkVerifier != address(0), "Invalid verifier address");
        zkVerifier = IZKVerifier(_zkVerifier);
    }
    
    // Emergency Functions
    function emergencyWithdraw(address token) external override onlyOwner {
        if (token == address(0)) {
            payable(owner()).transfer(address(this).balance);
        } else {
            IERC20(token).safeTransfer(owner(), IERC20(token).balanceOf(address(this)));
        }
    }
} 