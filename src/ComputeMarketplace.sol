// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "./interfaces/IComputeMarketplace.sol";
import "./interfaces/IZKVerifier.sol";

contract ComputeMarketplace is IComputeMarketplace, AccessControl, Pausable, ReentrancyGuard {
    using SafeERC20 for IERC20;
    
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant RESOLVER_ROLE = keccak256("RESOLVER_ROLE");
    
    mapping(uint256 => Job) private _jobs;
    mapping(address => ProviderStats) private _providerStats;
    mapping(address => uint256) private _pendingRewards;
    mapping(address => bool) private _supportedTokens;
    mapping(address => mapping(address => uint256)) private _paymentBalances;
    
    uint256 private _jobCounter;
    uint256 private _defaultTimeout = 24 hours;
    uint256 private _minBounty = 1e18; // 1 token
    address private _zkVerifier;
    
    address[] private _supportedTokensList;
    
    modifier validJob(uint256 jobId) {
        require(jobId > 0 && jobId <= _jobCounter, "Invalid job ID");
        _;
    }
    
    constructor(address admin, address zkVerifier) {
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(ADMIN_ROLE, admin);
        _grantRole(RESOLVER_ROLE, admin);
        _zkVerifier = zkVerifier;
    }
    
    // Core Job Functions
    function postJob(
        string calldata specURI,
        uint256 bounty,
        address paymentToken,
        uint256 timeout
    ) external payable override whenNotPaused nonReentrant returns (uint256 jobId) {
        require(bounty >= _minBounty, "Bounty too low");
        require(_supportedTokens[paymentToken] || paymentToken == address(0), "Token not supported");
        require(timeout > 0, "Invalid timeout");
        
        _jobCounter++;
        jobId = _jobCounter;
        
        Job storage job = _jobs[jobId];
        job.id = jobId;
        job.requester = msg.sender;
        job.bounty = bounty;
        job.paymentToken = paymentToken;
        job.specURI = specURI;
        job.status = JobStatus.Posted;
        job.postedAt = block.timestamp;
        job.timeout = timeout;
        
        // Handle payment
        if (paymentToken == address(0)) {
            require(msg.value >= bounty, "Insufficient ETH");
        } else {
            IERC20(paymentToken).safeTransferFrom(msg.sender, address(this), bounty);
        }
        
        emit JobPosted(jobId, msg.sender, bounty, paymentToken, specURI, timeout);
    }
    
    function claimJob(uint256 jobId) external override whenNotPaused validJob(jobId) {
        Job storage job = _jobs[jobId];
        require(job.status == JobStatus.Posted, "Job not available");
        require(!isJobTimeout(jobId), "Job expired");
        
        job.provider = msg.sender;
        job.status = JobStatus.Claimed;
        job.claimedAt = block.timestamp;
        
        emit JobClaimed(jobId, msg.sender, block.timestamp);
    }
    
    function completeJob(
        uint256 jobId,
        string calldata resultURI,
        bytes32 resultHash,
        bytes calldata zkProof
    ) external override whenNotPaused validJob(jobId) {
        Job storage job = _jobs[jobId];
        require(job.provider == msg.sender, "Not job provider");
        require(job.status == JobStatus.Claimed, "Job not claimed");
        require(!isJobTimeout(jobId), "Job expired");
        
        // Verify ZK proof if verifier is set
        if (_zkVerifier != address(0)) {
            bool proofValid = IZKVerifier(_zkVerifier).verifyProof(zkProof, new bytes32[](0));
            require(proofValid, "Invalid ZK proof");
            emit ZKProofVerified(jobId, true, gasleft());
        }
        
        job.status = JobStatus.Completed;
        job.completedAt = block.timestamp;
        job.resultHash = resultHash;
        
        // Update provider stats
        ProviderStats storage stats = _providerStats[msg.sender];
        stats.totalJobs++;
        stats.completedJobs++;
        stats.totalEarnings += job.bounty;
        stats.lastActive = block.timestamp;
        stats.reputation += 10; // Increase reputation for successful completion
        
        // Add to pending rewards
        _pendingRewards[msg.sender] += job.bounty;
        
        emit JobCompleted(jobId, msg.sender, resultURI, resultHash);
    }
    
    function disputeJob(uint256 jobId, string calldata reason) external override whenNotPaused validJob(jobId) {
        Job storage job = _jobs[jobId];
        require(job.requester == msg.sender, "Not job requester");
        require(job.status == JobStatus.Completed, "Job not completed");
        
        job.status = JobStatus.Disputed;
        emit JobDisputed(jobId, msg.sender, reason);
    }
    
    function resolveDispute(uint256 jobId, bool favorProvider) external override onlyRole(RESOLVER_ROLE) validJob(jobId) {
        Job storage job = _jobs[jobId];
        require(job.status == JobStatus.Disputed, "Job not disputed");
        
        if (favorProvider) {
            job.status = JobStatus.Completed;
        } else {
            job.status = JobStatus.Failed;
            // Remove from pending rewards
            if (_pendingRewards[job.provider] >= job.bounty) {
                _pendingRewards[job.provider] -= job.bounty;
            }
            
            // Update provider stats
            ProviderStats storage stats = _providerStats[job.provider];
            stats.failedJobs++;
            if (stats.completedJobs > 0) {
                stats.completedJobs--;
            }
            if (stats.totalEarnings >= job.bounty) {
                stats.totalEarnings -= job.bounty;
            }
            if (stats.reputation >= 5) {
                stats.reputation -= 5;
            }
        }
        
        emit DisputeResolved(jobId, favorProvider, msg.sender);
    }
    
    // Batch Functions
    function batchSubmitInputs(
        uint256 jobId,
        bytes32[] calldata inputs,
        uint256 processingFee
    ) external override whenNotPaused validJob(jobId) returns (uint256 batchId) {
        Job storage job = _jobs[jobId];
        require(job.requester == msg.sender, "Not job requester");
        
        batchId = job.batchInputs.length;
        job.batchInputs.push(BatchInput({
            inputs: inputs,
            batchId: batchId,
            processed: false,
            processingFee: processingFee
        }));
        
        emit BatchInputsSubmitted(jobId, batchId, inputs, processingFee);
    }
    
    function processBatch(uint256 jobId, uint256 batchId) external override whenNotPaused validJob(jobId) {
        Job storage job = _jobs[jobId];
        require(job.provider == msg.sender, "Not job provider");
        require(batchId < job.batchInputs.length, "Invalid batch ID");
        require(!job.batchInputs[batchId].processed, "Batch already processed");
        
        job.batchInputs[batchId].processed = true;
        emit BatchProcessed(jobId, batchId);
    }
    
    // Payment Functions
    function payWithERC20(address token, uint256 amount) external override whenNotPaused {
        require(_supportedTokens[token], "Token not supported");
        IERC20(token).safeTransferFrom(msg.sender, address(this), amount);
        _paymentBalances[msg.sender][token] += amount;
        emit PaymentReceived(token, amount, msg.sender);
    }
    
    function getPaymentBalance(address token) external view override returns (uint256) {
        return _paymentBalances[msg.sender][token];
    }
    
    function withdrawPayment(address token, uint256 amount) external override nonReentrant {
        require(_paymentBalances[msg.sender][token] >= amount, "Insufficient balance");
        _paymentBalances[msg.sender][token] -= amount;
        
        if (token == address(0)) {
            payable(msg.sender).transfer(amount);
        } else {
            IERC20(token).safeTransfer(msg.sender, amount);
        }
        
        emit PaymentWithdrawn(token, amount, msg.sender);
    }
    
    function withdrawBounty(uint256 jobId) external override nonReentrant validJob(jobId) {
        Job storage job = _jobs[jobId];
        require(job.requester == msg.sender, "Not job requester");
        require(job.status == JobStatus.Posted || job.status == JobStatus.Failed, "Cannot withdraw");
        require(isJobTimeout(jobId) || job.status == JobStatus.Failed, "Job still active");
        
        uint256 bounty = job.bounty;
        job.bounty = 0;
        
        if (job.paymentToken == address(0)) {
            payable(msg.sender).transfer(bounty);
        } else {
            IERC20(job.paymentToken).safeTransfer(msg.sender, bounty);
        }
    }
    
    function claimComputeRewards() external override nonReentrant {
        uint256 rewards = _pendingRewards[msg.sender];
        require(rewards > 0, "No pending rewards");
        
        _pendingRewards[msg.sender] = 0;
        // Note: This function should be called by the reward distributor
        // For now, we just emit the event
        emit ProviderRewardsClaimed(msg.sender, rewards);
    }
    
    function updateProviderScore(address provider, uint256 jobId, bool success) external override onlyRole(ADMIN_ROLE) {
        ProviderStats storage stats = _providerStats[provider];
        if (success) {
            stats.reputation += 10;
        } else {
            stats.reputation = stats.reputation > 5 ? stats.reputation - 5 : 0;
        }
    }
    
    // Admin Functions
    function setZKVerifier(address verifier) external override onlyRole(ADMIN_ROLE) {
        _zkVerifier = verifier;
        emit ZKVerifierUpdated(verifier);
    }
    
    function setJobTimeout(uint256 defaultTimeout) external override onlyRole(ADMIN_ROLE) {
        _defaultTimeout = defaultTimeout;
    }
    
    function setMinBounty(uint256 amount) external override onlyRole(ADMIN_ROLE) {
        _minBounty = amount;
    }
    
    function setSupportedModelTypes(string[] calldata modelTypes) external override onlyRole(ADMIN_ROLE) {
        // TODO: Implement model type validation
    }
    
    function setSupportedTokens(address[] calldata tokens) external override onlyRole(ADMIN_ROLE) {
        // Clear existing tokens
        for (uint256 i = 0; i < _supportedTokensList.length; i++) {
            _supportedTokens[_supportedTokensList[i]] = false;
        }
        
        // Set new tokens
        delete _supportedTokensList;
        for (uint256 i = 0; i < tokens.length; i++) {
            _supportedTokens[tokens[i]] = true;
            _supportedTokensList.push(tokens[i]);
        }
        
        emit SupportedTokensUpdated(tokens);
    }
    
    // Pausable functions
    function pause() external onlyRole(ADMIN_ROLE) {
        _pause();
    }
    
    function unpause() external onlyRole(ADMIN_ROLE) {
        _unpause();
    }
    
    // View Functions
    function getJob(uint256 jobId) external view override validJob(jobId) returns (Job memory) {
        return _jobs[jobId];
    }
    
    function getProviderStats(address provider) external view override returns (ProviderStats memory) {
        return _providerStats[provider];
    }
    
    function getActiveJobs() external view override returns (uint256[] memory) {
        uint256[] memory activeJobs = new uint256[](_jobCounter);
        uint256 count = 0;
        
        for (uint256 i = 1; i <= _jobCounter; i++) {
            if (_jobs[i].status == JobStatus.Posted || _jobs[i].status == JobStatus.Claimed) {
                activeJobs[count] = i;
                count++;
            }
        }
        
        // Resize array
        uint256[] memory result = new uint256[](count);
        for (uint256 i = 0; i < count; i++) {
            result[i] = activeJobs[i];
        }
        
        return result;
    }
    
    function getProviderJobs(address provider) external view override returns (uint256[] memory) {
        uint256[] memory providerJobs = new uint256[](_jobCounter);
        uint256 count = 0;
        
        for (uint256 i = 1; i <= _jobCounter; i++) {
            if (_jobs[i].provider == provider) {
                providerJobs[count] = i;
                count++;
            }
        }
        
        // Resize array
        uint256[] memory result = new uint256[](count);
        for (uint256 i = 0; i < count; i++) {
            result[i] = providerJobs[i];
        }
        
        return result;
    }
    
    function getRequesterJobs(address requester) external view override returns (uint256[] memory) {
        uint256[] memory requesterJobs = new uint256[](_jobCounter);
        uint256 count = 0;
        
        for (uint256 i = 1; i <= _jobCounter; i++) {
            if (_jobs[i].requester == requester) {
                requesterJobs[count] = i;
                count++;
            }
        }
        
        // Resize array
        uint256[] memory result = new uint256[](count);
        for (uint256 i = 0; i < count; i++) {
            result[i] = requesterJobs[i];
        }
        
        return result;
    }
    
    function getPendingRewards(address provider) external view override returns (uint256) {
        return _pendingRewards[provider];
    }
    
    function getTotalJobCount() external view override returns (uint256) {
        return _jobCounter;
    }
    
    function isJobTimeout(uint256 jobId) public view override validJob(jobId) returns (bool) {
        Job storage job = _jobs[jobId];
        return block.timestamp > job.postedAt + job.timeout;
    }
    
    function getSupportedTokens() external view override returns (address[] memory) {
        return _supportedTokensList;
    }
    
    function isTokenSupported(address token) external view override returns (bool) {
        return _supportedTokens[token];
    }
    
    function getBatchInputs(uint256 jobId) external view override validJob(jobId) returns (BatchInput[] memory) {
        return _jobs[jobId].batchInputs;
    }
} 