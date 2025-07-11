// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IComputeMarketplace {
    enum JobStatus {
        Posted,
        Claimed,
        Completed,
        Failed,
        Disputed
    }
    
    struct Job {
        uint256 id;
        address requester;
        address provider;
        uint256 bounty;
        address paymentToken;    // ERC20 token for payment
        string specURI;          // ONNX model specification URI
        JobStatus status;
        uint256 postedAt;
        uint256 claimedAt;
        uint256 completedAt;
        uint256 timeout;
        bytes32 resultHash;      // Hash of the computation result
        BatchInput[] batchInputs;
    }
    
    struct BatchInput {
        bytes32[] inputs;
        uint256 batchId;
        bool processed;
        uint256 processingFee;
    }
    
    struct ProviderStats {
        uint256 totalJobs;
        uint256 completedJobs;
        uint256 failedJobs;
        uint256 totalEarnings;
        uint256 reputation;      // Reputation score
        uint256 lastActive;
    }
    
    // Core Job Functions
    function postJob(string calldata specURI, uint256 bounty, address paymentToken, uint256 timeout) external payable returns (uint256 jobId);
    function claimJob(uint256 jobId) external;
    function completeJob(uint256 jobId, string calldata resultURI, bytes32 resultHash, bytes calldata zkProof) external; // Submit result with EZKL ZK proof
    function disputeJob(uint256 jobId, string calldata reason) external;
    function resolveDispute(uint256 jobId, bool favorProvider) external;
    
    function batchSubmitInputs(uint256 jobId, bytes32[] calldata inputs, uint256 processingFee) external returns (uint256 batchId);
    function processBatch(uint256 jobId, uint256 batchId) external;
    function getBatchInputs(uint256 jobId) external view returns (BatchInput[] memory);
    
    function payWithERC20(address token, uint256 amount) external;
    function getPaymentBalance(address token) external view returns (uint256);
    function withdrawPayment(address token, uint256 amount) external;
    
    // Payment and Reward Functions
    function withdrawBounty(uint256 jobId) external;
    function claimComputeRewards() external;
    function updateProviderScore(address provider, uint256 jobId, bool success) external;
    
    // Admin Functions
    function setZKVerifier(address verifier) external;                                                                      // Set EZKL verifier contract
    function setJobTimeout(uint256 defaultTimeout) external;
    function setMinBounty(uint256 amount) external;
    function setSupportedModelTypes(string[] calldata modelTypes) external;                                                 // Set supported ONNX model types
    function setSupportedTokens(address[] calldata tokens) external;                                                        // Set supported ERC20 tokens
    
    // View Functions
    function getJob(uint256 jobId) external view returns (Job memory);
    function getProviderStats(address provider) external view returns (ProviderStats memory);
    function getActiveJobs() external view returns (uint256[] memory);
    function getProviderJobs(address provider) external view returns (uint256[] memory);
    function getRequesterJobs(address requester) external view returns (uint256[] memory);
    function getPendingRewards(address provider) external view returns (uint256);
    function getTotalJobCount() external view returns (uint256);
    function isJobTimeout(uint256 jobId) external view returns (bool);
    function getSupportedTokens() external view returns (address[] memory);
    function isTokenSupported(address token) external view returns (bool);
    
    // Events
    event JobPosted(uint256 indexed jobId, address indexed requester, uint256 bounty, address paymentToken, string specURI, uint256 timeout);
    event JobClaimed(uint256 indexed jobId, address indexed provider, uint256 claimedAt);
    event JobCompleted(uint256 indexed jobId, address indexed provider, string resultURI, bytes32 resultHash);
    event JobFailed(uint256 indexed jobId, address indexed provider, string reason);
    event JobDisputed(uint256 indexed jobId, address indexed requester, string reason);
    event DisputeResolved(uint256 indexed jobId, bool favorProvider, address resolver);
    event ProviderRewardsClaimed(address indexed provider, uint256 amount);
    event ZKVerifierUpdated(address indexed newVerifier);
    event ZKProofVerified(uint256 indexed jobId, bool success, uint256 gasUsed);
    
    // New Events
    event BatchInputsSubmitted(uint256 indexed jobId, uint256 indexed batchId, bytes32[] inputs, uint256 processingFee);
    event BatchProcessed(uint256 indexed jobId, uint256 indexed batchId);
    event PaymentReceived(address indexed token, uint256 amount, address indexed payer);
    event PaymentWithdrawn(address indexed token, uint256 amount, address indexed recipient);
    event SupportedTokensUpdated(address[] tokens);
} 