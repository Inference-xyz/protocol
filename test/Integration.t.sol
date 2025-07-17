// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../contracts/ComputeMarketplace.sol";
import "../contracts/EZKLVerifier.sol";
import "../contracts/InferenceERC20.sol";

contract IntegrationTest is Test {
    ComputeMarketplace public marketplace;
    EZKLVerifier public zkVerifier;
    InferenceERC20 public token;
    
    address public owner;
    address public client;
    address public provider;
    
    function setUp() public {
        owner = address(this);
        client = address(0x201);
        provider = address(0x202);
        
        // Deploy system
        zkVerifier = new EZKLVerifier(owner);
        marketplace = new ComputeMarketplace(owner, address(zkVerifier));
        token = new InferenceERC20("Inference Token", "INF", 18, owner);
        
        // Configure marketplace
        address[] memory tokens = new address[](1);
        tokens[0] = address(token);
        marketplace.setSupportedTokens(tokens);
        
        // Register a model
        bytes32 modelHash = keccak256("test_model");
        marketplace.registerModelHash(modelHash);
        zkVerifier.registerModel(modelHash, "ipfs://test-model");
        
        // Fund users
        deal(client, 10 ether);
        deal(provider, 10 ether);
        token.mint(client, 1000 ether);
    }
    
    function testCompleteWorkflow() public {
        bytes32 modelHash = keccak256("test_model");
        bytes32 inputHash1 = keccak256("input1");
        bytes32 inputHash2 = keccak256("input2");
        bytes memory encryptedInputs = hex"abcdef1234567890";
        bytes memory encryptedOutput = hex"fedcba0987654321";
        bytes memory zkProof = hex"deadbeef12345678deadbeef12345678deadbeef12345678deadbeef12345678";
        uint256 bounty = 1 ether;
        uint256 timeout = 1 hours;
        
        // 1. Client posts job
        vm.prank(client);
        uint256 jobId = marketplace.postJob{value: bounty}(
            modelHash,
            inputHash1,
            inputHash2,
            encryptedInputs,
            bounty,
            address(0),
            timeout
        );
        
        assertEq(jobId, 1);
        
        // 2. Provider claims job
        vm.prank(provider);
        marketplace.claimJob(jobId);
        
        // 3. Provider completes job
        vm.prank(provider);
        marketplace.completeJob(jobId, encryptedOutput, zkProof);
        
        // 4. Verify final state
        IComputeMarketplace.Job memory job = marketplace.getJob(jobId);
        assertTrue(job.completed);
        assertEq(job.provider, provider);
        
        // 5. Check provider stats
        IComputeMarketplace.ProviderStats memory stats = marketplace.getProviderStats(provider);
        assertEq(stats.totalJobs, 1);
        assertEq(stats.completedJobs, 1);
        assertEq(stats.totalEarnings, bounty);
        assertEq(stats.reputationScore, 100);
        
        console.log("Complete workflow test passed!");
        console.log("   Job ID:", jobId);
        console.log("   Provider earnings:", stats.totalEarnings);
        console.log("   Reputation score:", stats.reputationScore);
    }
} 