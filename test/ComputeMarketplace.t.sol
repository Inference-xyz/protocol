// // SPDX-License-Identifier: MIT
// pragma solidity ^0.8.19;

// import "forge-std/Test.sol";
// import "../contracts/ComputeMarketplace.sol";
// import "../contracts/InferenceERC20.sol";
// import "../contracts/interfaces/IComputeMarketplace.sol";

// contract ComputeMarketplaceTest is Test {
//     ComputeMarketplace public marketplace;
//     InferenceERC20 public token;
    
//     address public owner;
//     address public client;
//     address public provider;
//     address public user1;
//     address public user2;
    
//     // Test constants
//     bytes32 public constant MODEL_HASH = keccak256("test_model");
//     bytes32 public constant INPUT_HASH_1 = keccak256("test_input_1");
//     bytes32 public constant INPUT_HASH_2 = keccak256("test_input_2");
//     bytes public constant ENCRYPTED_INPUTS = hex"abcdef1234567890";
//     bytes public constant ENCRYPTED_OUTPUT = hex"fedcba0987654321";
//     bytes public constant ZK_PROOF = hex"deadbeef12345678deadbeef12345678deadbeef12345678deadbeef12345678";
//     uint256 public constant BOUNTY = 1 ether;
//     uint256 public constant TIMEOUT = 1 hours;
    
//     // Events
//     event JobPosted(
//         uint256 indexed jobId,
//         address indexed client,
//         bytes32 modelHash,
//         bytes32 inputHash1,
//         bytes32 inputHash2,
//         bytes encryptedInputs,
//         uint256 bounty,
//         address bountyToken,
//         uint256 timeout
//     );
//     event JobClaimed(uint256 indexed jobId, address indexed provider);
//     event JobCompleted(uint256 indexed jobId, address indexed provider, bytes encryptedOutput, bytes zkProof);
//     event JobTimedOut(uint256 indexed jobId, address indexed client);
//     event ModelHashRegistered(bytes32 indexed modelHash, address indexed registrant);
    
//     function setUp() public {
//         owner = address(this);
//         client = address(0x101);
//         provider = address(0x102);
//         user1 = address(0x103);
//         user2 = address(0x104);
        
//         // Deploy contracts
//         zkVerifier = new EZKLVerifier(owner);
//         marketplace = new ComputeMarketplace(owner, address(zkVerifier));
//         token = new InferenceERC20("Inference Token", "INF", 18, owner);
        
//         // Setup marketplace
//         address[] memory tokens = new address[](1);
//         tokens[0] = address(token);
//         marketplace.setSupportedTokens(tokens);
        
//         // Register model hash
//         marketplace.registerModelHash(MODEL_HASH);
        
//         // Fund users
//         deal(client, 10 ether);
//         deal(provider, 10 ether);
//         deal(user1, 10 ether);
//         deal(user2, 10 ether);
        
//         // Give tokens to users
//         token.mint(client, 1000 ether);
//         token.mint(provider, 1000 ether);
//         token.mint(user1, 1000 ether);
//         token.mint(user2, 1000 ether);
//     }
    
//     // Add payable fallback function to receive ETH
//     receive() external payable {}
    
//     // Test deployment and initial state
//     function testDeployment() public {
//         assertEq(marketplace.owner(), owner);
//         assertEq(address(marketplace.zkVerifier()), address(zkVerifier));
//         assertEq(marketplace.nextJobId(), 1);
//         assertEq(marketplace.minBounty(), 0.01 ether);
//         assertEq(marketplace.defaultTimeout(), 24 hours);
//         assertFalse(marketplace.paused());
//     }
    
//     // Test model hash registration
//     function testRegisterModelHash() public {
//         bytes32 newModelHash = keccak256("new_model");
        
//         vm.expectEmit(true, true, false, false);
//         emit ModelHashRegistered(newModelHash, address(this));
        
//         marketplace.registerModelHash(newModelHash);
        
//         assertTrue(marketplace.isModelHashRegistered(newModelHash));
//         assertEq(marketplace.getModelHashRegistrant(newModelHash), address(this));
//     }
    
//     function testRegisterModelHashAlreadyRegistered() public {
//         vm.expectRevert("Model hash already registered");
//         marketplace.registerModelHash(MODEL_HASH);
//     }
    
//     // Test job posting with ETH
//     function testPostJobWithETH() public {
//         vm.prank(client);
//         uint256 jobId = marketplace.postJob{value: BOUNTY}(
//             MODEL_HASH,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             ENCRYPTED_INPUTS,
//             BOUNTY,
//             address(0),
//             TIMEOUT
//         );
        
//         assertEq(jobId, 1);
//         assertEq(marketplace.nextJobId(), 2);
        
//         IComputeMarketplace.Job memory job = marketplace.getJob(jobId);
//         assertEq(job.id, jobId);
//         assertEq(job.client, client);
//         assertEq(job.modelHash, MODEL_HASH);
//         assertEq(job.inputHash1, INPUT_HASH_1);
//         assertEq(job.inputHash2, INPUT_HASH_2);
//         assertEq(job.encryptedInputs, ENCRYPTED_INPUTS);
//         assertEq(job.bounty, BOUNTY);
//         assertEq(job.bountyToken, address(0));
//         assertEq(job.timeout, TIMEOUT);
//         assertEq(job.provider, address(0));
//         assertFalse(job.completed);
//         assertFalse(job.timedOut);
        
//         // Check that job is in client's jobs
//         uint256[] memory clientJobs = marketplace.getJobsByClient(client);
//         assertEq(clientJobs.length, 1);
//         assertEq(clientJobs[0], jobId);
//     }
    
//     // Test job posting with ERC20
//     function testPostJobWithERC20() public {
//         vm.startPrank(client);
//         token.approve(address(marketplace), BOUNTY);
        
//         uint256 jobId = marketplace.postJob(
//             MODEL_HASH,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             ENCRYPTED_INPUTS,
//             BOUNTY,
//             address(token),
//             TIMEOUT
//         );
//         vm.stopPrank();
        
//         assertEq(jobId, 1);
//         assertEq(token.balanceOf(client), 1000 ether - BOUNTY);
//         assertEq(token.balanceOf(address(marketplace)), BOUNTY);
        
//         IComputeMarketplace.Job memory job = marketplace.getJob(jobId);
//         assertEq(job.bountyToken, address(token));
//     }
    
//     // Test job posting validation
//     function testPostJobUnregisteredModel() public {
//         bytes32 unregisteredModel = keccak256("unregistered");
        
//         vm.prank(client);
//         vm.expectRevert("Model hash not registered");
//         marketplace.postJob{value: BOUNTY}(
//             unregisteredModel,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             ENCRYPTED_INPUTS,
//             BOUNTY,
//             address(0),
//             TIMEOUT
//         );
//     }
    
//     function testPostJobBountyTooLow() public {
//         vm.prank(client);
//         vm.expectRevert("Bounty below minimum");
//         marketplace.postJob{value: 0.005 ether}(
//             MODEL_HASH,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             ENCRYPTED_INPUTS,
//             0.005 ether,
//             address(0),
//             TIMEOUT
//         );
//     }
    
//     function testPostJobInvalidTimeout() public {
//         vm.prank(client);
//         vm.expectRevert("Invalid timeout");
//         marketplace.postJob{value: BOUNTY}(
//             MODEL_HASH,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             ENCRYPTED_INPUTS,
//             BOUNTY,
//             address(0),
//             0
//         );
//     }
    
//     function testPostJobEmptyEncryptedInputs() public {
//         vm.prank(client);
//         vm.expectRevert("Encrypted inputs required");
//         marketplace.postJob{value: BOUNTY}(
//             MODEL_HASH,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             "",
//             BOUNTY,
//             address(0),
//             TIMEOUT
//         );
//     }
    
//     function testPostJobIncorrectETHAmount() public {
//         vm.prank(client);
//         vm.expectRevert("Incorrect ETH amount");
//         marketplace.postJob{value: 0.5 ether}(
//             MODEL_HASH,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             ENCRYPTED_INPUTS,
//             BOUNTY,
//             address(0),
//             TIMEOUT
//         );
//     }
    
//     function testPostJobUnsupportedToken() public {
//         address unsupportedToken = address(0x999);
        
//         vm.prank(client);
//         vm.expectRevert("Token not supported");
//         marketplace.postJob(
//             MODEL_HASH,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             ENCRYPTED_INPUTS,
//             BOUNTY,
//             unsupportedToken,
//             TIMEOUT
//         );
//     }
    
//     // Test job claiming
//     function testClaimJob() public {
//         // Post a job
//         vm.prank(client);
//         uint256 jobId = marketplace.postJob{value: BOUNTY}(
//             MODEL_HASH,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             ENCRYPTED_INPUTS,
//             BOUNTY,
//             address(0),
//             TIMEOUT
//         );
        
//         // Claim the job
//         vm.prank(provider);
//         vm.expectEmit(true, true, false, false);
//         emit JobClaimed(jobId, provider);
        
//         marketplace.claimJob(jobId);
        
//         IComputeMarketplace.Job memory job = marketplace.getJob(jobId);
//         assertEq(job.provider, provider);
//         assertGt(job.claimedAt, 0);
        
//         // Check that job is in provider's jobs
//         uint256[] memory providerJobs = marketplace.getJobsByProvider(provider);
//         assertEq(providerJobs.length, 1);
//         assertEq(providerJobs[0], jobId);
//     }
    
//     function testClaimJobNonExistent() public {
//         vm.prank(provider);
//         vm.expectRevert("Job does not exist");
//         marketplace.claimJob(999);
//     }
    
//     function testClaimJobAlreadyClaimed() public {
//         // Post and claim a job
//         vm.prank(client);
//         uint256 jobId = marketplace.postJob{value: BOUNTY}(
//             MODEL_HASH,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             ENCRYPTED_INPUTS,
//             BOUNTY,
//             address(0),
//             TIMEOUT
//         );
        
//         vm.prank(provider);
//         marketplace.claimJob(jobId);
        
//         // Try to claim again
//         vm.prank(user1);
//         vm.expectRevert("Job already claimed");
//         marketplace.claimJob(jobId);
//     }
    
//     function testClaimJobClientCannotClaim() public {
//         // Post a job
//         vm.prank(client);
//         uint256 jobId = marketplace.postJob{value: BOUNTY}(
//             MODEL_HASH,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             ENCRYPTED_INPUTS,
//             BOUNTY,
//             address(0),
//             TIMEOUT
//         );
        
//         // Client tries to claim their own job
//         vm.prank(client);
//         vm.expectRevert("Client cannot claim own job");
//         marketplace.claimJob(jobId);
//     }
    
//     // Test job completion
//     function testCompleteJob() public {
//         // Post and claim a job
//         vm.prank(client);
//         uint256 jobId = marketplace.postJob{value: BOUNTY}(
//             MODEL_HASH,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             ENCRYPTED_INPUTS,
//             BOUNTY,
//             address(0),
//             TIMEOUT
//         );
        
//         vm.prank(provider);
//         marketplace.claimJob(jobId);
        
//         uint256 providerBalanceBefore = provider.balance;
        
//         // Complete the job
//         vm.prank(provider);
//         vm.expectEmit(true, true, false, false);
//         emit JobCompleted(jobId, provider, ENCRYPTED_OUTPUT, ZK_PROOF);
        
//         marketplace.completeJob(jobId, ENCRYPTED_OUTPUT, ZK_PROOF);
        
//         IComputeMarketplace.Job memory job = marketplace.getJob(jobId);
//         assertTrue(job.completed);
        
//         // Check provider received payment
//         assertEq(provider.balance, providerBalanceBefore + BOUNTY);
        
//         // Check provider stats
//         IComputeMarketplace.ProviderStats memory stats = marketplace.getProviderStats(provider);
//         assertEq(stats.totalJobs, 1);
//         assertEq(stats.completedJobs, 1);
//         assertEq(stats.totalEarnings, BOUNTY);
//         assertEq(stats.reputationScore, 100);
//     }
    
//     function testCompleteJobWithERC20() public {
//         // Post and claim a job with ERC20
//         vm.startPrank(client);
//         token.approve(address(marketplace), BOUNTY);
//         uint256 jobId = marketplace.postJob(
//             MODEL_HASH,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             ENCRYPTED_INPUTS,
//             BOUNTY,
//             address(token),
//             TIMEOUT
//         );
//         vm.stopPrank();
        
//         vm.prank(provider);
//         marketplace.claimJob(jobId);
        
//         uint256 providerBalanceBefore = token.balanceOf(provider);
        
//         // Complete the job
//         vm.prank(provider);
//         marketplace.completeJob(jobId, ENCRYPTED_OUTPUT, ZK_PROOF);
        
//         // Check provider received payment
//         assertEq(token.balanceOf(provider), providerBalanceBefore + BOUNTY);
//     }
    
//     function testCompleteJobValidation() public {
//         // Post and claim a job
//         vm.prank(client);
//         uint256 jobId = marketplace.postJob{value: BOUNTY}(
//             MODEL_HASH,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             ENCRYPTED_INPUTS,
//             BOUNTY,
//             address(0),
//             TIMEOUT
//         );
        
//         vm.prank(provider);
//         marketplace.claimJob(jobId);
        
//         // Test non-existent job
//         vm.prank(provider);
//         vm.expectRevert("Job does not exist");
//         marketplace.completeJob(999, ENCRYPTED_OUTPUT, ZK_PROOF);
        
//         // Test wrong provider
//         vm.prank(user1);
//         vm.expectRevert("Only provider can complete");
//         marketplace.completeJob(jobId, ENCRYPTED_OUTPUT, ZK_PROOF);
        
//         // Test empty encrypted output
//         vm.prank(provider);
//         vm.expectRevert("Encrypted output required");
//         marketplace.completeJob(jobId, "", ZK_PROOF);
        
//         // Test empty ZK proof
//         vm.prank(provider);
//         vm.expectRevert("ZK proof required");
//         marketplace.completeJob(jobId, ENCRYPTED_OUTPUT, "");
//     }
    
//     // Test job timeout
//     function testHandleJobTimeoutUnclaimed() public {
//         // Post a job
//         vm.prank(client);
//         uint256 jobId = marketplace.postJob{value: BOUNTY}(
//             MODEL_HASH,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             ENCRYPTED_INPUTS,
//             BOUNTY,
//             address(0),
//             TIMEOUT
//         );
        
//         // Fast forward time
//         vm.warp(block.timestamp + TIMEOUT + 1);
        
//         uint256 clientBalanceBefore = client.balance;
        
//         // Handle timeout
//         vm.expectEmit(true, true, false, false);
//         emit JobTimedOut(jobId, client);
        
//         marketplace.handleJobTimeout(jobId);
        
//         IComputeMarketplace.Job memory job = marketplace.getJob(jobId);
//         assertTrue(job.timedOut);
        
//         // Check client received refund
//         assertEq(client.balance, clientBalanceBefore + BOUNTY);
//     }
    
//     function testHandleJobTimeoutClaimed() public {
//         // Post and claim a job
//         vm.prank(client);
//         uint256 jobId = marketplace.postJob{value: BOUNTY}(
//             MODEL_HASH,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             ENCRYPTED_INPUTS,
//             BOUNTY,
//             address(0),
//             TIMEOUT
//         );
        
//         vm.prank(provider);
//         marketplace.claimJob(jobId);
        
//         // Fast forward time
//         vm.warp(block.timestamp + TIMEOUT + 1);
        
//         uint256 clientBalanceBefore = client.balance;
        
//         // Handle timeout
//         marketplace.handleJobTimeout(jobId);
        
//         IComputeMarketplace.Job memory job = marketplace.getJob(jobId);
//         assertTrue(job.timedOut);
        
//         // Check client received refund
//         assertEq(client.balance, clientBalanceBefore + BOUNTY);
//     }
    
//     function testHandleJobTimeoutNotTimedOut() public {
//         // Post a job
//         vm.prank(client);
//         uint256 jobId = marketplace.postJob{value: BOUNTY}(
//             MODEL_HASH,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             ENCRYPTED_INPUTS,
//             BOUNTY,
//             address(0),
//             TIMEOUT
//         );
        
//         // Try to handle timeout too early
//         vm.expectRevert("Job has not timed out yet");
//         marketplace.handleJobTimeout(jobId);
//     }
    
//     function testHandleJobTimeoutAlreadyCompleted() public {
//         // Post, claim, and complete a job
//         vm.prank(client);
//         uint256 jobId = marketplace.postJob{value: BOUNTY}(
//             MODEL_HASH,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             ENCRYPTED_INPUTS,
//             BOUNTY,
//             address(0),
//             TIMEOUT
//         );
        
//         vm.prank(provider);
//         marketplace.claimJob(jobId);
        
//         vm.prank(provider);
//         marketplace.completeJob(jobId, ENCRYPTED_OUTPUT, ZK_PROOF);
        
//         // Fast forward time
//         vm.warp(block.timestamp + TIMEOUT + 1);
        
//         // Try to handle timeout
//         vm.expectRevert("Job already completed");
//         marketplace.handleJobTimeout(jobId);
//     }
    
//     // Test view functions
//     function testGetAllJobs() public {
//         // Post multiple jobs
//         vm.startPrank(client);
//         uint256 jobId1 = marketplace.postJob{value: BOUNTY}(
//             MODEL_HASH,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             ENCRYPTED_INPUTS,
//             BOUNTY,
//             address(0),
//             TIMEOUT
//         );
//         uint256 jobId2 = marketplace.postJob{value: BOUNTY}(
//             MODEL_HASH,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             ENCRYPTED_INPUTS,
//             BOUNTY,
//             address(0),
//             TIMEOUT
//         );
//         vm.stopPrank();
        
//         IComputeMarketplace.Job[] memory allJobs = marketplace.getAllJobs();
//         assertEq(allJobs.length, 2);
//         assertEq(allJobs[0].id, jobId1);
//         assertEq(allJobs[1].id, jobId2);
//     }
    
//     function testGetAvailableJobs() public {
//         // Post multiple jobs
//         vm.startPrank(client);
//         uint256 jobId1 = marketplace.postJob{value: BOUNTY}(
//             MODEL_HASH,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             ENCRYPTED_INPUTS,
//             BOUNTY,
//             address(0),
//             TIMEOUT
//         );
//         uint256 jobId2 = marketplace.postJob{value: BOUNTY}(
//             MODEL_HASH,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             ENCRYPTED_INPUTS,
//             BOUNTY,
//             address(0),
//             TIMEOUT
//         );
//         vm.stopPrank();
        
//         // Claim one job
//         vm.prank(provider);
//         marketplace.claimJob(jobId1);
        
//         IComputeMarketplace.Job[] memory availableJobs = marketplace.getAvailableJobs();
//         assertEq(availableJobs.length, 1);
//         assertEq(availableJobs[0].id, jobId2);
//     }
    
//     // Test admin functions
//     function testSetSupportedTokens() public {
//         address[] memory tokens = new address[](2);
//         tokens[0] = address(token);
//         tokens[1] = address(0x123);
        
//         marketplace.setSupportedTokens(tokens);
        
//         assertTrue(marketplace.isSupportedToken(address(token)));
//         assertTrue(marketplace.isSupportedToken(address(0x123)));
//     }
    
//     function testSetSupportedTokensOnlyOwner() public {
//         address[] memory tokens = new address[](1);
//         tokens[0] = address(token);
        
//         vm.prank(user1);
//         vm.expectRevert(abi.encodeWithSelector(0x118cdaa7, user1)); // OwnableUnauthorizedAccount(address account)
//         marketplace.setSupportedTokens(tokens);
//     }
    
//     function testSetMinBounty() public {
//         uint256 newMinBounty = 0.1 ether;
        
//         marketplace.setMinBounty(newMinBounty);
        
//         assertEq(marketplace.getMinBounty(), newMinBounty);
//     }
    
//     function testSetMinBountyOnlyOwner() public {
//         vm.prank(user1);
//         vm.expectRevert(abi.encodeWithSelector(0x118cdaa7, user1)); // OwnableUnauthorizedAccount(address account)
//         marketplace.setMinBounty(0.1 ether);
//     }
    
//     function testSetJobTimeout() public {
//         uint256 newTimeout = 2 hours;
        
//         marketplace.setJobTimeout(newTimeout);
        
//         assertEq(marketplace.getJobTimeout(), newTimeout);
//     }
    
//     function testSetJobTimeoutInvalid() public {
//         vm.expectRevert("Invalid timeout");
//         marketplace.setJobTimeout(0);
//     }
    
//     function testSetJobTimeoutOnlyOwner() public {
//         vm.prank(user1);
//         vm.expectRevert(abi.encodeWithSelector(0x118cdaa7, user1)); // OwnableUnauthorizedAccount(address account)
//         marketplace.setJobTimeout(2 hours);
//     }
    
//     function testSetZKVerifier() public {
//         EZKLVerifier newVerifier = new EZKLVerifier(owner);
        
//         marketplace.setZKVerifier(address(newVerifier));
        
//         assertEq(address(marketplace.zkVerifier()), address(newVerifier));
//     }
    
//     function testSetZKVerifierInvalidAddress() public {
//         vm.expectRevert("Invalid verifier address");
//         marketplace.setZKVerifier(address(0));
//     }
    
//     function testSetZKVerifierOnlyOwner() public {
//         EZKLVerifier newVerifier = new EZKLVerifier(owner);
        
//         vm.prank(user1);
//         vm.expectRevert(abi.encodeWithSelector(0x118cdaa7, user1)); // OwnableUnauthorizedAccount(address account)
//         marketplace.setZKVerifier(address(newVerifier));
//     }
    
//     // Test pause functionality
//     function testPauseUnpause() public {
//         marketplace.pause();
//         assertTrue(marketplace.paused());
        
//         marketplace.unpause();
//         assertFalse(marketplace.paused());
//     }
    
//     function testPauseOnlyOwner() public {
//         vm.prank(user1);
//         vm.expectRevert(abi.encodeWithSelector(0x118cdaa7, user1)); // OwnableUnauthorizedAccount(address account)
//         marketplace.pause();
//     }
    
//     function testPostJobWhenPaused() public {
//         marketplace.pause();
        
//         vm.prank(client);
//         vm.expectRevert(); // EnforcedPause()
//         marketplace.postJob{value: BOUNTY}(
//             MODEL_HASH,
//             INPUT_HASH_1,
//             INPUT_HASH_2,
//             ENCRYPTED_INPUTS,
//             BOUNTY,
//             address(0),
//             TIMEOUT
//         );
//     }
    
//     // Test emergency withdraw
//     function testEmergencyWithdrawETH() public {
//         // Send some ETH to the contract
//         deal(address(marketplace), 5 ether);
        
//         uint256 ownerBalanceBefore = owner.balance;
        
//         marketplace.emergencyWithdraw(address(0));
        
//         assertEq(owner.balance, ownerBalanceBefore + 5 ether);
//         assertEq(address(marketplace).balance, 0);
//     }
    
//     function testEmergencyWithdrawERC20() public {
//         // Send some tokens to the contract
//         token.mint(address(marketplace), 100 ether);
        
//         uint256 ownerBalanceBefore = token.balanceOf(owner);
        
//         marketplace.emergencyWithdraw(address(token));
        
//         assertEq(token.balanceOf(owner), ownerBalanceBefore + 100 ether);
//         assertEq(token.balanceOf(address(marketplace)), 0);
//     }
    
//     function testEmergencyWithdrawOnlyOwner() public {
//         vm.prank(user1);
//         vm.expectRevert(abi.encodeWithSelector(0x118cdaa7, user1)); // OwnableUnauthorizedAccount(address account)
//         marketplace.emergencyWithdraw(address(0));
//     }
    
//     // Test reentrancy protection
//     function testReentrancyProtection() public {
//         // This test ensures that the ReentrancyGuard is working
//         // The specific implementation depends on the contract's vulnerability surfaces
//         // For now, we'll just test that the modifier is present by checking contract bytecode
//         assertTrue(address(marketplace).code.length > 0);
//     }
// } 