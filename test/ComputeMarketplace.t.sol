// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../contracts/ComputeMarketplace.sol";
import "../contracts/InfToken.sol";
import "../contracts/Verifier.sol";

contract ComputeMarketplaceTest is Test {
    ComputeMarketplace public marketplace;
    InfToken public token;
    Verifier public verifier;
    
    address public client = address(0x101);
    address public provider = address(0x102);
    
    bytes32 public constant MODEL_HASH = keccak256("test_model");
    bytes32[] public inputHashes;
    uint256 public constant PAYMENT_AMOUNT = 100 ether;
    
    function setUp() public {
        verifier = new Verifier();
        token = new InfToken();
        marketplace = new ComputeMarketplace(address(verifier));
        
        inputHashes.push(keccak256("input1"));
        
        token.transfer(client, PAYMENT_AMOUNT * 10);
        
        vm.prank(client);
        token.approve(address(marketplace), PAYMENT_AMOUNT * 10);
    }
    
    function testRegisterModelHash() public {
        marketplace.registerModelHash(MODEL_HASH);
        assertTrue(marketplace.registeredModelHashes(MODEL_HASH));
    }
    
    function testPostJob() public {
        marketplace.registerModelHash(MODEL_HASH);
        
        vm.prank(client);
        uint256 jobId = marketplace.postJob(MODEL_HASH, inputHashes, PAYMENT_AMOUNT, address(token));
        
        assertEq(jobId, 1);
        assertEq(marketplace.nextJobId(), 2);
    }
    
    function testClaimAndCompleteJob() public {
        marketplace.registerModelHash(MODEL_HASH);
        
        vm.prank(client);
        uint256 jobId = marketplace.postJob(MODEL_HASH, inputHashes, PAYMENT_AMOUNT, address(token));
        
        vm.prank(provider);
        marketplace.claimJob(jobId);
        
        bytes memory encryptedOutput = "encrypted_output";
        bytes memory zkProof = "zk_proof";
        uint256[] memory publicInputs = new uint256[](1);
        publicInputs[0] = uint256(keccak256("public_input"));
        
        uint256 providerBalanceBefore = token.balanceOf(provider);
        
        vm.prank(provider);
        marketplace.completeJob(jobId, encryptedOutput, zkProof, publicInputs);
        
        uint256 providerBalanceAfter = token.balanceOf(provider);
        assertEq(providerBalanceAfter, providerBalanceBefore + PAYMENT_AMOUNT);
    }
} 