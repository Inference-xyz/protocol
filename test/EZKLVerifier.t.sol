// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../contracts/EZKLVerifier.sol";
import "../contracts/interfaces/IZKVerifier.sol";

contract EZKLVerifierTest is Test {
    EZKLVerifier public verifier;
    address public owner;
    address public user1;
    address public user2;
    
    // Test data
    bytes32 public constant MODEL_HASH = keccak256("test_model");
    bytes32 public constant SCORING_MODEL_HASH = keccak256("test_scoring_model");
    string public constant METADATA_URI = "ipfs://test-metadata";
    bytes public constant VERIFYING_KEY = hex"1234567890abcdef";
    bytes public constant PROOF_DATA = hex"abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890";
    
    event ModelRegistered(bytes32 indexed modelHash, string metadataURI, IZKVerifier.ProofType proofType);
    event VerifyingKeySet(bytes32 indexed modelHash, IZKVerifier.ProofType proofType);
    event TrustedProverUpdated(address indexed prover, bool trusted, IZKVerifier.ProofType proofType);
    event VerificationFeeUpdated(bytes32 indexed modelHash, uint256 fee, IZKVerifier.ProofType proofType);
    event ProofVerified(address indexed verifier, bytes32 indexed modelHash, bool success, uint256 gasUsed, IZKVerifier.ProofType proofType);
    event BatchVerificationCompleted(uint256 proofCount, uint256 successCount, IZKVerifier.ProofType proofType);
    
    function setUp() public {
        owner = address(this);
        user1 = address(0x101);
        user2 = address(0x102);
        
        verifier = new EZKLVerifier(owner);
    }
    
    // Test basic deployment
    function testDeployment() public {
        assertEq(verifier.owner(), owner);
        // Basic deployment test - removed configuration methods for simplified contract
    }
    
    // Test model registration
    function testRegisterModel() public {
        vm.expectEmit(true, false, false, true);
        emit ModelRegistered(MODEL_HASH, METADATA_URI, IZKVerifier.ProofType.Inference);
        
        verifier.registerModel(MODEL_HASH, METADATA_URI);
        
        assertTrue(verifier.isModelRegistered(MODEL_HASH));
        assertEq(verifier.getModelMetadata(MODEL_HASH), METADATA_URI);
        
        bytes32[] memory supportedModels = verifier.getSupportedModels();
        assertEq(supportedModels.length, 1);
        assertEq(supportedModels[0], MODEL_HASH);
    }
    
    function testRegisterModelAlreadyRegistered() public {
        verifier.registerModel(MODEL_HASH, METADATA_URI);
        
        vm.expectRevert("Model already registered");
        verifier.registerModel(MODEL_HASH, METADATA_URI);
    }
    
    function testRegisterScoringModel() public {
        vm.expectEmit(true, false, false, true);
        emit ModelRegistered(SCORING_MODEL_HASH, METADATA_URI, IZKVerifier.ProofType.Scoring);
        
        verifier.registerScoringModel(SCORING_MODEL_HASH, METADATA_URI);
        
        assertTrue(verifier.isScoringModelRegistered(SCORING_MODEL_HASH));
        assertEq(verifier.getScoringModelMetadata(SCORING_MODEL_HASH), METADATA_URI);
        
        bytes32[] memory supportedScoringModels = verifier.getSupportedScoringModels();
        assertEq(supportedScoringModels.length, 1);
        assertEq(supportedScoringModels[0], SCORING_MODEL_HASH);
    }
    
    function testRegisterScoringModelAlreadyRegistered() public {
        verifier.registerScoringModel(SCORING_MODEL_HASH, METADATA_URI);
        
        vm.expectRevert("Scoring model already registered");
        verifier.registerScoringModel(SCORING_MODEL_HASH, METADATA_URI);
    }
    
    // Test verifying key management
    function testSetVerifyingKey() public {
        verifier.registerModel(MODEL_HASH, METADATA_URI);
        
        vm.expectEmit(true, false, false, true);
        emit VerifyingKeySet(MODEL_HASH, IZKVerifier.ProofType.Inference);
        
        verifier.setVerifyingKey(MODEL_HASH, VERIFYING_KEY);
        
        assertEq(verifier.getVerifyingKey(MODEL_HASH), VERIFYING_KEY);
        assertTrue(verifier.hasVerifyingKey(MODEL_HASH));
    }
    
    function testSetVerifyingKeyModelNotRegistered() public {
        vm.expectRevert("Model not registered");
        verifier.setVerifyingKey(MODEL_HASH, VERIFYING_KEY);
    }
    
    function testSetVerifyingKeyOnlyOwner() public {
        verifier.registerModel(MODEL_HASH, METADATA_URI);
        
        vm.prank(user1);
        vm.expectRevert(abi.encodeWithSelector(0x118cdaa7, user1)); // OwnableUnauthorizedAccount(address account)
        verifier.setVerifyingKey(MODEL_HASH, VERIFYING_KEY);
    }
    
    function testSetScoringVerifyingKey() public {
        verifier.registerScoringModel(SCORING_MODEL_HASH, METADATA_URI);
        
        vm.expectEmit(true, false, false, true);
        emit VerifyingKeySet(SCORING_MODEL_HASH, IZKVerifier.ProofType.Scoring);
        
        verifier.setScoringVerifyingKey(SCORING_MODEL_HASH, VERIFYING_KEY);
        
        assertEq(verifier.getScoringVerifyingKey(SCORING_MODEL_HASH), VERIFYING_KEY);
        assertTrue(verifier.hasScoringVerifyingKey(SCORING_MODEL_HASH));
    }
    
    function testSetScoringVerifyingKeyModelNotRegistered() public {
        vm.expectRevert("Scoring model not registered");
        verifier.setScoringVerifyingKey(SCORING_MODEL_HASH, VERIFYING_KEY);
    }
    
    // Test proof verification
    function testVerifyProof() public {
        bytes32[] memory publicInputs = new bytes32[](2);
        publicInputs[0] = bytes32(uint256(1));
        publicInputs[1] = bytes32(uint256(2));
        
        bool result = verifier.verifyProof(PROOF_DATA, publicInputs);
        assertTrue(result);
    }
    
    function testVerifyProofEmptyProof() public {
        bytes32[] memory publicInputs = new bytes32[](0);
        bytes memory emptyProof = "";
        
        vm.expectRevert("Empty proof");
        verifier.verifyProof(emptyProof, publicInputs);
    }
    
    function testVerifyProofTooLarge() public {
        bytes32[] memory publicInputs = new bytes32[](0);
        bytes memory largeProof = new bytes(2 * 1024 * 1024); // 2MB
        
        vm.expectRevert("Proof too large");
        verifier.verifyProof(largeProof, publicInputs);
    }
    
    function testVerifyProofTooManyPublicInputs() public {
        bytes32[] memory publicInputs = new bytes32[](300); // More than max of 256
        
        vm.expectRevert("Too many public inputs");
        verifier.verifyProof(PROOF_DATA, publicInputs);
    }
    
    function testVerifyProofWithMetadata() public {
        bytes32[] memory publicInputs = new bytes32[](2);
        publicInputs[0] = bytes32(uint256(1));
        publicInputs[1] = bytes32(uint256(2));
        
        IZKVerifier.ProofData memory proofData = IZKVerifier.ProofData({
            proof: PROOF_DATA,
            publicInputs: publicInputs,
            modelHash: MODEL_HASH,
            inputHash: bytes32(uint256(123)),
            outputHash: bytes32(uint256(456)),
            timestamp: block.timestamp,
            proofType: IZKVerifier.ProofType.Inference
        });
        
        IZKVerifier.VerificationResult memory result = verifier.verifyProofWithMetadata(proofData);
        
        assertTrue(result.isValid);
        assertGt(result.gasUsed, 0);
        assertGt(result.computationComplexity, 0);
        assertEq(result.failureReason, "");
        assertEq(uint256(result.proofType), uint256(IZKVerifier.ProofType.Inference));
    }
    
    function testBatchVerifyProofs() public {
        bytes32[] memory publicInputs = new bytes32[](2);
        publicInputs[0] = bytes32(uint256(1));
        publicInputs[1] = bytes32(uint256(2));
        
        IZKVerifier.ProofData[] memory proofs = new IZKVerifier.ProofData[](2);
        proofs[0] = IZKVerifier.ProofData({
            proof: PROOF_DATA,
            publicInputs: publicInputs,
            modelHash: MODEL_HASH,
            inputHash: bytes32(uint256(123)),
            outputHash: bytes32(uint256(456)),
            timestamp: block.timestamp,
            proofType: IZKVerifier.ProofType.Inference
        });
        proofs[1] = IZKVerifier.ProofData({
            proof: PROOF_DATA,
            publicInputs: publicInputs,
            modelHash: MODEL_HASH,
            inputHash: bytes32(uint256(789)),
            outputHash: bytes32(uint256(101)),
            timestamp: block.timestamp,
            proofType: IZKVerifier.ProofType.Inference
        });
        
        IZKVerifier.VerificationResult[] memory results = verifier.batchVerifyProofs(proofs);
        
        assertEq(results.length, 2);
        assertTrue(results[0].isValid);
        assertTrue(results[1].isValid);
    }
    
    function testBatchVerifyProofsEmpty() public {
        IZKVerifier.ProofData[] memory proofs = new IZKVerifier.ProofData[](0);
        
        vm.expectRevert("No proofs provided");
        verifier.batchVerifyProofs(proofs);
    }
    
    function testBatchVerifyProofsTooMany() public {
        IZKVerifier.ProofData[] memory proofs = new IZKVerifier.ProofData[](11); // More than max of 10
        
        vm.expectRevert("Too many proofs in batch");
        verifier.batchVerifyProofs(proofs);
    }
    
    // Test scoring proof verification
    function testVerifyScoringProof() public {
        verifier.registerScoringModel(SCORING_MODEL_HASH, METADATA_URI);
        
        bytes32[] memory publicInputs = new bytes32[](2);
        publicInputs[0] = bytes32(uint256(1));
        publicInputs[1] = bytes32(uint256(2));
        
        bool result = verifier.verifyScoringProof(PROOF_DATA, publicInputs, SCORING_MODEL_HASH);
        assertTrue(result);
    }
    
    function testVerifyScoringProofModelNotRegistered() public {
        bytes32[] memory publicInputs = new bytes32[](2);
        publicInputs[0] = bytes32(uint256(1));
        publicInputs[1] = bytes32(uint256(2));
        
        vm.expectRevert("Scoring model not registered");
        verifier.verifyScoringProof(PROOF_DATA, publicInputs, SCORING_MODEL_HASH);
    }
    
    // Test trusted prover management
    function testAddTrustedProver() public {
        vm.expectEmit(true, false, false, true);
        emit TrustedProverUpdated(user1, true, IZKVerifier.ProofType.Inference);
        
        verifier.addTrustedProver(user1);
        
        assertTrue(verifier.isTrustedProver(user1));
    }
    
    function testRemoveTrustedProver() public {
        verifier.addTrustedProver(user1);
        assertTrue(verifier.isTrustedProver(user1));
        
        vm.expectEmit(true, false, false, true);
        emit TrustedProverUpdated(user1, false, IZKVerifier.ProofType.Inference);
        
        verifier.removeTrustedProver(user1);
        
        assertFalse(verifier.isTrustedProver(user1));
    }
    
    function testAddTrustedProverOnlyOwner() public {
        vm.prank(user1);
        vm.expectRevert(abi.encodeWithSelector(0x118cdaa7, user1)); // OwnableUnauthorizedAccount(address account)
        verifier.addTrustedProver(user2);
    }
    
    function testAddTrustedScoringProver() public {
        vm.expectEmit(true, false, false, true);
        emit TrustedProverUpdated(user1, true, IZKVerifier.ProofType.Scoring);
        
        verifier.addTrustedScoringProver(user1);
        
        assertTrue(verifier.isTrustedScoringProver(user1));
    }
    
    function testRemoveTrustedScoringProver() public {
        verifier.addTrustedScoringProver(user1);
        assertTrue(verifier.isTrustedScoringProver(user1));
        
        vm.expectEmit(true, false, false, true);
        emit TrustedProverUpdated(user1, false, IZKVerifier.ProofType.Scoring);
        
        verifier.removeTrustedScoringProver(user1);
        
        assertFalse(verifier.isTrustedScoringProver(user1));
    }
    
    // Test fee management
    function testSetVerificationFee() public {
        uint256 fee = 1000;
        
        vm.expectEmit(true, false, false, true);
        emit VerificationFeeUpdated(MODEL_HASH, fee, IZKVerifier.ProofType.Inference);
        
        verifier.setVerificationFee(MODEL_HASH, fee);
        
        assertEq(verifier.getVerificationFee(MODEL_HASH), fee);
    }
    
    function testSetScoringVerificationFee() public {
        uint256 fee = 2000;
        
        vm.expectEmit(true, false, false, true);
        emit VerificationFeeUpdated(SCORING_MODEL_HASH, fee, IZKVerifier.ProofType.Scoring);
        
        verifier.setScoringVerificationFee(SCORING_MODEL_HASH, fee);
        
        assertEq(verifier.getScoringVerificationFee(SCORING_MODEL_HASH), fee);
    }
    
    function testSetVerificationFeeOnlyOwner() public {
        vm.prank(user1);
        vm.expectRevert(abi.encodeWithSelector(0x118cdaa7, user1)); // OwnableUnauthorizedAccount(address account)
        verifier.setVerificationFee(MODEL_HASH, 1000);
    }
    
    // Test admin functions
    function testSetMaxProofSize() public {
        uint256 newMaxSize = 2 * 1024 * 1024; // 2MB
        
        verifier.setMaxProofSize(newMaxSize);
        
        assertEq(verifier.maxProofSize(), newMaxSize);
    }
    
    function testSetMaxPublicInputs() public {
        uint256 newMaxInputs = 512;
        
        verifier.setMaxPublicInputs(newMaxInputs);
        
        assertEq(verifier.maxPublicInputs(), newMaxInputs);
    }
    
    function testSetMaxScoringProofSize() public {
        uint256 newMaxSize = 3 * 1024 * 1024; // 3MB
        
        verifier.setMaxScoringProofSize(newMaxSize);
        
        assertEq(verifier.maxScoringProofSize(), newMaxSize);
    }
    
    function testSetMaxScoringPublicInputs() public {
        uint256 newMaxInputs = 1024;
        
        verifier.setMaxScoringPublicInputs(newMaxInputs);
        
        assertEq(verifier.maxScoringPublicInputs(), newMaxInputs);
    }
    
    function testSetMaxProofSizeOnlyOwner() public {
        vm.prank(user1);
        vm.expectRevert(abi.encodeWithSelector(0x118cdaa7, user1)); // OwnableUnauthorizedAccount(address account)
        verifier.setMaxProofSize(2 * 1024 * 1024);
    }
    
    // Test gas estimation
    function testEstimateVerificationGas() public {
        uint256 gasEstimate = verifier.estimateVerificationGas(PROOF_DATA);
        assertGt(gasEstimate, 0);
    }
    
    function testEstimateScoringVerificationGas() public {
        uint256 gasEstimate = verifier.estimateScoringVerificationGas(PROOF_DATA);
        assertGt(gasEstimate, 0);
    }
    
    // Test complexity estimation
    function testGetProofComplexity() public {
        uint256 complexity = verifier.getProofComplexity(PROOF_DATA);
        assertGt(complexity, 0);
    }
    
    function testGetScoringProofComplexity() public {
        uint256 complexity = verifier.getScoringProofComplexity(PROOF_DATA);
        assertGt(complexity, 0);
    }
    
    // Test statistics
    function testGetVerificationStats() public {
        (uint256 total, uint256 successful, uint256 failed) = verifier.getVerificationStats();
        assertEq(total, 0);
        assertEq(successful, 0);
        assertEq(failed, 0);
    }
    
    function testGetScoringVerificationStats() public {
        (uint256 total, uint256 successful, uint256 failed) = verifier.getScoringVerificationStats();
        assertEq(total, 0);
        assertEq(successful, 0);
        assertEq(failed, 0);
    }
} 