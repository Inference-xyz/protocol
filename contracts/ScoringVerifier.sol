// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";

interface IZKVerifier {
    function verifyProof(bytes calldata proof, uint256[] calldata publicInputs) external view returns (bool);
}

/**
 * @title ScoringVerifier
 * @dev Contract for verifying ZK proofs of scoring computations in contests
 */
contract ScoringVerifier is Ownable {
    struct ScoringProofData {
        bytes32 inputHash;          // Hash of participant outputs being scored
        bytes32 outputHash;         // Hash of original inference output
        uint256[] scores;           // Computed scores
        bytes32 scoringModelHash;   // Hash of scoring model used
        bytes proof;               // ZK proof
        uint256[] publicInputs;    // Public inputs for verification
    }
    
    struct VerificationResult {
        bool isValid;
        uint256 gasUsed;
        string failureReason;
    }
    
    mapping(bytes32 => IZKVerifier) public verifiers; // modelHash => verifier contract
    mapping(bytes32 => bool) public registeredModels;
    mapping(address => bool) public trustedVerifiers;
    

    
    event ScoringModelRegistered(bytes32 indexed modelHash, address indexed verifier);
    event ScoringProofVerified(
        bytes32 indexed modelHash,
        bytes32 inputHash,
        bytes32 outputHash,
        uint256[] scores,
        bool isValid,
        address verifier
    );

    event TrustedVerifierUpdated(address indexed verifier, bool trusted);
    
    constructor() Ownable(msg.sender) {
        trustedVerifiers[msg.sender] = true;
    }
    
    /**
     * @dev Register a scoring model with its ZK verifier contract
     * @param modelHash Hash of the scoring model
     * @param verifierContract Address of the ZK verifier contract
     */
    function registerScoringModel(bytes32 modelHash, address verifierContract) external onlyOwner {
        require(modelHash != bytes32(0), "Invalid model hash");
        require(verifierContract != address(0), "Invalid verifier contract");
        require(!registeredModels[modelHash], "Model already registered");
        
        // Verify the contract implements the interface
        try IZKVerifier(verifierContract).verifyProof(hex"00", new uint256[](0)) {
            // Contract implements interface (even if verification fails)
        } catch {
            revert("Contract does not implement IZKVerifier interface");
        }
        
        verifiers[modelHash] = IZKVerifier(verifierContract);
        registeredModels[modelHash] = true;
        
        emit ScoringModelRegistered(modelHash, verifierContract);
    }
    
    /**
     * @dev Verify a scoring ZK proof
     * @param proofData The scoring proof data to verify
     * @return result Verification result with success status and details
     */
    function verifyScoringProof(ScoringProofData calldata proofData) 
        external
        returns (VerificationResult memory result) 
    {
        require(registeredModels[proofData.scoringModelHash], "Scoring model not registered");
        require(proofData.inputHash != bytes32(0), "Invalid input hash");
        require(proofData.outputHash != bytes32(0), "Invalid output hash");
        require(proofData.scores.length > 0, "No scores provided");
        require(proofData.proof.length > 0, "No proof provided");
        
        uint256 gasStart = gasleft();
        
        try verifiers[proofData.scoringModelHash].verifyProof(proofData.proof, proofData.publicInputs) returns (bool isValid) {
            uint256 gasUsed = gasStart - gasleft();
            
            result = VerificationResult({
                isValid: isValid,
                gasUsed: gasUsed,
                failureReason: isValid ? "" : "Proof verification failed"
            });
            
            emit ScoringProofVerified(
                proofData.scoringModelHash,
                proofData.inputHash,
                proofData.outputHash,
                proofData.scores,
                isValid,
                msg.sender
            );
            
        } catch Error(string memory reason) {
            uint256 gasUsed = gasStart - gasleft();
            result = VerificationResult({
                isValid: false,
                gasUsed: gasUsed,
                failureReason: reason
            });
            
        } catch {
            uint256 gasUsed = gasStart - gasleft();
            result = VerificationResult({
                isValid: false,
                gasUsed: gasUsed,
                failureReason: "Unknown verification error"
            });
        }
    }
    
    /**
     * @dev Batch verify multiple scoring proofs
     * @param proofDataArray Array of scoring proof data to verify
     * @return results Array of verification results
     */
    function batchVerifyScoringProofs(ScoringProofData[] calldata proofDataArray)
        external
        returns (VerificationResult[] memory results)
    {
        require(proofDataArray.length > 0, "No proofs provided");
        
        results = new VerificationResult[](proofDataArray.length);
        
        for (uint256 i = 0; i < proofDataArray.length; i++) {
            results[i] = this.verifyScoringProof(proofDataArray[i]);
        }
    }
    
    /**
     * @dev Generate scoring proof hash for off-chain verification
     * @param proofData The scoring proof data
     * @return proofHash Hash of the proof data
     */
    function generateScoringProofHash(ScoringProofData calldata proofData) 
        external 
        pure 
        returns (bytes32 proofHash) 
    {
        return keccak256(abi.encode(
            proofData.inputHash,
            proofData.outputHash,
            proofData.scores,
            proofData.scoringModelHash,
            proofData.proof
        ));
    }
    
    // Admin Functions
    
    function setTrustedVerifier(address verifier, bool trusted) external onlyOwner {
        require(verifier != address(0), "Invalid verifier address");
        trustedVerifiers[verifier] = trusted;
        emit TrustedVerifierUpdated(verifier, trusted);
    }
    
    // View Functions
    function isScoringModelRegistered(bytes32 modelHash) external view returns (bool) {
        return registeredModels[modelHash];
    }
    
    function getScoringVerifier(bytes32 modelHash) external view returns (address) {
        require(registeredModels[modelHash], "Scoring model not registered");
        return address(verifiers[modelHash]);
    }
    
    function isTrustedVerifier(address verifier) external view returns (bool) {
        return trustedVerifiers[verifier];
    }
    

} 