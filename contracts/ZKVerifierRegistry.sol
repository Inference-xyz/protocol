// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";

interface IZKVerifier {
    function verifyProof(bytes calldata proof, uint256[] calldata publicInputs) external view returns (bool);
}

/**
 * @title ZKVerifierRegistry  
 * @dev Global registry for scoring function ZK verifiers
 */
contract ZKVerifierRegistry is Ownable {
    struct VerifierInfo {
        bytes32 modelHash;
        address verifierContract;
        address deployer;
        string metadataURI;
        uint256 deployedAt;
        bool isActive;
    }

    mapping(bytes32 => VerifierInfo) public verifiers;
    bytes32[] public allVerifierHashes;
    
    event VerifierRegistered(
        bytes32 indexed modelHash,
        address indexed verifierContract,
        address indexed deployer,
        string metadataURI,
        uint256 timestamp
    );
    event VerifierDeactivated(bytes32 indexed modelHash, address indexed deactivator);

    constructor() Ownable(msg.sender) {}

    /**
     * @dev Register a new ZK verifier for scoring functions
     * @param modelHash Hash of the scoring model 
     * @param verifierContract Address of the deployed ZK verifier contract
     * @param metadataURI URI containing verifier metadata (circuit info, etc.)
     */
    function registerVerifier(
        bytes32 modelHash,
        address verifierContract,
        string calldata metadataURI
    ) external {
        require(modelHash != bytes32(0), "Invalid model hash");
        require(verifierContract != address(0), "Invalid verifier contract");
        require(!isVerifierRegistered(modelHash), "Verifier already registered");
        require(bytes(metadataURI).length > 0, "Empty metadata URI");

        // Verify the contract implements the interface by calling verifyProof with dummy data
        try IZKVerifier(verifierContract).verifyProof(hex"00", new uint256[](0)) {
            // Contract implements interface (even if verification fails)
        } catch {
            revert("Contract does not implement IZKVerifier interface");
        }

        verifiers[modelHash] = VerifierInfo({
            modelHash: modelHash,
            verifierContract: verifierContract,
            deployer: msg.sender,
            metadataURI: metadataURI,
            deployedAt: block.timestamp,
            isActive: true
        });

        allVerifierHashes.push(modelHash);

        emit VerifierRegistered(modelHash, verifierContract, msg.sender, metadataURI, block.timestamp);
    }

    /**
     * @dev Deactivate a verifier (only deployer or owner)
     * @param modelHash Hash of the scoring model to deactivate
     */
    function deactivateVerifier(bytes32 modelHash) external {
        require(isVerifierRegistered(modelHash), "Verifier not registered");
        VerifierInfo storage verifier = verifiers[modelHash];
        require(
            msg.sender == verifier.deployer || msg.sender == owner(),
            "Only deployer or owner can deactivate"
        );
        require(verifier.isActive, "Verifier already inactive");

        verifier.isActive = false;
        emit VerifierDeactivated(modelHash, msg.sender);
    }

    /**
     * @dev Verify a proof using the registered verifier
     * @param modelHash Hash of the scoring model
     * @param proof ZK proof to verify
     * @param publicInputs Public inputs for verification
     */
    function verifyProof(
        bytes32 modelHash,
        bytes calldata proof,
        uint256[] calldata publicInputs
    ) external view returns (bool) {
        require(isVerifierActive(modelHash), "Verifier not active");
        
        VerifierInfo storage verifier = verifiers[modelHash];
        return IZKVerifier(verifier.verifierContract).verifyProof(proof, publicInputs);
    }

    // View Functions
    function isVerifierRegistered(bytes32 modelHash) public view returns (bool) {
        return verifiers[modelHash].verifierContract != address(0);
    }

    function isVerifierActive(bytes32 modelHash) public view returns (bool) {
        return isVerifierRegistered(modelHash) && verifiers[modelHash].isActive;
    }

    function getVerifierInfo(bytes32 modelHash) external view returns (VerifierInfo memory) {
        require(isVerifierRegistered(modelHash), "Verifier not registered");
        return verifiers[modelHash];
    }

    function getVerifierContract(bytes32 modelHash) external view returns (address) {
        require(isVerifierRegistered(modelHash), "Verifier not registered");
        return verifiers[modelHash].verifierContract;
    }

    function getAllVerifiers() external view returns (bytes32[] memory) {
        return allVerifierHashes;
    }

    function getActiveVerifiers() external view returns (bytes32[] memory) {
        uint256 activeCount = 0;
        
        // First pass: count active verifiers
        for (uint256 i = 0; i < allVerifierHashes.length; i++) {
            if (verifiers[allVerifierHashes[i]].isActive) {
                activeCount++;
            }
        }

        // Second pass: collect active verifiers
        bytes32[] memory activeVerifiers = new bytes32[](activeCount);
        uint256 index = 0;
        for (uint256 i = 0; i < allVerifierHashes.length; i++) {
            if (verifiers[allVerifierHashes[i]].isActive) {
                activeVerifiers[index] = allVerifierHashes[i];
                index++;
            }
        }

        return activeVerifiers;
    }

    function getVerifierCount() external view returns (uint256) {
        return allVerifierHashes.length;
    }

    function getActiveVerifierCount() external view returns (uint256) {
        uint256 count = 0;
        for (uint256 i = 0; i < allVerifierHashes.length; i++) {
            if (verifiers[allVerifierHashes[i]].isActive) {
                count++;
            }
        }
        return count;
    }
} 