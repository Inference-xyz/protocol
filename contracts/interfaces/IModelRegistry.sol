// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IModelRegistry {
    struct ModelInfo {
        bytes32 modelHash;
        address deployer;
        string metadataURI;
        uint256 deployedAt;
        bool isActive;
        uint256 usageCount;
    }

    // Model Registration
    function registerModel(
        bytes32 modelHash, 
        string calldata metadataURI,
        bytes calldata zkProof,
        uint256[] calldata publicInputs
    ) external;
    
    function deactivateModel(bytes32 modelHash) external;
    function recordModelUsage(bytes32 modelHash) external;
    
    // Registry Management
    function setVerifierRegistry(address _verifierRegistry) external;
    
    // View Functions
    function isModelRegistered(bytes32 modelHash) external view returns (bool);
    function isModelActive(bytes32 modelHash) external view returns (bool);
    function getModelInfo(bytes32 modelHash) external view returns (ModelInfo memory);
    function getDeployerModels(address deployer) external view returns (bytes32[] memory);
    function getAllModels() external view returns (bytes32[] memory);
    function getActiveModels() external view returns (bytes32[] memory);
    function getModelCount() external view returns (uint256);
    function getActiveModelCount() external view returns (uint256);
    
    // Events
    event ModelRegistered(
        bytes32 indexed modelHash,
        address indexed deployer,
        string metadataURI,
        uint256 timestamp
    );
    event ModelDeactivated(bytes32 indexed modelHash, address indexed deactivator);
    event ModelUsed(bytes32 indexed modelHash, address indexed user);
} 