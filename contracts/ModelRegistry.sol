// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title ModelRegistry
 * @dev Permissionless registry for AI model architectures
 */
contract ModelRegistry is Ownable {
    struct ModelInfo {
        bytes32 modelHash;
        address deployer;
        string metadataURI;
        uint256 deployedAt;
        bool isActive;
        uint256 usageCount;
    }

    mapping(bytes32 => ModelInfo) public models;
    mapping(address => bytes32[]) public deployerModels;
    bytes32[] public allModelHashes;
    
    event ModelRegistered(
        bytes32 indexed modelHash,
        address indexed deployer,
        string metadataURI,
        uint256 timestamp
    );
    event ModelDeactivated(bytes32 indexed modelHash, address indexed deactivator);
    event ModelUsed(bytes32 indexed modelHash, address indexed user);

    constructor() Ownable(msg.sender) {}

    /**
     * @dev Register a new model architecture (permissionless)
     * @param modelHash Hash of the model architecture
     * @param metadataURI URI containing model metadata (architecture info, input/output shapes, etc.)
     */
    function registerModel(bytes32 modelHash, string calldata metadataURI) external {
        require(modelHash != bytes32(0), "Invalid model hash");
        require(!isModelRegistered(modelHash), "Model already registered");
        require(bytes(metadataURI).length > 0, "Empty metadata URI");

        models[modelHash] = ModelInfo({
            modelHash: modelHash,
            deployer: msg.sender,
            metadataURI: metadataURI,
            deployedAt: block.timestamp,
            isActive: true,
            usageCount: 0
        });

        deployerModels[msg.sender].push(modelHash);
        allModelHashes.push(modelHash);

        emit ModelRegistered(modelHash, msg.sender, metadataURI, block.timestamp);
    }

    /**
     * @dev Deactivate a model (only deployer or owner)
     * @param modelHash Hash of the model to deactivate
     */
    function deactivateModel(bytes32 modelHash) external {
        require(isModelRegistered(modelHash), "Model not registered");
        ModelInfo storage model = models[modelHash];
        require(
            msg.sender == model.deployer || msg.sender == owner(),
            "Only deployer or owner can deactivate"
        );
        require(model.isActive, "Model already inactive");

        model.isActive = false;
        emit ModelDeactivated(modelHash, msg.sender);
    }

    /**
     * @dev Record model usage (called by contests)
     * @param modelHash Hash of the model being used
     */
    function recordModelUsage(bytes32 modelHash) external {
        require(isModelRegistered(modelHash), "Model not registered");
        require(models[modelHash].isActive, "Model is inactive");
        
        models[modelHash].usageCount++;
        emit ModelUsed(modelHash, msg.sender);
    }

    // View Functions
    function isModelRegistered(bytes32 modelHash) public view returns (bool) {
        return models[modelHash].deployer != address(0);
    }

    function isModelActive(bytes32 modelHash) public view returns (bool) {
        return isModelRegistered(modelHash) && models[modelHash].isActive;
    }

    function getModelInfo(bytes32 modelHash) external view returns (ModelInfo memory) {
        require(isModelRegistered(modelHash), "Model not registered");
        return models[modelHash];
    }

    function getDeployerModels(address deployer) external view returns (bytes32[] memory) {
        return deployerModels[deployer];
    }

    function getAllModels() external view returns (bytes32[] memory) {
        return allModelHashes;
    }

    function getActiveModels() external view returns (bytes32[] memory) {
        uint256 activeCount = 0;
        
        // First pass: count active models
        for (uint256 i = 0; i < allModelHashes.length; i++) {
            if (models[allModelHashes[i]].isActive) {
                activeCount++;
            }
        }

        // Second pass: collect active models
        bytes32[] memory activeModels = new bytes32[](activeCount);
        uint256 index = 0;
        for (uint256 i = 0; i < allModelHashes.length; i++) {
            if (models[allModelHashes[i]].isActive) {
                activeModels[index] = allModelHashes[i];
                index++;
            }
        }

        return activeModels;
    }

    function getModelCount() external view returns (uint256) {
        return allModelHashes.length;
    }

    function getActiveModelCount() external view returns (uint256) {
        uint256 count = 0;
        for (uint256 i = 0; i < allModelHashes.length; i++) {
            if (models[allModelHashes[i]].isActive) {
                count++;
            }
        }
        return count;
    }
} 