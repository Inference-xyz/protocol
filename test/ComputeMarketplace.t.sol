// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../contracts/ModelRegistry.sol";
import "../contracts/InfToken.sol";

contract ModelRegistryTest is Test {
    ModelRegistry public modelRegistry;
    InfToken public infToken;

    address public deployer1 = address(0x1);
    address public deployer2 = address(0x2);
    address public user = address(0x3);

    bytes32 constant MODEL_HASH_1 = bytes32(uint256(0x123456789));
    bytes32 constant MODEL_HASH_2 = bytes32(uint256(0x987654321));
    string constant METADATA_URI_1 = "ipfs://model-metadata-1";
    string constant METADATA_URI_2 = "ipfs://model-metadata-2";

    function setUp() public {
        modelRegistry = new ModelRegistry();
        infToken = new InfToken();
    }

    function testRegisterModel() public {
        vm.prank(deployer1);
        modelRegistry.registerModel(MODEL_HASH_1, METADATA_URI_1);

        // Verify model was registered
        assertTrue(modelRegistry.isModelRegistered(MODEL_HASH_1));
        assertTrue(modelRegistry.isModelActive(MODEL_HASH_1));

        ModelRegistry.ModelInfo memory modelInfo = modelRegistry.getModelInfo(MODEL_HASH_1);
        assertEq(modelInfo.modelHash, MODEL_HASH_1);
        assertEq(modelInfo.deployer, deployer1);
        assertEq(modelInfo.metadataURI, METADATA_URI_1);
        assertTrue(modelInfo.isActive);
        assertEq(modelInfo.usageCount, 0);
    }

    function testRegisterModelRevert() public {
        // Test invalid model hash
        vm.prank(deployer1);
        vm.expectRevert("Invalid model hash");
        modelRegistry.registerModel(bytes32(0), METADATA_URI_1);

        // Test empty metadata URI
        vm.prank(deployer1);
        vm.expectRevert("Empty metadata URI");
        modelRegistry.registerModel(MODEL_HASH_1, "");

        // Register model first time
        vm.prank(deployer1);
        modelRegistry.registerModel(MODEL_HASH_1, METADATA_URI_1);

        // Test duplicate registration
        vm.prank(deployer2);
        vm.expectRevert("Model already registered");
        modelRegistry.registerModel(MODEL_HASH_1, METADATA_URI_2);
    }

    function testDeactivateModel() public {
        // Register model
        vm.prank(deployer1);
        modelRegistry.registerModel(MODEL_HASH_1, METADATA_URI_1);

        // Deactivate model
        vm.prank(deployer1);
        modelRegistry.deactivateModel(MODEL_HASH_1);

        // Verify model was deactivated
        assertFalse(modelRegistry.isModelActive(MODEL_HASH_1));
        assertTrue(modelRegistry.isModelRegistered(MODEL_HASH_1));

        // Try to deactivate again
        vm.prank(deployer1);
        vm.expectRevert("Model already inactive");
        modelRegistry.deactivateModel(MODEL_HASH_1);
    }

    function testDeactivateModelRevert() public {
        // Test deactivating non-existent model
        vm.prank(deployer1);
        vm.expectRevert("Model not registered");
        modelRegistry.deactivateModel(MODEL_HASH_1);

        // Register model
        vm.prank(deployer1);
        modelRegistry.registerModel(MODEL_HASH_1, METADATA_URI_1);

        // Test deactivating with different deployer
        vm.prank(deployer2);
        vm.expectRevert("Only deployer or owner can deactivate");
        modelRegistry.deactivateModel(MODEL_HASH_1);
    }

    function testRecordModelUsage() public {
        // Register model
        vm.prank(deployer1);
        modelRegistry.registerModel(MODEL_HASH_1, METADATA_URI_1);

        // Record usage
        vm.prank(user);
        modelRegistry.recordModelUsage(MODEL_HASH_1);

        // Verify usage was recorded
        ModelRegistry.ModelInfo memory modelInfo = modelRegistry.getModelInfo(MODEL_HASH_1);
        assertEq(modelInfo.usageCount, 1);

        // Record usage again
        vm.prank(user);
        modelRegistry.recordModelUsage(MODEL_HASH_1);

        modelInfo = modelRegistry.getModelInfo(MODEL_HASH_1);
        assertEq(modelInfo.usageCount, 2);
    }

    function testRecordModelUsageRevert() public {
        // Test recording usage for non-existent model
        vm.prank(user);
        vm.expectRevert("Model not registered");
        modelRegistry.recordModelUsage(MODEL_HASH_1);

        // Register and deactivate model
        vm.prank(deployer1);
        modelRegistry.registerModel(MODEL_HASH_1, METADATA_URI_1);

        vm.prank(deployer1);
        modelRegistry.deactivateModel(MODEL_HASH_1);

        // Test recording usage for inactive model
        vm.prank(user);
        vm.expectRevert("Model is inactive");
        modelRegistry.recordModelUsage(MODEL_HASH_1);
    }

    function testGetDeployerModels() public {
        // Register models from different deployers
        vm.prank(deployer1);
        modelRegistry.registerModel(MODEL_HASH_1, METADATA_URI_1);

        vm.prank(deployer2);
        modelRegistry.registerModel(MODEL_HASH_2, METADATA_URI_2);

        // Get deployer1's models
        bytes32[] memory deployer1Models = modelRegistry.getDeployerModels(deployer1);
        assertEq(deployer1Models.length, 1);
        assertEq(deployer1Models[0], MODEL_HASH_1);

        // Get deployer2's models
        bytes32[] memory deployer2Models = modelRegistry.getDeployerModels(deployer2);
        assertEq(deployer2Models.length, 1);
        assertEq(deployer2Models[0], MODEL_HASH_2);
    }

    function testGetAllModels() public {
        // Register multiple models
        vm.prank(deployer1);
        modelRegistry.registerModel(MODEL_HASH_1, METADATA_URI_1);

        vm.prank(deployer2);
        modelRegistry.registerModel(MODEL_HASH_2, METADATA_URI_2);

        // Get all models
        bytes32[] memory allModels = modelRegistry.getAllModels();
        assertEq(allModels.length, 2);
        assertEq(allModels[0], MODEL_HASH_1);
        assertEq(allModels[1], MODEL_HASH_2);
    }

    function testGetActiveModels() public {
        // Register models
        vm.prank(deployer1);
        modelRegistry.registerModel(MODEL_HASH_1, METADATA_URI_1);

        vm.prank(deployer2);
        modelRegistry.registerModel(MODEL_HASH_2, METADATA_URI_2);

        // Deactivate one model
        vm.prank(deployer1);
        modelRegistry.deactivateModel(MODEL_HASH_1);

        // Get active models
        bytes32[] memory activeModels = modelRegistry.getActiveModels();
        assertEq(activeModels.length, 1);
        assertEq(activeModels[0], MODEL_HASH_2);
    }

    function testGetModelCount() public {
        assertEq(modelRegistry.getModelCount(), 0);

        vm.prank(deployer1);
        modelRegistry.registerModel(MODEL_HASH_1, METADATA_URI_1);

        assertEq(modelRegistry.getModelCount(), 1);

        vm.prank(deployer2);
        modelRegistry.registerModel(MODEL_HASH_2, METADATA_URI_2);

        assertEq(modelRegistry.getModelCount(), 2);
    }

    function testGetActiveModelCount() public {
        assertEq(modelRegistry.getActiveModelCount(), 0);

        vm.prank(deployer1);
        modelRegistry.registerModel(MODEL_HASH_1, METADATA_URI_1);

        assertEq(modelRegistry.getActiveModelCount(), 1);

        vm.prank(deployer1);
        modelRegistry.deactivateModel(MODEL_HASH_1);

        assertEq(modelRegistry.getActiveModelCount(), 0);
    }
}
