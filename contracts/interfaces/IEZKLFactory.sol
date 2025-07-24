// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IEZKLFactory {
    struct VerificationContract {
        address contractAddress;
        bytes32 modelHash;
        bytes circuit;
        bool isActive;
        uint256 deploymentTime;
        address deployer;
    }

    struct DeploymentConfig {
        uint256 maxContractSize;
        uint256 maxGasLimit;
        uint256 deploymentFee;
        bool requireApproval;
    }

    // Core Functions
    function deployVerificationContract(bytes32 modelHash, bytes calldata circuit, bytes calldata verifyingKey)
        external
        returns (address contractAddress);

    function registerVerificationContract(address contractAddress, bytes32 modelHash, bytes calldata circuit)
        external;

    // Management Functions
    function setTrustedDeployer(address deployer, bool trusted) external;
    function pauseContract(bytes32 modelHash) external;
    function unpauseContract(bytes32 modelHash) external;
    function updateDeploymentConfig(DeploymentConfig calldata config) external;
    function approveContractDeployment(bytes32 modelHash) external;

    // View Functions
    function getVerificationContract(bytes32 modelHash) external view returns (address);
    function getContractInfo(bytes32 modelHash) external view returns (VerificationContract memory);
    function isTrustedDeployer(address deployer) external view returns (bool);
    function getDeployedContracts() external view returns (bytes32[] memory);
    function getDeploymentConfig() external view returns (DeploymentConfig memory);
    function isContractActive(bytes32 modelHash) external view returns (bool);
    function getContractDeployer(bytes32 modelHash) external view returns (address);
    function getContractDeploymentTime(bytes32 modelHash) external view returns (uint256);

    // Events
    event VerificationContractDeployed(
        address indexed contractAddress, bytes32 indexed modelHash, address indexed deployer
    );
    event VerificationContractRegistered(address indexed contractAddress, bytes32 indexed modelHash);
    event TrustedDeployerUpdated(address indexed deployer, bool trusted);
    event ContractPaused(bytes32 indexed modelHash);
    event ContractUnpaused(bytes32 indexed modelHash);
    event DeploymentConfigUpdated(DeploymentConfig config);
    event ContractDeploymentApproved(bytes32 indexed modelHash);
}
