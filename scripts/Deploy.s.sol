pragma solidity ^0.8.19;

import "forge-std/Script.sol";
import "../contracts/ComputeMarketplace.sol";
import "../contracts/MockVerifier.sol";
import "../contracts/InfToken.sol";

contract DeployScript is Script {
    function run() external {
        vm.startBroadcast();

        // Deploy MockVerifier
        MockVerifier verifier = new MockVerifier();
        console.log("MockVerifier deployed at:", address(verifier));

        // Deploy InfToken
        InfToken token = new InfToken();
        console.log("InfToken deployed at:", address(token));

        // Deploy ComputeMarketplace
        ComputeMarketplace marketplace = new ComputeMarketplace(address(verifier));
        console.log("ComputeMarketplace deployed at:", address(marketplace));

        vm.stopBroadcast();
    }
}