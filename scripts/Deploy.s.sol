pragma solidity ^0.8.19;

import "forge-std/Script.sol";
import "../contracts/ComputeMarketplace.sol";
import "../contracts/verifier.sol";

contract DeployScript is Script {
    function run() external {
        vm.startBroadcast();

        Halo2Verifier verifier = new Halo2Verifier();
        ComputeMarketplace marketplace = new ComputeMarketplace(address(verifier));

        console.log("Verifier deployed at:", address(verifier));
        console.log("Marketplace deployed at:", address(marketplace));

        vm.stopBroadcast();
    }
}