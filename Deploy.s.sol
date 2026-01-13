pragma solidity ^0.8.19;

import "forge-std/Script.sol";
import "contracts/InfToken.sol";
import "contracts/ContestFactory.sol";

contract DeployScript is Script {
    function run() external {
        vm.startBroadcast();
        InfToken token = new InfToken();
        ContestFactory contestFactory = new ContestFactory();
        vm.stopBroadcast();
    }
}