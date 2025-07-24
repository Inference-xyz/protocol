// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./ComputeMarketplace.sol";

// generate a real verifier via ezkl cli
contract MockVerifier is IHalo2Verifier {
    function verifyProof(bytes calldata proof, uint256[] calldata instances) external pure returns (bool) {
        return true;
    }
}
