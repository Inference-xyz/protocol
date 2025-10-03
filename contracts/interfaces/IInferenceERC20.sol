// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {IERC20Metadata} from "@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol";
import {IERC20Permit} from "@openzeppelin/contracts/token/ERC20/extensions/IERC20Permit.sol";

interface IInferenceERC20 is IERC20, IERC20Metadata, IERC20Permit {
    function mint(address to, uint256 amount) external;
    function burn(uint256 amount) external;
    function burnFrom(address account, uint256 amount) external;
    function setMinter(address minter, bool enabled) external;
    function isMinter(address account) external view returns (bool);
    function pause() external;
    function unpause() external;
    
    // View Functions
    function cap() external view returns (uint256);
    function paused() external view returns (bool);
}
