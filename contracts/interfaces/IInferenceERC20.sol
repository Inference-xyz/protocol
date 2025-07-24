// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IInferenceERC20 {
    // ERC20 Standard Functions
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);

    // Protocol-specific Functions
    function mint(address to, uint256 amount) external;
    function burn(address from, uint256 amount) external;
    function setMinter(address minter, bool enabled) external;
    function isMinter(address account) external view returns (bool);

    // Protocol-specific Events (ERC20 events are inherited from OpenZeppelin)
    event MinterUpdated(address indexed minter, bool enabled);
    event Permit2AddressUpdated(address indexed oldAddress, address indexed newAddress);
    event Permit2Transfer(address indexed from, address indexed to, uint256 amount);
}
