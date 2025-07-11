// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";
import "./interfaces/IInferenceERC20.sol";

contract InferenceERC20 is ERC20, AccessControl, Pausable, IInferenceERC20 {
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    
    uint8 private _decimals;
    
    constructor(
        string memory name_,
        string memory symbol_,
        uint8 decimals_,
        address admin
    ) ERC20(name_, symbol_) {
        _decimals = decimals_;
        
        // Grant roles to admin
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(MINTER_ROLE, admin);
        _grantRole(PAUSER_ROLE, admin);
    }
    
    function decimals() public view virtual override returns (uint8) {
        return _decimals;
    }
    
    // Protocol-specific Functions
    function mint(address to, uint256 amount) external override onlyRole(MINTER_ROLE) whenNotPaused {
        _mint(to, amount);
    }
    
    function burn(address from, uint256 amount) external override onlyRole(MINTER_ROLE) whenNotPaused {
        _burn(from, amount);
    }
    
    function setMinter(address minter, bool enabled) external override onlyRole(DEFAULT_ADMIN_ROLE) {
        if (enabled) {
            _grantRole(MINTER_ROLE, minter);
        } else {
            _revokeRole(MINTER_ROLE, minter);
        }
        emit MinterUpdated(minter, enabled);
    }
    
    function isMinter(address account) external view override returns (bool) {
        return hasRole(MINTER_ROLE, account);
    }
    
    // Pausable functionality
    function pause() external onlyRole(PAUSER_ROLE) {
        _pause();
    }
    
    function unpause() external onlyRole(PAUSER_ROLE) {
        _unpause();
    }
    
    // Override transfer functions to add pause functionality
    function transfer(address to, uint256 amount) public virtual override(ERC20, IInferenceERC20) whenNotPaused returns (bool) {
        return super.transfer(to, amount);
    }
    
    function transferFrom(address from, address to, uint256 amount) public virtual override(ERC20, IInferenceERC20) whenNotPaused returns (bool) {
        return super.transferFrom(from, to, amount);
    }
    
    // Interface compliance
    function totalSupply() public view virtual override(ERC20, IInferenceERC20) returns (uint256) {
        return super.totalSupply();
    }
    
    function balanceOf(address account) public view virtual override(ERC20, IInferenceERC20) returns (uint256) {
        return super.balanceOf(account);
    }
    
    function allowance(address owner, address spender) public view virtual override(ERC20, IInferenceERC20) returns (uint256) {
        return super.allowance(owner, spender);
    }
    
    function approve(address spender, uint256 amount) public virtual override(ERC20, IInferenceERC20) whenNotPaused returns (bool) {
        return super.approve(spender, amount);
    }
    
    // The following functions are overrides required by Solidity
    function supportsInterface(bytes4 interfaceId) public view virtual override returns (bool) {
        return super.supportsInterface(interfaceId);
    }
} 