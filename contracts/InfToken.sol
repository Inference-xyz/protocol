// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {ERC20Permit} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";
import {ERC20Capped} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Capped.sol";
import {ERC20Burnable} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import {ERC20Pausable} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Pausable.sol";
import {AccessControl} from "@openzeppelin/contracts/access/AccessControl.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import {Pausable} from "@openzeppelin/contracts/utils/Pausable.sol";
import {IERC20Permit} from "@openzeppelin/contracts/token/ERC20/extensions/IERC20Permit.sol";
import {IERC20Metadata} from "@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol";
import {IInferenceERC20} from "./interfaces/IInferenceERC20.sol";

contract InfToken is 
    ERC20,
    ERC20Permit,
    ERC20Capped,
    ERC20Burnable,
    ERC20Pausable,
    AccessControl,
    ReentrancyGuard,
    IInferenceERC20
{
    uint256 public constant MAX_SUPPLY = 1_000_000_000 * 10**18; // 1B tokens
    uint256 public constant INITIAL_SUPPLY = 100_000_000 * 10**18; // 100M tokens
    
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    
    mapping(address => bool) private _minters;
    
    error OnlyMinter();
    error MinterAlreadySet();
    error InvalidMinterAddress();
    
    event MinterUpdated(address indexed minter, bool enabled);
    event TokensMinted(address indexed to, uint256 amount);
    event TokensBurned(address indexed from, uint256 amount);
    
    constructor() 
        ERC20("Inference Token", "INF")
        ERC20Permit("Inference Token")
        ERC20Capped(MAX_SUPPLY)
    {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
        _grantRole(PAUSER_ROLE, msg.sender);
        
        _mint(msg.sender, INITIAL_SUPPLY);
    }
    
    function _update(address from, address to, uint256 value)
        internal
        override(ERC20, ERC20Capped, ERC20Pausable)
    {
        super._update(from, to, value);
    }
    
    function mint(address to, uint256 amount) 
        external 
        override
        onlyRole(MINTER_ROLE)
        nonReentrant
    {
        _mint(to, amount);
        emit TokensMinted(to, amount);
    }
    
    function burn(uint256 amount) public override(ERC20Burnable, IInferenceERC20) {
        super.burn(amount);
        emit TokensBurned(msg.sender, amount);
    }
    
    function burnFrom(address account, uint256 amount) public override(ERC20Burnable, IInferenceERC20) {
        super.burnFrom(account, amount);
        emit TokensBurned(account, amount);
    }
    
    function setMinter(address minter, bool enabled) 
        external 
        override
        onlyRole(DEFAULT_ADMIN_ROLE)
    {
        if (minter == address(0)) revert InvalidMinterAddress();
        if (_minters[minter] == enabled) revert MinterAlreadySet();
        
        _minters[minter] = enabled;
        
        if (enabled) {
            _grantRole(MINTER_ROLE, minter);
        } else {
            _revokeRole(MINTER_ROLE, minter);
        }
        
        emit MinterUpdated(minter, enabled);
    }
    
    function isMinter(address account) external view override returns (bool) {
        return _minters[account];
    }
    
    function pause() external override onlyRole(PAUSER_ROLE) {
        _pause();
    }
    
    function unpause() external override onlyRole(PAUSER_ROLE) {
        _unpause();
    }
    
    function cap() public view override(ERC20Capped, IInferenceERC20) returns (uint256) {
        return super.cap();
    }
    
    function paused() public view override(Pausable, IInferenceERC20) returns (bool) {
        return super.paused();
    }
    
    function decimals() public pure override(ERC20, IERC20Metadata) returns (uint8) {
        return 18;
    }
    
    function nonces(address owner) public view override(ERC20Permit, IERC20Permit) returns (uint256) {
        return super.nonces(owner);
    }
}
