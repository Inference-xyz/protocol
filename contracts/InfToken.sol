// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {ERC20Permit} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";
import {ERC20Capped} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Capped.sol";
import {ERC20Burnable} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import {ERC20Pausable} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Pausable.sol";
import {AccessControl} from "@openzeppelin/contracts/access/AccessControl.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title InfToken
 * @notice INF utility token for Inference protocol - used for staking, rewards, and governance
 * @dev Combines multiple OpenZeppelin extensions:
 * - ERC20Permit: Gasless approvals via signatures (EIP-2612)
 * - ERC20Capped: Hard supply cap at 1B tokens
 * - ERC20Burnable: Token holders can burn their tokens
 * - ERC20Pausable: Emergency pause transfers
 * - AccessControl: Role-based minting and pausing
 */
contract InfToken is 
    ERC20,
    ERC20Permit,
    ERC20Capped,
    ERC20Burnable,
    ERC20Pausable,
    AccessControl,
    ReentrancyGuard
{
    uint256 public constant MAX_SUPPLY = 1_000_000_000 * 10**18; // 1B hard cap
    uint256 public constant INITIAL_SUPPLY = 100_000_000 * 10**18; // 100M minted to deployer
    
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    
    error InvalidMinterAddress();
    
    event MinterUpdated(address indexed minter, bool enabled);
    
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
    
    /**
     * @dev Override required by Solidity when inheriting multiple base contracts
     * Enforces cap limit and pause state on all transfers
     */
    function _update(address from, address to, uint256 value)
        internal
        override(ERC20, ERC20Capped, ERC20Pausable)
    {
        super._update(from, to, value);
    }
    
    function mint(address to, uint256 amount) 
        external 
        onlyRole(MINTER_ROLE)
    {
        _mint(to, amount);
    }
    
    // burn() and burnFrom() are inherited from ERC20Burnable - no need to override
    
    /**
     * @notice Grant or revoke MINTER_ROLE to an address
     * @param minter Address to update
     * @param enabled True to grant role, false to revoke
     */
    function setMinter(address minter, bool enabled) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE)
    {
        if (minter == address(0)) revert InvalidMinterAddress();
        
        bool currentlyMinter = hasRole(MINTER_ROLE, minter);
        require(currentlyMinter != enabled, "Minter status already set");
        
        if (enabled) {
            _grantRole(MINTER_ROLE, minter);
        } else {
            _revokeRole(MINTER_ROLE, minter);
        }
        
        emit MinterUpdated(minter, enabled);
    }
    
    function isMinter(address account) external view returns (bool) {
        return hasRole(MINTER_ROLE, account);
    }
    
    function pause() external onlyRole(PAUSER_ROLE) {
        _pause();
    }
    
    function unpause() external onlyRole(PAUSER_ROLE) {
        _unpause();
    }
    
    // Inherited view functions (no need to override):
    // - cap() from ERC20Capped
    // - paused() from ERC20Pausable  
    // - decimals() from ERC20 (returns 18)
    // - nonces() from ERC20Permit
}

