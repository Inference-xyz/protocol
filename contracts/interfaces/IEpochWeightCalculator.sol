// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IEpochWeightCalculator {
    struct ParticipantWeight {
        address participant;
        uint256 weight; // Fixed point number 0-1 with 9 decimals precision
    }
    
    /**
     * @dev Calculate epoch weights for all participants
     * @param epochNumber The epoch number
     * @param participants Array of participant addresses
     * @param epochScores Array of score arrays for each participant
     * @return weights Array of (participant, weight) tuples
     */
    function calculateEpochWeights(
        uint256 epochNumber,
        address[] calldata participants,
        uint256[][] calldata epochScores
    ) external view returns (ParticipantWeight[] memory weights);
} 