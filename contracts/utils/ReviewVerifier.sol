// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "../types/ReviewStructs.sol";

/**
 * @title ReviewVerifier
 * @dev Utilities for verifying signed reviews and managing review validation
 */
contract ReviewVerifier {
    using ECDSA for bytes32;
    using ReviewStructs for ReviewStructs.SignedReview;

    string public constant DOMAIN_NAME = "InferenceProtocol";
    string public constant DOMAIN_VERSION = "1";

    // Mapping to track used nonces per reviewer
    mapping(address => mapping(uint256 => bool)) public usedNonces;

    // Events
    event ReviewVerified(
        address indexed reviewer,
        uint256 indexed submissionId,
        uint256 indexed contestId,
        uint256 score
    );

    error InvalidSignature();
    error ReviewExpired();
    error NonceAlreadyUsed();
    error InvalidScore();
    error InvalidReviewer();

    /**
     * @dev Verify a signed review
     * @param review The signed review to verify
     * @param contestAddress The address of the contest contract
     * @return isValid Whether the review signature is valid
     */
    function verifyReview(
        ReviewStructs.SignedReview memory review,
        address contestAddress
    ) external returns (bool isValid) {
        // Check if review has expired
        if (block.timestamp > review.deadline) {
            revert ReviewExpired();
        }

        // Check if nonce has been used
        if (usedNonces[review.reviewer][review.nonce]) {
            revert NonceAlreadyUsed();
        }

        // Verify signature
        bytes32 domainSeparator = ReviewStructs.getDomainSeparator(
            DOMAIN_NAME,
            DOMAIN_VERSION,
            block.chainid,
            contestAddress
        );

        bytes32 typedDataHash = ReviewStructs.getReviewTypedDataHash(review, domainSeparator);
        address recoveredSigner = typedDataHash.recover(review.signature);

        if (recoveredSigner != review.reviewer) {
            revert InvalidSignature();
        }

        // Mark nonce as used
        usedNonces[review.reviewer][review.nonce] = true;

        emit ReviewVerified(review.reviewer, review.submissionId, review.contestId, review.score);
        return true;
    }

    /**
     * @dev Batch verify multiple reviews
     * @param reviews Array of signed reviews to verify
     * @param contestAddress The address of the contest contract
     * @return validReviews Array of booleans indicating which reviews are valid
     */
    function batchVerifyReviews(
        ReviewStructs.SignedReview[] memory reviews,
        address contestAddress
    ) external returns (bool[] memory validReviews) {
        validReviews = new bool[](reviews.length);
        
        for (uint256 i = 0; i < reviews.length; i++) {
            try this.verifyReview(reviews[i], contestAddress) returns (bool isValid) {
                validReviews[i] = isValid;
            } catch {
                validReviews[i] = false;
            }
        }
        
        return validReviews;
    }

    /**
     * @dev Check if a nonce has been used by a reviewer
     * @param reviewer The reviewer address
     * @param nonce The nonce to check
     * @return used Whether the nonce has been used
     */
    function isNonceUsed(address reviewer, uint256 nonce) external view returns (bool used) {
        return usedNonces[reviewer][nonce];
    }

    /**
     * @dev Validate score is within acceptable range
     * @param score The score to validate
     * @param minScore Minimum acceptable score
     * @param maxScore Maximum acceptable score
     * @return valid Whether the score is valid
     */
    function validateScore(
        uint256 score,
        uint256 minScore,
        uint256 maxScore
    ) external pure returns (bool valid) {
        return score >= minScore && score <= maxScore;
    }

    /**
     * @dev Calculate aggregated score from multiple reviews
     * @param scores Array of scores
     * @param method Aggregation method (0 = mean, 1 = median)
     * @return aggregatedScore The calculated aggregated score
     */
    function aggregateScores(
        uint256[] memory scores,
        uint256 method
    ) external pure returns (uint256 aggregatedScore) {
        if (scores.length == 0) return 0;

        if (method == 0) {
            // Mean
            uint256 sum = 0;
            for (uint256 i = 0; i < scores.length; i++) {
                sum += scores[i];
            }
            return sum / scores.length;
        } else if (method == 1) {
            // Median
            uint256[] memory sortedScores = _quickSort(scores);
            uint256 length = sortedScores.length;
            
            if (length % 2 == 0) {
                return (sortedScores[length / 2 - 1] + sortedScores[length / 2]) / 2;
            } else {
                return sortedScores[length / 2];
            }
        }
        
        return 0;
    }

    /**
     * @dev Internal function to sort scores using quicksort
     * @param arr Array to sort
     * @return sorted The sorted array
     */
    function _quickSort(uint256[] memory arr) internal pure returns (uint256[] memory sorted) {
        sorted = new uint256[](arr.length);
        for (uint256 i = 0; i < arr.length; i++) {
            sorted[i] = arr[i];
        }
        
        if (sorted.length <= 1) return sorted;
        
        _quickSortRecursive(sorted, 0, int256(sorted.length - 1));
        return sorted;
    }

    /**
     * @dev Recursive quicksort implementation
     */
    function _quickSortRecursive(uint256[] memory arr, int256 left, int256 right) internal pure {
        if (left < right) {
            int256 pivotIndex = _partition(arr, left, right);
            _quickSortRecursive(arr, left, pivotIndex - 1);
            _quickSortRecursive(arr, pivotIndex + 1, right);
        }
    }

    /**
     * @dev Partition function for quicksort
     */
    function _partition(uint256[] memory arr, int256 left, int256 right) internal pure returns (int256) {
        uint256 pivot = arr[uint256(right)];
        int256 i = left - 1;

        for (int256 j = left; j < right; j++) {
            if (arr[uint256(j)] <= pivot) {
                i++;
                (arr[uint256(i)], arr[uint256(j)]) = (arr[uint256(j)], arr[uint256(i)]);
            }
        }
        
        (arr[uint256(i + 1)], arr[uint256(right)]) = (arr[uint256(right)], arr[uint256(i + 1)]);
        return i + 1;
    }
}
