// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title ReviewStructs
 * @notice Shared data structures for commit-reveal submission and peer review system
 * @dev Contains EIP-712 type hashes for signature verification (currently unused, reserved for future)
 */
library ReviewStructs {
    // EIP-712 Domain
    bytes32 public constant DOMAIN_TYPEHASH = keccak256(
        "EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"
    );

    // Review signature type hash
    bytes32 public constant REVIEW_TYPEHASH = keccak256(
        "Review(uint256 submissionId,uint256 contestId,uint256 score,address reviewer,uint256 nonce,uint256 deadline)"
    );

    /**
     * @dev Structure for a submission
     */
    struct Submission {
        address participant;
        bytes32 commitHash;      // Hash of the output
        string ipfsURI;          // IPFS URI for the submission
        bytes32 outputHash;      // Hash of the actual output
        uint256 commitTime;      // When the submission was made
        uint256 revealTime;      // When the submission was made
        bool isRevealed;         // Always true in simplified version
        uint256 aggregatedScore; // Final aggregated score from reviews
        uint256 reviewCount;     // Number of reviews received
    }

    /**
     * @dev Structure for a signed peer review
     */
    struct SignedReview {
        uint256 submissionId;    // ID of the submission being reviewed
        uint256 contestId;       // ID of the contest
        uint256 score;           // Score given
        address reviewer;        // Address of the reviewer
        uint256 nonce;           // Nonce to prevent replay attacks
        uint256 deadline;        // Review deadline timestamp
        bytes signature;         // EIP-712 signature
    }

    /**
     * @dev Structure for review assignment
     */
    struct ReviewAssignment {
        address reviewer;        // Who should review
        uint256 submissionId;    // Which submission to review
        bool completed;          // Whether review was submitted
        uint256 assignedAt;      // When assignment was made
    }

    /**
     * @dev Structure for a review with commit-reveal
     */
    struct Review {
        address reviewer;        // Who submitted the review
        bytes32 commitHash;      // Hash of (submissionId, score, nonce)
        uint256 score;           // Actual score (0 until revealed)
        uint256 commitTime;      // When the review was committed
        uint256 revealTime;      // When the review was revealed
        bool isRevealed;         // Whether the review has been revealed
    }

    /**
     * @dev Structure for contest phases and deadlines
     */
    struct ContestPhases {
        uint256 commitDeadline;  // Deadline for commit phase
        uint256 revealDeadline;  // Deadline for reveal phase
        uint256 reviewDeadline;  // Deadline for review phase
        uint256 reviewersPerSubmission; // Number of reviewers per submission
    }

    /**
     * @notice Contest lifecycle phases (duration divided into 4 equal parts)
     */
    enum Phase {
        Commit,         // Phase 1 (0-25%): Participants commit output hashes
        Reveal,         // Phase 2 (25-50%): Participants reveal outputs with nonces
        ReviewCommit,   // Phase 3 (50-75%): Reviewers commit review scores
        ReviewReveal,   // Phase 4 (75-100%): Reviewers reveal scores, aggregation happens
        Finalized       // After duration: Owner can finalize and distribute rewards
    }

    /**
     * @dev Generate EIP-712 domain separator
     */
    function getDomainSeparator(
        string memory name,
        string memory version,
        uint256 chainId,
        address verifyingContract
    ) internal pure returns (bytes32) {
        return keccak256(
            abi.encode(
                DOMAIN_TYPEHASH,
                keccak256(bytes(name)),
                keccak256(bytes(version)),
                chainId,
                verifyingContract
            )
        );
    }

    /**
     * @dev Generate struct hash for Review
     */
    function getReviewStructHash(SignedReview memory review) internal pure returns (bytes32) {
        return keccak256(
            abi.encode(
                REVIEW_TYPEHASH,
                review.submissionId,
                review.contestId,
                review.score,
                review.reviewer,
                review.nonce,
                review.deadline
            )
        );
    }

    /**
     * @dev Generate EIP-712 typed data hash for Review
     */
    function getReviewTypedDataHash(
        SignedReview memory review,
        bytes32 domainSeparator
    ) internal pure returns (bytes32) {
        return keccak256(
            abi.encodePacked(
                "\x19\x01",
                domainSeparator,
                getReviewStructHash(review)
            )
        );
    }
}
