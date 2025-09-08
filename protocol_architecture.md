# Inference Protocol Architecture

## Overview

The Inference Protocol is a decentralized system for AI inference with an optimistic rollup design using EZKL ZK proofs for dispute resolution. The system enables trustless AI inference by allowing compute providers to optimistically execute jobs and only requiring ZK proofs when disputes arise.

## Core Components

### 1. Token System (`IInferenceERC20.sol`)
- Native ERC20 token for rewards and governance
- Minting capabilities for reward distribution

### 2. Reward Distributor (`IRewardDistributor.sol`)
- Manages token emissions across the protocol
- Tamper-proof reward distribution
- Math-based emission calculations

### 3. Optimistic Rollup Compute Marketplace

The core innovation is an optimistic rollup design for AI inference that combines the efficiency of optimistic execution with the security of ZK proofs for disputes.

#### Key Features:
- **Optimistic Execution**: Compute providers run inference immediately and post claims
- **Dispute Period**: Time window for challenges before finalization
- **Two-Way Verification Game**: Binary search to find disputed step
- **EZKL ZK Proofs**: Cryptographic proof of incorrect computation at specific step
- **Bond Mechanism**: Economic incentives for honest behavior

#### Workflow:
```mermaid
sequenceDiagram
    participant User
    participant Marketplace
    participant Provider
    participant Challenger
    participant DisputeMechanism

    User->>Marketplace: openJob(modelId, inputHash, rngSeed, maxTokens, payment, timeout)
    Provider->>Marketplace: submitClaim(jobId, finalHash, outputCommit, bond)
    
    alt No Challenge
        Marketplace->>Provider: finalizeJob() - release payment
    else Challenge
        Challenger->>Marketplace: challenge(jobId)
        Marketplace->>DisputeMechanism: initiateDispute()
        
        loop Binary Search Game
            Provider->>DisputeMechanism: submitMove(midpoint, hash)
            Challenger->>DisputeMechanism: submitMove(midpoint, hash)
        end
        
        Challenger->>DisputeMechanism: proveOneStep(step, hashS, hashS+1, ezklProof)
        DisputeMechanism->>Marketplace: resolveDispute(winner, bond)
    end
```

#### Interface Design:
```mermaid
classDiagram
    class IComputeMarketplace {
        +openJob(modelId, inputHash, rngSeed, maxTokens, payment, timeout)
        +submitClaim(jobId, finalHash, outputCommit, bond)
        +challenge(jobId)
        +submitMove(disputeId, midpoint, hash)
        +proveOneStep(disputeId, step, hashS, hashS+1, proof)
    }
    
    class IDisputeMechanism {
        +initiateDispute(jobId, challenger, provider, leftBound, rightBound)
        +submitMove(disputeId, midpoint, hash)
        +proveOneStep(disputeId, step, hashS, hashS+1, ezklProof)
        +getCurrentBounds(disputeId)
    }
    
    class IEZKLVerifier {
        +verifyProof(proof, publicInputs)
        +verifyOneStep(step, hashS, hashS+1, proof)
    }
    
    IComputeMarketplace --> IDisputeMechanism : uses for disputes
    IDisputeMechanism --> IEZKLVerifier : verifies ZK proofs
```

### 4. Contest System

#### Variant 1: Validator Set Only
```mermaid
classDiagram
    class IContest {
        +ContestInfo contestInfo
        +Participant[] participants
        +ValidatorSet validators
        +RewardSplit rewardSplit
        +joinContest()
        +submitEntry(bytes32 submissionHash)
        +finalizeEpoch()
        +distributeRewards()
    }
    
    class IValidatorSet {
        +Validator[] validators
        +uint256 totalStake
        +stake(uint256 amount)
        +unstake(uint256 amount)
        +submitScores(address[] participants, uint256[] scores)
        +getWeightedScores() : uint256[]
    }
    
    IContest --> IValidatorSet : uses for scoring
```

**Key Features:**
- Stake-weighted validator selection
- Validators directly score submissions
- Consensus-based final scoring
- Slashing mechanism for malicious behavior
- Contest owner controls reward percentages

#### Variant 2: ZK Proof Only
```mermaid
classDiagram
    class IContest {
        +ContestInfo contestInfo
        +Participant[] participants
        +IZKVerifier computationVerifier
        +IZKScoringVerifier scoringVerifier
        +joinContest()
        +submitZKProof(bytes32[] inputs, bytes32[] outputs, bytes proof)
        +submitScoringProof(bytes32[] outputs, uint256[] scores, bytes proof)
        +distributeRewards()
    }
    
    class IZKVerifier {
        +verifyProof(bytes proof, bytes32[] inputs, bytes32[] outputs)
        +registerModel(bytes32 modelHash)
    }
    
    class IZKScoringVerifier {
        +verifyScoringProof(bytes proof, bytes32[] outputs, uint256[] scores)
        +registerScoringModel(bytes32 modelHash)
    }
    
    IContest --> IZKVerifier : verifies computation
    IContest --> IZKScoringVerifier : verifies scoring
```

**Key Features:**
- Participants prove computation (input → output)
- Single verifier proves scoring function execution
- Contest owner deploys scoring verifier contract
- Participants can deploy computation verifiers
- Tamper-proof reward distribution

### 4. Contest Factory (`IContestFactory.sol`)
- Creates contests with configurable parameters
- Supports both validator set and ZK proof variants

### 5. Contest Registry (`IContestRegistry.sol`)
- Manages contest slots with performance-based ranking
- Dutch auction mechanism for slot replacement

## Design Comparison

| Aspect | Optimistic Rollup | Validator Set | ZK Proof Only |
|--------|------------------|---------------|---------------|
| **Decentralization** | High (anyone can challenge) | High (multiple validators) | Medium (single verifier) |
| **Gas Costs** | Low (only on disputes) | Low (no ZK verification) | High (ZK proof verification) |
| **Trust Model** | Trust in economic incentives | Trust in validator consensus | Trust in ZK proof system |
| **Scalability** | High (optimistic execution) | Limited by validator count | Limited by ZK proof costs |
| **Complexity** | Medium (dispute mechanism) | Simple consensus mechanism | Complex ZK proof integration |
| **Latency** | Fast (immediate results) | Medium (consensus time) | Slow (proof generation) |

## Implementation Flow

### Optimistic Rollup Variant
1. User posts inference job with model_id, input_hash, rng_seed, max_tokens, payment, timeout
2. Compute provider runs inference immediately and returns result
3. Provider submits claim with final_hash, output_commit, and bond
4. Dispute period begins - anyone can challenge the claim
5. If no challenge: funds released to provider after dispute period
6. If challenged: two-way verification game begins

#### Detailed Optimistic Rollup Flow:
```mermaid
flowchart TD
    A[User: openJob] --> B[Provider: Run Inference]
    B --> C[Provider: submitClaim with bond]
    C --> D{Dispute Period}
    D -->|No Challenge| E[Finalize: Release Payment]
    D -->|Challenge| F[Initiate Dispute Game]
    F --> G["Binary Search: L,R = 0,N"]
    G --> H[Provider: submitMove]
    H --> I[Challenger: submitMove]
    I --> J{"R-L = 1?"}
    J -->|No| K[Update Bounds]
    K --> H
    J -->|Yes| L[Challenger: proveOneStep with EZKL]
    L --> M{"Proof Valid?"}
    M -->|Yes| N[Challenger Wins Bond]
    M -->|No| O[Provider Wins Bond]
```

### Validator Set Variant
1. Contest owner creates contest with validator parameters
2. Validators stake tokens to participate
3. Participants submit inference results
4. Validators independently score submissions
5. Stake-weighted consensus determines final scores
6. Rewards distributed based on consensus scores

### ZK Proof Variant
1. Contest owner deploys scoring verifier contract
2. Participants submit inference results with ZK proofs
3. Verifier runs scoring function and generates ZK proof
4. Smart contract verifies all proofs on-chain
5. Rewards automatically distributed based on verified scores
