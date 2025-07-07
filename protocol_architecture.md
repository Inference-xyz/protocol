# Inference Protocol Architecture - Current vs Proposed

## Current Architecture Overview

```mermaid
classDiagram
    class IInferenceERC20 {
        +mint(address to, uint256 amount)
        +burn(uint256 amount)
        +transfer(address to, uint256 amount)
    }
    
    class IRewardDistributor {
        +EmissionConfig emissionConfig
        +ContestWeight[] contestWeights
        +setComputeSplit(uint256 percent)
        +setContestWeight(address contest, uint256 weight)
        +checkpointEmissions()
        +distributeRewards()
    }
    
    class IContest {
        +ContestInfo contestInfo
        +Participant[] participants
        +EpochResult[] epochResults
        +joinContest()
        +submitEntry(bytes32 submissionHash)
        +finalizeEpoch(address[] winners, uint256[] rewards)
        +claimReward()
    }
    
    class IContestFactory {
        +createContest(ContestConfig config)
        +createContestWithInitialWeight(ContestConfig config, uint256 weight)
    }
    
    class IContestRegistry {
        +ContestSlot[] contestSlots
        +DutchAuction[] auctions
        +registerContest(address contest)
        +updateContestPerformance(address contest, uint256 score)
        +initiateReplacement(address newContest)
    }
    
    class IComputeMarketplace {
        +Job[] jobs
        +ProviderStats[] providerStats
        +postJob(string specURI, uint256 bounty)
        +claimJob(uint256 jobId)
        +completeJob(uint256 jobId, bytes zkProof)
        +distributeComputeRewards(uint256 amount)
    }
    
    class IZKVerifier {
        +ProofData[] proofs
        +verifyProof(bytes proof, bytes32[] publicInputs)
        +batchVerifyProofs(ProofData[] proofs)
        +registerModel(bytes32 modelHash)
    }
    
    IRewardDistributor --> IInferenceERC20 : distributes tokens
    IRewardDistributor --> IContest : distributes to contests
    IRewardDistributor --> IComputeMarketplace : distributes to compute
    IContestFactory --> IContest : creates
    IContestRegistry --> IContest : manages slots
    IComputeMarketplace --> IZKVerifier : verifies proofs
```

## Proposed Architecture Changes

### 1. Updated Reward Distribution System

```mermaid
classDiagram
    class IRewardDistributor {
        +EmissionConfig emissionConfig
        +ContestWeight[] contestWeights
        +uint256 minEmissionInterval
        +uint256 lastEmissionBlock
        +setContestWeight(address contest, uint256 weight)
        +checkpointEmissions() : uint256
        +distributeRewards() : tamper-proof
        +getEmissionAmount() : math-based
    }
    
    class IDAOGovernance {
        +proposeEmissionSplit(uint256[] contestWeights)
        +voteOnProposal(uint256 proposalId, bool support)
        +executeProposal(uint256 proposalId)
        +getActiveProposals()
    }
    
    class IContest {
        +ContestInfo contestInfo
        +Participant[] participants
        +ValidatorSet validators
        +RewardSplit rewardSplit
        +setRewardSplit(uint256 ownerPct, uint256 participantPct, uint256 validatorPct)
        +submitZKProof(bytes32[] inputs, bytes32[] outputs, bytes proof)
        +submitScores(address[] participants, uint256[] scores, bytes proof)
        +distributeEpochRewards()
    }
    
    class IValidatorSet {
        +Validator[] validators
        +uint256 maxValidators
        +uint256 totalStake
        +stake(uint256 amount)
        +unstake(uint256 amount)
        +setWeights(address[] participants, uint256[] weights)
        +getWeightedScores() : uint256[]
    }
    
    class IZKScoringVerifier {
        +verifyScoringProof(bytes proof, bytes32[] inputs, bytes32[] outputs, uint256[] scores)
        +batchVerifyScoringProofs(ProofData[] proofs)
        +registerScoringModel(bytes32 modelHash)
    }
    
    class IEZKLFactory {
        +deployVerificationContract(bytes32 modelHash, bytes circuit)
        +registerVerificationContract(address contract, bytes32 modelHash)
        +getVerificationContract(bytes32 modelHash) : address
    }
    
    IDAOGovernance --> IRewardDistributor : controls emission splits
    IRewardDistributor --> IContest : distributes to contests only
    IContest --> IValidatorSet : uses for scoring
    IContest --> IZKScoringVerifier : verifies scoring proofs
    IEZKLFactory --> IZKScoringVerifier : deploys verification contracts
```

### 2. Enhanced Contest System with ZK Proofs

```mermaid
sequenceDiagram
    participant Owner as Contest Owner
    participant Contest as IContest
    participant Validator as IValidatorSet
    participant ZKVerifier as IZKVerifier
    participant ScoringVerifier as IZKScoringVerifier
    participant Factory as IEZKLFactory
    
    Owner->>Contest: createContest(config)
    Contest->>Factory: deployVerificationContract(modelHash, circuit)
    Factory->>ScoringVerifier: deploy new verification contract
    
    Note over Contest,ZKVerifier: Epoch Execution
    loop For each participant
        Contest->>ZKVerifier: submitZKProof(inputs, outputs, proof)
        ZKVerifier-->>Contest: verification result
    end
    
    Note over Contest,ScoringVerifier: Scoring Phase
    Contest->>ScoringVerifier: submitScores(participants, scores, proof)
    ScoringVerifier-->>Contest: scoring verification result
    
    Note over Contest,Validator: Validator Weighting
    loop For each validator
        Validator->>Contest: setWeights(participants, weights)
    end
    
    Contest->>Contest: calculateWeightedRewards()
    Contest->>Contest: distributeEpochRewards()
```
