// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./interfaces/IContest.sol";
import "./interfaces/IEpochWeightCalculator.sol";
import "./InfToken.sol";
import "./ModelRegistry.sol";
import "./ZKVerifierRegistry.sol";
import "@openzeppelin/contracts/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

contract Contest is IContest, Initializable, Ownable {
    ContestInfo public contestInfo;
    mapping(address => Participant) public participants;
    mapping(address => uint256) public claimableRewards;
    address[] public participantsList;
    
    // Epoch management
    mapping(uint256 => EpochInfo) public epochs;
    mapping(uint256 => InferenceSubmission[]) public epochInferenceSubmissions;
    mapping(uint256 => ScoringSubmission[]) public epochScoringSubmissions;
    mapping(uint256 => mapping(address => mapping(bytes32 => bool))) public hasSubmittedInference; // epoch -> participant -> inputHash -> bool
    mapping(uint256 => mapping(address => mapping(bytes32 => mapping(address => bool)))) public hasSubmittedScoring; // epoch -> scorer -> inputHash -> participant -> bool
    
    // Participant scores tracking per epoch
    mapping(uint256 => mapping(address => uint256[])) public participantEpochScores; // epoch -> participant -> scores array
    
    // Validator rewards tracking
    mapping(uint256 => mapping(address => uint256)) public validatorScores; // epoch -> validator -> score count
    mapping(uint256 => address[]) public epochValidators; // epoch -> validators who participated
    uint256 public validatorRewardRatio = 10; // 10% default, in basis points (100 = 1%)
    
    // Reward distribution contract
    address public rewardDistributor;
    
    // Epoch weight calculator contract
    IEpochWeightCalculator public epochWeightCalculator;
    
    Winner[] public winners;
    
    InfToken public infToken;
    ModelRegistry public modelRegistry;
    ZKVerifierRegistry public verifierRegistry;
    
    bool private initialized;
    
    modifier onlyActive() {
        require(contestInfo.status == ContestStatus.Active, "Contest not active");
        _;
    }
    
    modifier onlyCreator() {
        require(msg.sender == contestInfo.creator, "Only creator can call this");
        _;
    }
    
    modifier onlyParticipant() {
        require(participants[msg.sender].isActive, "Not a participant");
        _;
    }
    
    constructor() Ownable(msg.sender) {}
    
    function initialize(
        address creator,
        string calldata metadataURI,
        uint256 duration,
        uint256 epochDuration,
        bytes32 scoringModelHash,
        address _modelRegistry,
        address _verifierRegistry
    ) external override initializer {
        require(!initialized, "Already initialized");
        require(creator != address(0), "Invalid creator");
        require(_modelRegistry != address(0), "Invalid model registry");
        require(_verifierRegistry != address(0), "Invalid verifier registry");
        require(epochDuration > 0, "Epoch duration must be positive");
        
        // Verify scoring model hash exists in verifier registry
        verifierRegistry = ZKVerifierRegistry(_verifierRegistry);
        require(verifierRegistry.isVerifierActive(scoringModelHash), "Scoring model not registered or inactive");
        
        modelRegistry = ModelRegistry(_modelRegistry);
        
        contestInfo = ContestInfo({
            creator: creator,
            metadataURI: metadataURI,
            startTime: block.timestamp,
            duration: duration,
            status: ContestStatus.Active,
            totalRewards: 0,
            participantCount: 0,
            scoringModelHash: scoringModelHash,
            currentEpoch: 0,
            epochDuration: epochDuration,
            modelRegistry: _modelRegistry,
            verifierRegistry: _verifierRegistry
        });
        
        // Create initial epoch
        _createNewEpoch();
        
        initialized = true;
        _transferOwnership(creator);
        
        emit ContestInitialized(
            creator, 
            metadataURI, 
            duration, 
            epochDuration,
            scoringModelHash, 
            _modelRegistry,
            _verifierRegistry
        );
    }

    function setInfToken(address _infToken) external onlyCreator {
        require(_infToken != address(0), "Invalid InfToken");
        infToken = InfToken(_infToken);
    }
    
    function setValidatorRewardRatio(uint256 _ratio) external onlyCreator {
        require(_ratio <= 500, "Ratio cannot exceed 50%"); // Max 50% for validators
        require(_ratio > 0, "Ratio must be positive");
        validatorRewardRatio = _ratio;
        emit ValidatorRewardRatioUpdated(_ratio);
    }
    
    function setRewardDistributor(address _rewardDistributor) external onlyCreator {
        require(_rewardDistributor != address(0), "Invalid reward distributor");
        rewardDistributor = _rewardDistributor;
        emit RewardDistributorUpdated(_rewardDistributor);
    }
    
    function setEpochWeightCalculator(address _epochWeightCalculator) external onlyCreator {
        require(_epochWeightCalculator != address(0), "Invalid epoch weight calculator");
        epochWeightCalculator = IEpochWeightCalculator(_epochWeightCalculator);
        emit EpochWeightCalculatorUpdated(_epochWeightCalculator);
    }
    w
    // Internal function to get participants who submitted in this epoch
    function _getEpochParticipants(uint256 epochNumber) internal view returns (address[] memory) {
        uint256 participantCount = 0;
        
        // First pass: count participants
        for (uint256 i = 0; i < participantsList.length; i++) {
            address participant = participantsList[i];
            if (participants[participant].isActive && 
                participantEpochScores[epochNumber][participant].length > 0) {
                participantCount++;
            }
        }
        
        // Second pass: collect participants
        address[] memory epochParticipants = new address[](participantCount);
        uint256 index = 0;
        for (uint256 i = 0; i < participantsList.length; i++) {
            address participant = participantsList[i];
            if (participants[participant].isActive && 
                participantEpochScores[epochNumber][participant].length > 0) {
                epochParticipants[index] = participant;
                index++;
            }
        }
        
        return epochParticipants;
    }
    

    
    function joinContest() external override onlyActive {
        require(!participants[msg.sender].isActive, "Already a participant");
        
        participants[msg.sender] = Participant({
            account: msg.sender,
            joinedAt: block.timestamp,
            totalRewards: 0,
            isActive: true,
            score: 0
        });
        
        participantsList.push(msg.sender);
        contestInfo.participantCount++;
        
        emit ParticipantJoined(msg.sender, block.timestamp);
    }

    function leaveContest() external override onlyParticipant {
        require(contestInfo.status == ContestStatus.Active, "Contest not active");
        
        participants[msg.sender].isActive = false;
        contestInfo.participantCount--;
        
        // Remove from participants list
        for (uint256 i = 0; i < participantsList.length; i++) {
            if (participantsList[i] == msg.sender) {
                participantsList[i] = participantsList[participantsList.length - 1];
                participantsList.pop();
                break;
            }
        }
        
        emit ParticipantLeft(msg.sender, block.timestamp);
    }

    // Epoch Management
    function postInputHashes(bytes32[] calldata inputHashes) external override onlyCreator onlyActive {
        require(inputHashes.length > 0, "Must provide at least one input hash");
        
        uint256 currentEpoch = contestInfo.currentEpoch;
        EpochInfo storage epoch = epochs[currentEpoch];
        
        // Add input hashes to current epoch
        for (uint256 i = 0; i < inputHashes.length; i++) {
            require(inputHashes[i] != bytes32(0), "Invalid input hash");
            epoch.inputHashes.push(inputHashes[i]);
        }
        
        emit InputHashesPosted(currentEpoch, inputHashes);
    }
    
    function startNewEpoch() external override onlyCreator {
        require(contestInfo.status == ContestStatus.Active, "Contest not active");
        
        uint256 currentEpoch = contestInfo.currentEpoch;
        
        // Finalize current epoch if it has submissions
        if (epochs[currentEpoch].totalSubmissions > 0) {
            epochs[currentEpoch].finalized = true;
            emit EpochFinalized(currentEpoch, epochs[currentEpoch].totalSubmissions, epochs[currentEpoch].totalScores);
        }
        
        // Create new epoch
        _createNewEpoch();
    }
    
    function _createNewEpoch() internal {
        uint256 newEpochNumber = contestInfo.currentEpoch + 1;
        contestInfo.currentEpoch = newEpochNumber;
        
        uint256 startTime = block.timestamp;
        uint256 endTime = startTime + contestInfo.epochDuration;
        
        epochs[newEpochNumber] = EpochInfo({
            epochNumber: newEpochNumber,
            startTime: startTime,
            endTime: endTime,
            inputHashes: new bytes32[](0),
            finalized: false,
            totalSubmissions: 0,
            totalScores: 0
        });
        
        emit EpochStarted(newEpochNumber, startTime, endTime);
    }
    
    // Submission Functions
    function submitInference(
        uint256 epochNumber,
        bytes32 inputHash,
        bytes32 modelHash,
        bytes32 outputHash,
        bytes calldata zkProof,
        uint256[] calldata publicInputs
    ) external override onlyParticipant onlyActive {
        require(epochNumber <= contestInfo.currentEpoch, "Invalid epoch number");
        require(!epochs[epochNumber].finalized, "Epoch already finalized");
        require(inputHash != bytes32(0), "Invalid input hash");
        require(modelHash != bytes32(0), "Invalid model hash");
        require(outputHash != bytes32(0), "Invalid output hash");
        require(zkProof.length > 0, "Missing ZK proof");
        require(!hasSubmittedInference[epochNumber][msg.sender][inputHash], "Already submitted for this input");
        
        // Verify input hash is valid for this epoch
        bool validInputHash = false;
        bytes32[] memory epochInputHashes = epochs[epochNumber].inputHashes;
        for (uint256 i = 0; i < epochInputHashes.length; i++) {
            if (epochInputHashes[i] == inputHash) {
                validInputHash = true;
                break;
            }
        }
        require(validInputHash, "Input hash not valid for this epoch");
        
        // Verify model is registered and active
        require(modelRegistry.isModelActive(modelHash), "Model not registered or inactive");
        
        // Record model usage
        modelRegistry.recordModelUsage(modelHash);
        
        bool isValid = verifierRegistry.verifyProof(modelHash, zkProof, publicInputs);
        require(isValid, "Invalid ZK proof");
        
        // Create submission
        InferenceSubmission memory submission = InferenceSubmission({
            participant: msg.sender,
            epochNumber: epochNumber,
            inputHash: inputHash,
            modelHash: modelHash,
            outputHash: outputHash,
            zkProof: zkProof,
            timestamp: block.timestamp,
            scored: false
        });
        
        epochInferenceSubmissions[epochNumber].push(submission);
        hasSubmittedInference[epochNumber][msg.sender][inputHash] = true;
        epochs[epochNumber].totalSubmissions++;
        
        emit InferenceSubmitted(msg.sender, epochNumber, inputHash, modelHash, outputHash, block.timestamp);
    }
    
    function submitScoring(
        uint256 epochNumber,
        bytes32 inputHash,
        address participant,
        bytes32 outputHash,
        uint256[] calldata scores,
        bytes calldata zkProof,
        uint256[] calldata publicInputs
    ) external override onlyValidator onlyActive {
        require(epochNumber <= contestInfo.currentEpoch, "Invalid epoch number");
        require(!epochs[epochNumber].finalized, "Epoch already finalized");
        require(inputHash != bytes32(0), "Invalid input hash");
        require(participant != address(0), "Invalid participant");
        require(outputHash != bytes32(0), "Invalid output hash");
        require(scores.length > 0, "Must provide scores");
        require(zkProof.length > 0, "Missing ZK proof");
        require(!hasSubmittedScoring[epochNumber][msg.sender][inputHash][participant], "Already submitted scoring for this combination");
        
        // Verify participant submitted inference for this input/output combination
        bool foundInference = false;
        InferenceSubmission[] memory inferences = epochInferenceSubmissions[epochNumber];
        uint256 inferenceIndex;
        for (uint256 i = 0; i < inferences.length; i++) {
            if (inferences[i].participant == participant && 
                inferences[i].inputHash == inputHash && 
                inferences[i].outputHash == outputHash) {
                foundInference = true;
                inferenceIndex = i;
                break;
            }
        }
        require(foundInference, "No matching inference submission found");
        
        // Verify ZK proof using the scoring model's verifier
        bool isValid = verifierRegistry.verifyProof(contestInfo.scoringModelHash, zkProof, publicInputs);
        require(isValid, "Invalid ZK proof");
        
        // Create scoring submission
        ScoringSubmission memory scoringSubmission = ScoringSubmission({
            scorer: msg.sender,
            epochNumber: epochNumber,
            inputHash: inputHash,
            participant: participant,
            outputHash: outputHash,
            scores: scores,
            zkProof: zkProof,
            timestamp: block.timestamp,
            scored: true
        });
        
        epochScoringSubmissions[epochNumber].push(scoringSubmission);
        hasSubmittedScoring[epochNumber][msg.sender][inputHash][participant] = true;
        epochs[epochNumber].totalScores++;
        
        // Track validator participation and scores
        if (validatorScores[epochNumber][msg.sender] == 0) {
            // First time this validator is scoring in this epoch
            epochValidators[epochNumber].push(msg.sender);
        }
        validatorScores[epochNumber][msg.sender]++;
        
        // Mark corresponding inference as scored
        epochInferenceSubmissions[epochNumber][inferenceIndex].scored = true;
        
        // Store scores in epoch array for participant
        for (uint256 i = 0; i < scores.length; i++) {
            participantEpochScores[epochNumber][participant].push(scores[i]);
        }
        
        // Update participant score (will be calculated by epoch function later)
        // For now, keep simple average as fallback
        uint256 avgScore = 0;
        for (uint256 i = 0; i < scores.length; i++) {
            avgScore += scores[i];
        }
        avgScore = avgScore / scores.length;
        participants[participant].score = avgScore;
        
        emit ScoringSubmitted(msg.sender, epochNumber, inputHash, participant, outputHash, scores, block.timestamp);
        emit InferenceVerified(participant, epochNumber, true, msg.sender);
        emit ScoringVerified(msg.sender, epochNumber, true, address(this));
    }
    
    function finalizeEpoch(uint256 epochNumber) external override {
        require(epochNumber <= contestInfo.currentEpoch, "Invalid epoch number");
        require(!epochs[epochNumber].finalized, "Epoch already finalized");
        require(address(epochWeightCalculator) != address(0), "Epoch weight calculator not set");
        
        epochs[epochNumber].finalized = true;
        
        // Get epoch participants and their scores
        address[] memory participants = _getEpochParticipants(epochNumber);
        uint256[][] memory epochScores = new uint256[][](participants.length);
        
        for (uint256 i = 0; i < participants.length; i++) {
            epochScores[i] = participantEpochScores[epochNumber][participants[i]];
        }
        
        // Call contest-owner-defined epoch function
        IEpochWeightCalculator.ParticipantWeight[] memory weights = 
            epochWeightCalculator.calculateEpochWeights(epochNumber, participants, epochScores);
        
        // Update participant scores with calculated weights
        for (uint256 i = 0; i < weights.length; i++) {
            participants[weights[i].participant].score = weights[i].weight;
        }
        
        // Distribute rewards based on weights
        _distributeEpochRewards(epochNumber, weights);
        
        // Distribute validator rewards for this epoch
        _distributeValidatorRewards(epochNumber);
        
        emit EpochFinalized(epochNumber, epochs[epochNumber].totalSubmissions, epochs[epochNumber].totalScores);
    }
    
    function _distributeEpochRewards(uint256 epochNumber, IEpochWeightCalculator.ParticipantWeight[] memory weights) internal {
        uint256 epochRewards = epochs[epochNumber].totalRewards;
        uint256 totalWeight = 0;
        
        // Calculate total weight
        for (uint256 i = 0; i < weights.length; i++) {
            totalWeight += weights[i].weight;
        }
        
        // Distribute rewards based on weights
        for (uint256 i = 0; i < weights.length; i++) {
            if (weights[i].weight > 0 && totalWeight > 0) {
                uint256 reward = (epochRewards * weights[i].weight) / totalWeight;
                if (reward > 0) {
                    SafeERC20.safeTransfer(IERC20(address(infToken)), weights[i].participant, reward);
                    participants[weights[i].participant].totalRewards += reward;
                    emit ParticipantRewarded(weights[i].participant, epochNumber, reward, weights[i].weight);
                }
            }
        }
    }
    
    function _distributeValidatorRewards(uint256 epochNumber) internal {
        address[] memory validators = epochValidators[epochNumber];
        uint256 totalScores = epochs[epochNumber].totalScores;
        
        if (validators.length == 0 || totalScores == 0) {
            return;
        }
        
        // Calculate validator reward pool based on configurable ratio
        uint256 epochRewards = epochs[epochNumber].totalRewards;
        uint256 validatorRewardPool = (epochRewards * validatorRewardRatio) / 1000; // basis points
        uint256 totalValidatorScores = 0;
        
        // Calculate total validator scores
        for (uint256 i = 0; i < validators.length; i++) {
            totalValidatorScores += validatorScores[epochNumber][validators[i]];
        }
        
        // Distribute rewards proportionally
        for (uint256 i = 0; i < validators.length; i++) {
            address validator = validators[i];
            uint256 validatorScore = validatorScores[epochNumber][validator];
            
            if (validatorScore > 0 && totalValidatorScores > 0) {
                uint256 reward = (validatorRewardPool * validatorScore) / totalValidatorScores;
                if (reward > 0) {
                    SafeERC20.safeTransfer(IERC20(address(infToken)), validator, reward);
                    emit ValidatorRewarded(validator, epochNumber, reward, validatorScore);
                }
            }
        }
    }
    
    function finalizeContest() external override onlyCreator {
        require(contestInfo.status == ContestStatus.Active, "Contest not active");
        require(contestInfo.totalRewards > 0, "No rewards to distribute");
        
        // Calculate winners based on scores
        Winner[] memory calculatedWinners = calculateWinners();
        require(calculatedWinners.length > 0, "No valid winners");
        
        // Automatically distribute rewards to winners
        uint256 totalDistributed = 0;
        for (uint256 i = 0; i < calculatedWinners.length; i++) {
            Winner memory winner = calculatedWinners[i];
            
            // Direct transfer to winner (no claimableRewards needed)
            if (winner.reward > 0) {
                infToken.transfer(winner.participant, winner.reward);
                participants[winner.participant].totalRewards += winner.reward;
                totalDistributed += winner.reward;
            }
        }
        
        // Store winners
        delete winners;
        for (uint256 i = 0; i < calculatedWinners.length; i++) {
            winners.push(calculatedWinners[i]);
        }
        
        contestInfo.status = ContestStatus.Finalized;
        
        emit ContestFinalized(calculatedWinners, totalDistributed);
    }
    
    function pause() external override onlyCreator {
        require(contestInfo.status == ContestStatus.Active, "Contest not active");
        contestInfo.status = ContestStatus.Paused;
        emit ContestPaused();
    }
    
    function unpause() external override onlyCreator {
        require(contestInfo.status == ContestStatus.Paused, "Contest not paused");
        contestInfo.status = ContestStatus.Active;
        emit ContestUnpaused();
    }
    
    function receiveRewards(uint256 amount) external override {
        require(msg.sender == rewardDistributor, "Only reward distributor can call this");
        
        // For finite contests, always epoch 1. For infinite contests, use current epoch
        uint256 targetEpoch = contestInfo.duration == 0 ? contestInfo.currentEpoch : 1;
        
        // Update the reward pool for the target epoch
        epochs[targetEpoch].totalRewards += amount;
        contestInfo.totalRewards += amount;
        
        emit RewardsReceived(amount, targetEpoch);
    }
    
    function calculateWinners() public view override returns (Winner[] memory) {
        if (participantsList.length == 0) {
            return new Winner[](0);
        }
        
        // Sort participants by score (descending)
        address[] memory sortedParticipants = new address[](participantsList.length);
        uint256[] memory scores = new uint256[](participantsList.length);
        uint256 validCount = 0;
        
        for (uint256 i = 0; i < participantsList.length; i++) {
            address participant = participantsList[i];
            if (participants[participant].isActive && participants[participant].score > 0) {
                sortedParticipants[validCount] = participant;
                scores[validCount] = participants[participant].score;
                validCount++;
            }
        }
        
        if (validCount == 0) {
            return new Winner[](0);
        }
        
        // Simple bubble sort for top winners
        for (uint256 i = 0; i < validCount && i < 10; i++) { // Top 10 max
            for (uint256 j = i + 1; j < validCount; j++) {
                if (scores[j] > scores[i]) {
                    // Swap scores
                    uint256 tempScore = scores[i];
                    scores[i] = scores[j];
                    scores[j] = tempScore;
                    
                    // Swap participants
                    address tempParticipant = sortedParticipants[i];
                    sortedParticipants[i] = sortedParticipants[j];
                    sortedParticipants[j] = tempParticipant;
                }
            }
        }
        
        // Create winners array with proportional rewards
        uint256 winnerCount = validCount < 10 ? validCount : 10;
        Winner[] memory calculatedWinners = new Winner[](winnerCount);
        
        uint256 totalRewards = contestInfo.totalRewards;
        uint256 totalScore = 0;
        for (uint256 i = 0; i < winnerCount; i++) {
            totalScore += scores[i];
        }
        
        for (uint256 i = 0; i < winnerCount; i++) {
            uint256 reward = totalScore > 0 ? (totalRewards * scores[i]) / totalScore : totalRewards / winnerCount;
            
            calculatedWinners[i] = Winner({
                participant: sortedParticipants[i],
                score: scores[i],
                reward: reward
            });
        }
        
        return calculatedWinners;
    }

    // View Functions
    function getContestInfo() external view override returns (ContestInfo memory) {
        return contestInfo;
    }

    function getParticipant(address account) external view override returns (Participant memory) {
        return participants[account];
    }

    function getParticipants() external view override returns (address[] memory) {
        return participantsList;
    }

    function getEpochInfo(uint256 epochNumber) external view override returns (EpochInfo memory) {
        return epochs[epochNumber];
    }

    function getCurrentEpoch() external view override returns (uint256) {
        return contestInfo.currentEpoch;
    }

    function getInferenceSubmissions(uint256 epochNumber) external view override returns (InferenceSubmission[] memory) {
        return epochInferenceSubmissions[epochNumber];
    }

    function getScoringSubmissions(uint256 epochNumber) external view override returns (ScoringSubmission[] memory) {
        return epochScoringSubmissions[epochNumber];
    }

    function getWinners() external view override returns (Winner[] memory) {
        return winners;
    }

    function getClaimableRewards(address participant) external view override returns (uint256) {
        return claimableRewards[participant];
    }

    function isParticipant(address account) external view override returns (bool) {
        return participants[account].isActive;
    }

    function isActive() external view override returns (bool) {
        if (contestInfo.duration == 0) {
            return contestInfo.status == ContestStatus.Active;
        }
        return contestInfo.status == ContestStatus.Active && 
               block.timestamp < contestInfo.startTime + contestInfo.duration;
    }

    // Validator reward functions
    function getValidatorScore(uint256 epochNumber, address validator) external view returns (uint256) {
        return validatorScores[epochNumber][validator];
    }
    
    function getEpochValidators(uint256 epochNumber) external view returns (address[] memory) {
        return epochValidators[epochNumber];
    }
    
    function getValidatorRewardPool(uint256 epochNumber) external view returns (uint256) {
        uint256 epochRewards = epochs[epochNumber].totalRewards;
        return (epochRewards * validatorRewardRatio) / 1000; // basis points
    }
    
    function getValidatorRewardRatio() external view returns (uint256) {
        return validatorRewardRatio;
    }
    
    function getRewardDistributor() external view returns (address) {
        return rewardDistributor;
    }
    
    function getParticipantEpochScores(uint256 epochNumber, address participant) external view returns (uint256[] memory) {
        return participantEpochScores[epochNumber][participant];
    }
    
    function getEpochParticipants(uint256 epochNumber) external view returns (address[] memory) {
        return _getEpochParticipants(epochNumber);
    }
    
    function getEpochWeightCalculator() external view returns (address) {
        return address(epochWeightCalculator);
    }

    function getParticipantCount() external view returns (uint256) {
        return contestInfo.participantCount;
    }
} 