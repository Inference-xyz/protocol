// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./interfaces/IContest.sol";
import "./InfToken.sol";
import "./ModelRegistry.sol";
import "./ZKVerifierRegistry.sol";
import "@openzeppelin/contracts/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract Contest is IContest, Initializable, Ownable {
    ContestInfo public contestInfo;
    mapping(address => Participant) public participants;
    mapping(address => uint256) public claimableRewards;
    mapping(address => bool) public validators;
    address[] public participantsList;
    address[] public validatorsList;
    
    // Epoch management
    mapping(uint256 => EpochInfo) public epochs;
    mapping(uint256 => InferenceSubmission[]) public epochInferenceSubmissions;
    mapping(uint256 => ScoringSubmission[]) public epochScoringSubmissions;
    mapping(uint256 => mapping(address => mapping(bytes32 => bool))) public hasSubmittedInference; // epoch -> participant -> inputHash -> bool
    mapping(uint256 => mapping(address => mapping(bytes32 => mapping(address => bool)))) public hasSubmittedScoring; // epoch -> scorer -> inputHash -> participant -> bool
    
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
    
    modifier onlyValidator() {
        require(validators[msg.sender], "Only validators can call this");
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
        address _verifierRegistry,
        address[] calldata initialValidators
    ) external override initializer {
        require(!initialized, "Already initialized");
        require(creator != address(0), "Invalid creator");
        require(initialValidators.length > 0, "Must have at least one validator");
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
            validators: new address[](0),
            currentEpoch: 0,
            epochDuration: epochDuration,
            modelRegistry: _modelRegistry,
            verifierRegistry: _verifierRegistry
        });
        
        // Add initial validators
        for (uint256 i = 0; i < initialValidators.length; i++) {
            require(initialValidators[i] != address(0), "Invalid validator");
            validators[initialValidators[i]] = true;
            validatorsList.push(initialValidators[i]);
        }
        contestInfo.validators = validatorsList;
        
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
            _verifierRegistry,
            initialValidators
        );
    }

    function setInfToken(address _infToken) external {
        require(_infToken != address(0), "Invalid InfToken");
        infToken = InfToken(_infToken);
    }
    
    function addValidator(address validator) external override onlyCreator {
        require(validator != address(0), "Invalid validator");
        require(!validators[validator], "Already a validator");
        
        validators[validator] = true;
        validatorsList.push(validator);
        contestInfo.validators = validatorsList;
        
        emit ValidatorAdded(validator);
    }
    
    function removeValidator(address validator) external override onlyCreator {
        require(validators[validator], "Not a validator");
        require(validatorsList.length > 1, "Cannot remove last validator");
        
        validators[validator] = false;
        
        // Remove from validatorsList
        for (uint256 i = 0; i < validatorsList.length; i++) {
            if (validatorsList[i] == validator) {
                validatorsList[i] = validatorsList[validatorsList.length - 1];
                validatorsList.pop();
                break;
            }
        }
        contestInfo.validators = validatorsList;
        
        emit ValidatorRemoved(validator);
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
        
        // Create submission
        InferenceSubmission memory submission = InferenceSubmission({
            participant: msg.sender,
            epochNumber: epochNumber,
            inputHash: inputHash,
            modelHash: modelHash,
            outputHash: outputHash,
            zkProof: zkProof,
            timestamp: block.timestamp,
            verified: false // Will be verified when scoring is submitted
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
            verified: true
        });
        
        epochScoringSubmissions[epochNumber].push(scoringSubmission);
        hasSubmittedScoring[epochNumber][msg.sender][inputHash][participant] = true;
        epochs[epochNumber].totalScores++;
        
        // Mark corresponding inference as verified
        epochInferenceSubmissions[epochNumber][inferenceIndex].verified = true;
        
        // Update participant score (simple average for now)
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
    
    function finalizeEpoch(uint256 epochNumber) external override onlyCreator {
        require(epochNumber <= contestInfo.currentEpoch, "Invalid epoch number");
        require(!epochs[epochNumber].finalized, "Epoch already finalized");
        
        epochs[epochNumber].finalized = true;
        
        emit EpochFinalized(epochNumber, epochs[epochNumber].totalSubmissions, epochs[epochNumber].totalScores);
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
    
    function distributeRewards(uint256 amount) external override onlyCreator {
        require(amount > 0, "No rewards to distribute");
        require(address(infToken) != address(0), "InfToken not set");
        require(contestInfo.status == ContestStatus.Active, "Contest not active");
        
        // Transfer tokens from creator to contract
        infToken.transferFrom(msg.sender, address(this), amount);
        
        contestInfo.totalRewards += amount;
        emit RewardsDistributed(amount);
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

    function getValidators() external view override returns (address[] memory) {
        return validatorsList;
    }

    function isValidator(address validator) external view override returns (bool) {
        return validators[validator];
    }

    function getParticipantCount() external view returns (uint256) {
        return contestInfo.participantCount;
    }
} 