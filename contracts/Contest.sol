// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./interfaces/IContest.sol";
import "./InfToken.sol";
import "@openzeppelin/contracts/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "./MockVerifier.sol";

contract Contest is IContest, Initializable, Ownable {
    ContestInfo public contestInfo;
    mapping(address => Participant) public participants;
    mapping(address => uint256) public claimableRewards;
    mapping(address => bool) public validators;
    address[] public participantsList;
    address[] public validatorsList;
    Submission[] public submissions;
    Winner[] public winners;
    
    MockVerifier public verifier;
    InfToken public infToken;
    
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
        bytes32 scoringModelHash,
        address[] calldata initialValidators
    ) external override initializer {
        require(!initialized, "Already initialized");
        require(creator != address(0), "Invalid creator");
        require(initialValidators.length > 0, "Must have at least one validator");
        
        contestInfo = ContestInfo({
            creator: creator,
            metadataURI: metadataURI,
            startTime: block.timestamp,
            duration: duration,
            status: ContestStatus.Active,
            totalRewards: 0,
            participantCount: 0,
            scoringModelHash: scoringModelHash,
            validators: new address[](0)
        });
        
        // Add initial validators
        for (uint256 i = 0; i < initialValidators.length; i++) {
            require(initialValidators[i] != address(0), "Invalid validator");
            validators[initialValidators[i]] = true;
            validatorsList.push(initialValidators[i]);
        }
        contestInfo.validators = validatorsList;
        
        initialized = true;
        _transferOwnership(creator);
        
        emit ContestInitialized(creator, metadataURI, duration, scoringModelHash, initialValidators);
    }
    
    function setVerifier(address _verifier) external onlyCreator {
        require(_verifier != address(0), "Invalid verifier");
        verifier = MockVerifier(_verifier);
    }
    
    function setInfToken(address _infToken) external onlyCreator {
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
    
    function submitEntry(
        string calldata metadataURI,
        bytes32 inputHash,
        bytes32 outputHash,
        bytes calldata zkProof,
        uint256[] calldata publicInputs
    ) external override onlyParticipant onlyActive {
        require(verifier != MockVerifier(address(0)), "Verifier not set");
        require(zkProof.length > 0, "Missing ZK proof");
        require(publicInputs.length > 0, "Missing public inputs");
        
        // Verify the ZK proof
        bool isValid = verifier.verifyProof(zkProof, publicInputs);
        require(isValid, "Invalid ZK proof");
        
        // Calculate score from public inputs (assuming first input is score)
       // TODO: validate via validator set
        
        submissions.push(Submission({
            participant: msg.sender,
            metadataURI: metadataURI,
            timestamp: block.timestamp,
            inputHash: inputHash,
            outputHash: outputHash,
            zkProof: zkProof,
            verified: true,
            validator: msg.sender // For now, participant is their own validator
        }));
        
        emit EntrySubmitted(msg.sender, metadataURI, inputHash, outputHash, block.timestamp);
        emit EntryVerified(msg.sender, true, score, msg.sender);
    }
    
    function finalizeContest() external override onlyCreator {
        require(contestInfo.status == ContestStatus.Active, "Contest not active");
        require(contestInfo.totalRewards > 0, "No rewards to distribute");
        
        // Calculate winners on-chain
        Winner[] memory calculatedWinners = calculateWinners();
        require(calculatedWinners.length > 0, "No valid winners");
        
        // Distribute rewards to winners
        uint256 totalDistributed = 0;
        for (uint256 i = 0; i < calculatedWinners.length; i++) {
            Winner memory winner = calculatedWinners[i];
            claimableRewards[winner.participant] += winner.reward;
            participants[winner.participant].totalRewards += winner.reward;
            totalDistributed += winner.reward;
        }
        
        // Store winners
        winners = calculatedWinners;
        
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
    
    function claimReward() external override {
        uint256 amount = claimableRewards[msg.sender];
        require(amount > 0, "No rewards to claim");
        
        claimableRewards[msg.sender] = 0;
        
        // Transfer InfToken
        infToken.transfer(msg.sender, amount);
        
        emit RewardClaimed(msg.sender, amount);
    }
    
    function distributeRewards(uint256 amount) external override onlyCreator {
        require(amount > 0, "No rewards to distribute");
        require(address(infToken) != address(0), "InfToken not set");
        
        // Transfer tokens from creator to contract
        infToken.transferFrom(msg.sender, address(this), amount);
        
        contestInfo.totalRewards += amount;
        emit RewardsDistributed(amount);
    }
    
    function calculateWinners() public view override returns (Winner[] memory) {
        if (submissions.length == 0) {
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
        
        // Simple bubble sort for top 3 winners
        for (uint256 i = 0; i < validCount && i < 3; i++) {
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
        
        // Create winners array (top 3 or less)
        uint256 winnerCount = validCount < 3 ? validCount : 3;
        Winner[] memory calculatedWinners = new Winner[](winnerCount);
        
        uint256 totalRewards = contestInfo.totalRewards;
        for (uint256 i = 0; i < winnerCount; i++) {
            uint256 reward;
            if (i == 0) {
                reward = (totalRewards * 50) / 100; // 50% for 1st place
            } else if (i == 1) {
                reward = (totalRewards * 30) / 100; // 30% for 2nd place
            } else {
                reward = (totalRewards * 20) / 100; // 20% for 3rd place
            }
            
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
    
    function getSubmissions() external view override returns (Submission[] memory) {
        return submissions;
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
} 