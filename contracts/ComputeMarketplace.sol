// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

interface IHalo2Verifier {
    function verifyProof(bytes calldata proof, uint256[] calldata instances) external returns (bool);
}

contract ComputeMarketplace is Ownable {
    struct Job {
        uint256 id;
        address client;
        bytes32 modelHash;
        bytes32[] inputHashes;
        address provider;
        bool completed;
        uint256 paymentAmount;
        address paymentToken;
    }

    event JobPosted(uint256 indexed jobId, address indexed client, bytes32 indexed modelHash, bytes32[] inputHashes, uint256 paymentAmount, address paymentToken);
    event JobClaimed(uint256 indexed jobId, address indexed provider);
    event JobCompleted(uint256 indexed jobId, address indexed provider, bytes encryptedOutput, bytes zkProof);
    event ModelHashRegistered(bytes32 indexed modelHash, address indexed registrant);
    event PaymentReleased(uint256 indexed jobId, address indexed provider, uint256 amount, address token);

    uint256 public nextJobId = 1;
    IHalo2Verifier public verifier;

    mapping(uint256 => Job) public jobs;
    mapping(bytes32 => bool) public registeredModelHashes;
    mapping(bytes32 => address) public modelHashOwners;

    constructor(address _verifier) Ownable(msg.sender) {
        verifier = IHalo2Verifier(_verifier);
    }

    function registerModelHash(bytes32 modelHash) external {
        require(!registeredModelHashes[modelHash], "Model already registered");
        registeredModelHashes[modelHash] = true;
        modelHashOwners[modelHash] = msg.sender;
        emit ModelHashRegistered(modelHash, msg.sender);
    }

    function postJob(
        bytes32 modelHash, 
        bytes32[] calldata inputHashes, 
        uint256 paymentAmount, 
        address paymentToken
    ) external returns (uint256 jobId) {
        require(registeredModelHashes[modelHash], "Model not registered");
        require(inputHashes.length > 0, "Must provide at least one input hash");
        require(paymentAmount > 0, "Payment amount must be greater than 0");
        require(paymentToken != address(0), "Invalid payment token");

        // Transfer payment tokens from client to this contract
        IERC20(paymentToken).transferFrom(msg.sender, address(this), paymentAmount);

        jobId = nextJobId++;
        jobs[jobId] = Job({
            id: jobId,
            client: msg.sender,
            modelHash: modelHash,
            inputHashes: inputHashes,
            provider: address(0),
            completed: false,
            paymentAmount: paymentAmount,
            paymentToken: paymentToken
        });

        emit JobPosted(jobId, msg.sender, modelHash, inputHashes, paymentAmount, paymentToken);
    }

    function claimJob(uint256 jobId) external {
        Job storage job = jobs[jobId];
        require(job.id != 0, "Invalid job");
        require(job.provider == address(0), "Already claimed");
        require(!job.completed, "Already completed");

        job.provider = msg.sender;
        emit JobClaimed(jobId, msg.sender);
    }

    function completeJob(uint256 jobId, bytes calldata encryptedOutput, bytes calldata zkProof, uint256[] calldata publicInputs) external {
        Job storage job = jobs[jobId];
        require(job.id != 0, "Invalid job");
        require(msg.sender == job.provider, "Not job provider");
        require(!job.completed, "Already completed");
        require(encryptedOutput.length > 0, "Missing output");
        require(zkProof.length > 0, "Missing proof");

        bool isValid = verifier.verifyProof(zkProof, publicInputs);
        require(isValid, "Invalid ZK proof");

        job.completed = true;
        
        // Release payment to provider
        IERC20(job.paymentToken).transfer(job.provider, job.paymentAmount);
        
        emit JobCompleted(jobId, msg.sender, encryptedOutput, zkProof);
        emit PaymentReleased(jobId, job.provider, job.paymentAmount, job.paymentToken);
    }

    function setVerifier(address _verifier) external onlyOwner {
        verifier = IHalo2Verifier(_verifier);
    }

    // Emergency function to allow owner to withdraw stuck tokens
    function emergencyWithdraw(address token, address to, uint256 amount) external onlyOwner {
        IERC20(token).transfer(to, amount);
    }
}