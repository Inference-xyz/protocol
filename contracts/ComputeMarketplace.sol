// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";

interface IHalo2Verifier {
    function verifyProof(bytes calldata proof, uint256[] calldata instances) external returns (bool);
}

contract ComputeMarketplace is Ownable {
    struct Job {
        uint256 id;
        address client;
        bytes32 modelHash;
        bytes32 inputHash;
        address provider;
        bool completed;
    }

    event JobPosted(uint256 indexed jobId, address indexed client, bytes32 indexed modelHash, bytes32 inputHash);
    event JobClaimed(uint256 indexed jobId, address indexed provider);
    event JobCompleted(uint256 indexed jobId, address indexed provider, bytes encryptedOutput, bytes zkProof);
    event ModelHashRegistered(bytes32 indexed modelHash, address indexed registrant);

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

    function postJob(bytes32 modelHash, bytes32 inputHash) external returns (uint256 jobId) {
        require(registeredModelHashes[modelHash], "Model not registered");

        jobId = nextJobId++;
        jobs[jobId] = Job({
            id: jobId,
            client: msg.sender,
            modelHash: modelHash,
            inputHash: inputHash,
            provider: address(0),
            completed: false
        });

        emit JobPosted(jobId, msg.sender, modelHash, inputHash);
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
        emit JobCompleted(jobId, msg.sender, encryptedOutput, zkProof);
    }

    function setVerifier(address _verifier) external onlyOwner {
        verifier = IHalo2Verifier(_verifier);
    }
}