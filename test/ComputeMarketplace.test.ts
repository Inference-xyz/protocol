import { expect } from 'chai';
import { ethers } from 'ethers';
import * as fs from 'fs';
import * as path from 'path';

interface ContractArtifact {
  abi: any[];
  bytecode: string;
}

describe('ComputeMarketplace', () => {
  let provider: ethers.JsonRpcProvider;
  let deployer: ethers.Wallet;
  let user1: ethers.Wallet;
  let user2: ethers.Wallet;
  let computeMarketplace: ethers.Contract;
  let inferenceToken: ethers.Contract;

  const loadContractArtifact = (contractName: string): ContractArtifact => {
    const artifactPath = path.join(process.cwd(), 'out', contractName, `${contractName}.json`);
    if (!fs.existsSync(artifactPath)) {
      throw new Error(`Contract artifact not found: ${artifactPath}`);
    }
    return JSON.parse(fs.readFileSync(artifactPath, 'utf8'));
  };

  beforeEach(async () => {
    // Setup local test provider
    provider = new ethers.JsonRpcProvider('http://localhost:8545');
    
    // Create test accounts
    deployer = new ethers.Wallet('0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d', provider);
    user1 = new ethers.Wallet('0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a', provider);
    user2 = new ethers.Wallet('0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6', provider);

    // Load contract artifacts
    const tokenArtifact = loadContractArtifact('InferenceERC20.sol/InferenceERC20');
    const marketplaceArtifact = loadContractArtifact('ComputeMarketplace.sol/ComputeMarketplace');

    // Deploy InferenceERC20
    const tokenFactory = new ethers.ContractFactory(
      tokenArtifact.abi,
      tokenArtifact.bytecode,
      deployer
    );
    
    inferenceToken = await tokenFactory.deploy(
      'Inference Token',
      'INF',
      18,
      deployer.address
    );
    await inferenceToken.waitForDeployment();

    // Deploy ComputeMarketplace
    const marketplaceFactory = new ethers.ContractFactory(
      marketplaceArtifact.abi,
      marketplaceArtifact.bytecode,
      deployer
    );
    
    computeMarketplace = await marketplaceFactory.deploy(
      deployer.address, // admin
      ethers.ZeroAddress // zkVerifier
    );
    await computeMarketplace.waitForDeployment();

    // Initial setup
    await computeMarketplace.getFunction('setSupportedTokens')([await inferenceToken.getAddress()]);
    await computeMarketplace.getFunction('setMinBounty')(ethers.parseEther('1.0'));
    await computeMarketplace.getFunction('setJobTimeout')(86400); // 24 hours

    // Mint tokens to users
    await inferenceToken.getFunction('mint')(user1.address, ethers.parseEther('1000'));
    await inferenceToken.getFunction('mint')(user2.address, ethers.parseEther('1000'));
  });

  describe('Deployment', () => {
    it('should deploy with correct initial state', async () => {
      expect(await computeMarketplace.getFunction('getTotalJobCount')()).to.equal(0);
      
      const supportedTokens = await computeMarketplace.getFunction('getSupportedTokens')();
      expect(supportedTokens).to.include(await inferenceToken.getAddress());
    });

    it('should set correct admin role', async () => {
      const adminRole = await computeMarketplace.getFunction('ADMIN_ROLE')();
      expect(await computeMarketplace.getFunction('hasRole')(adminRole, deployer.address)).to.be.true;
    });
  });

  describe('Job Management', () => {
    it('should post a job with ETH payment', async () => {
      const bounty = ethers.parseEther('2.0');
      const timeout = 86400; // 24 hours
      
      const tx = await computeMarketplace.connect(user1).getFunction('postJob')(
        'ipfs://model-spec-uri',
        bounty,
        ethers.ZeroAddress, // ETH payment
        timeout,
        { value: bounty }
      );
      
      await tx.wait();
      
      expect(await computeMarketplace.getFunction('getTotalJobCount')()).to.equal(1);
      
      const job = await computeMarketplace.getFunction('getJob')(1);
      expect(job.requester).to.equal(user1.address);
      expect(job.bounty).to.equal(bounty);
      expect(job.paymentToken).to.equal(ethers.ZeroAddress);
      expect(job.specURI).to.equal('ipfs://model-spec-uri');
      expect(job.status).to.equal(0); // Posted
    });

    it('should post a job with ERC20 payment', async () => {
      const bounty = ethers.parseEther('5.0');
      const timeout = 86400;
      
      // Approve token transfer
      await inferenceToken.connect(user1).getFunction('approve')(await computeMarketplace.getAddress(), bounty);
      
      const tx = await computeMarketplace.connect(user1).getFunction('postJob')(
        'ipfs://model-spec-uri',
        bounty,
        await inferenceToken.getAddress(),
        timeout
      );
      
      await tx.wait();
      
      const job = await computeMarketplace.getFunction('getJob')(1);
      expect(job.bounty).to.equal(bounty);
      expect(job.paymentToken).to.equal(await inferenceToken.getAddress());
    });

    it('should fail to post job with insufficient bounty', async () => {
      const bounty = ethers.parseEther('0.5'); // Less than minimum
      
      await expect(
        computeMarketplace.connect(user1).getFunction('postJob')(
          'ipfs://model-spec-uri',
          bounty,
          ethers.ZeroAddress,
          86400,
          { value: bounty }
        )
      ).to.be.revertedWith('Bounty too low');
    });

    it('should claim a job', async () => {
      // Post a job first
      const bounty = ethers.parseEther('2.0');
      await computeMarketplace.connect(user1).getFunction('postJob')(
        'ipfs://model-spec-uri',
        bounty,
        ethers.ZeroAddress,
        86400,
        { value: bounty }
      );
      
      // Claim the job
      const tx = await computeMarketplace.connect(user2).getFunction('claimJob')(1);
      await tx.wait();
      
      const job = await computeMarketplace.getFunction('getJob')(1);
      expect(job.provider).to.equal(user2.address);
      expect(job.status).to.equal(1); // Claimed
    });

    it('should complete a job', async () => {
      // Post and claim a job
      const bounty = ethers.parseEther('2.0');
      await computeMarketplace.connect(user1).getFunction('postJob')(
        'ipfs://model-spec-uri',
        bounty,
        ethers.ZeroAddress,
        86400,
        { value: bounty }
      );
      
      await computeMarketplace.connect(user2).getFunction('claimJob')(1);
      
      // Complete the job
      const resultHash = ethers.keccak256(ethers.toUtf8Bytes('result-data'));
      const tx = await computeMarketplace.connect(user2).getFunction('completeJob')(
        1,
        'ipfs://result-uri',
        resultHash,
        '0x' // empty proof for now
      );
      await tx.wait();
      
      const job = await computeMarketplace.getFunction('getJob')(1);
      expect(job.status).to.equal(2); // Completed
      expect(job.resultHash).to.equal(resultHash);
      
      // Check provider stats
      const providerStats = await computeMarketplace.getFunction('getProviderStats')(user2.address);
      expect(providerStats.totalJobs).to.equal(1);
      expect(providerStats.completedJobs).to.equal(1);
      expect(providerStats.totalEarnings).to.equal(bounty);
    });

    it('should dispute a job', async () => {
      // Post, claim, and complete a job
      const bounty = ethers.parseEther('2.0');
      await computeMarketplace.connect(user1).getFunction('postJob')(
        'ipfs://model-spec-uri',
        bounty,
        ethers.ZeroAddress,
        86400,
        { value: bounty }
      );
      
      await computeMarketplace.connect(user2).getFunction('claimJob')(1);
      
      const resultHash = ethers.keccak256(ethers.toUtf8Bytes('result-data'));
      await computeMarketplace.connect(user2).getFunction('completeJob')(
        1,
        'ipfs://result-uri',
        resultHash,
        '0x'
      );
      
      // Dispute the job
      const tx = await computeMarketplace.connect(user1).getFunction('disputeJob')(
        1,
        'Result is incorrect'
      );
      await tx.wait();
      
      const job = await computeMarketplace.getFunction('getJob')(1);
      expect(job.status).to.equal(4); // Disputed
    });

    it('should resolve dispute in favor of provider', async () => {
      // Setup and dispute a job
      const bounty = ethers.parseEther('2.0');
      await computeMarketplace.connect(user1).getFunction('postJob')(
        'ipfs://model-spec-uri',
        bounty,
        ethers.ZeroAddress,
        86400,
        { value: bounty }
      );
      
      await computeMarketplace.connect(user2).getFunction('claimJob')(1);
      
      const resultHash = ethers.keccak256(ethers.toUtf8Bytes('result-data'));
      await computeMarketplace.connect(user2).getFunction('completeJob')(
        1,
        'ipfs://result-uri',
        resultHash,
        '0x'
      );
      
      await computeMarketplace.connect(user1).getFunction('disputeJob')(1, 'Dispute reason');
      
      // Resolve in favor of provider
      const tx = await computeMarketplace.connect(deployer).getFunction('resolveDispute')(1, true);
      await tx.wait();
      
      const job = await computeMarketplace.getFunction('getJob')(1);
      expect(job.status).to.equal(2); // Completed
    });

    it('should resolve dispute in favor of requester', async () => {
      // Setup and dispute a job
      const bounty = ethers.parseEther('2.0');
      await computeMarketplace.connect(user1).getFunction('postJob')(
        'ipfs://model-spec-uri',
        bounty,
        ethers.ZeroAddress,
        86400,
        { value: bounty }
      );
      
      await computeMarketplace.connect(user2).getFunction('claimJob')(1);
      
      const resultHash = ethers.keccak256(ethers.toUtf8Bytes('result-data'));
      await computeMarketplace.connect(user2).getFunction('completeJob')(
        1,
        'ipfs://result-uri',
        resultHash,
        '0x'
      );
      
      await computeMarketplace.connect(user1).getFunction('disputeJob')(1, 'Dispute reason');
      
      // Resolve in favor of requester
      const tx = await computeMarketplace.connect(deployer).getFunction('resolveDispute')(1, false);
      await tx.wait();
      
      const job = await computeMarketplace.getFunction('getJob')(1);
      expect(job.status).to.equal(3); // Failed
    });
  });

  describe('Batch Processing', () => {
    it('should submit batch inputs', async () => {
      // Post a job first
      const bounty = ethers.parseEther('2.0');
      await computeMarketplace.connect(user1).getFunction('postJob')(
        'ipfs://model-spec-uri',
        bounty,
        ethers.ZeroAddress,
        86400,
        { value: bounty }
      );
      
      // Submit batch inputs
      const inputs = [
        ethers.keccak256(ethers.toUtf8Bytes('input1')),
        ethers.keccak256(ethers.toUtf8Bytes('input2'))
      ];
      
      const tx = await computeMarketplace.connect(user1).getFunction('batchSubmitInputs')(
        1,
        inputs,
        ethers.parseEther('0.1')
      );
      await tx.wait();
      
      const batchInputs = await computeMarketplace.getFunction('getBatchInputs')(1);
      expect(batchInputs.length).to.equal(1);
      expect(batchInputs[0].inputs.length).to.equal(2);
      expect(batchInputs[0].processed).to.be.false;
    });

    it('should process batch', async () => {
      // Post job and claim it
      const bounty = ethers.parseEther('2.0');
      await computeMarketplace.connect(user1).getFunction('postJob')(
        'ipfs://model-spec-uri',
        bounty,
        ethers.ZeroAddress,
        86400,
        { value: bounty }
      );
      
      await computeMarketplace.connect(user2).getFunction('claimJob')(1);
      
      // Submit batch inputs
      const inputs = [ethers.keccak256(ethers.toUtf8Bytes('input1'))];
      await computeMarketplace.connect(user1).getFunction('batchSubmitInputs')(
        1,
        inputs,
        ethers.parseEther('0.1')
      );
      
      // Process batch
      const tx = await computeMarketplace.connect(user2).getFunction('processBatch')(1, 0);
      await tx.wait();
      
      const batchInputs = await computeMarketplace.getFunction('getBatchInputs')(1);
      expect(batchInputs[0].processed).to.be.true;
    });
  });

  describe('Payment Management', () => {
    it('should handle ERC20 payments', async () => {
      const amount = ethers.parseEther('10.0');
      
      // Approve and deposit
      await inferenceToken.connect(user1).getFunction('approve')(await computeMarketplace.getAddress(), amount);
      await computeMarketplace.connect(user1).getFunction('payWithERC20')(await inferenceToken.getAddress(), amount);
      
      const balance = await computeMarketplace.connect(user1).getFunction('getPaymentBalance')(await inferenceToken.getAddress());
      expect(balance).to.equal(amount);
    });

    it('should withdraw bounty for expired job', async () => {
      const bounty = ethers.parseEther('2.0');
      
      // Post a job with very short timeout
      await computeMarketplace.connect(user1).getFunction('postJob')(
        'ipfs://model-spec-uri',
        bounty,
        ethers.ZeroAddress,
        1, // 1 second timeout
        { value: bounty }
      );
      
      // Wait for timeout
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Check if job is timed out
      const isTimedOut = await computeMarketplace.getFunction('isJobTimeout')(1);
      expect(isTimedOut).to.be.true;
      
      // Withdraw bounty
      const balanceBefore = await provider.getBalance(user1.address);
      const tx = await computeMarketplace.connect(user1).getFunction('withdrawBounty')(1);
      const receipt = await tx.wait();
      
      const balanceAfter = await provider.getBalance(user1.address);
      const gasUsed = receipt.gasUsed * receipt.gasPrice;
      
      expect(balanceAfter).to.be.closeTo(balanceBefore + bounty - gasUsed, ethers.parseEther('0.01'));
    });
  });

  describe('Admin Functions', () => {
    it('should set supported tokens', async () => {
      const newTokenAddress = ethers.Wallet.createRandom().address;
      
      await computeMarketplace.connect(deployer).getFunction('setSupportedTokens')([newTokenAddress]);
      
      const supportedTokens = await computeMarketplace.getFunction('getSupportedTokens')();
      expect(supportedTokens).to.include(newTokenAddress);
      expect(await computeMarketplace.getFunction('isTokenSupported')(newTokenAddress)).to.be.true;
    });

    it('should set minimum bounty', async () => {
      const newMinBounty = ethers.parseEther('5.0');
      
      await computeMarketplace.connect(deployer).getFunction('setMinBounty')(newMinBounty);
      
      // Try to post job with old minimum - should fail
      await expect(
        computeMarketplace.connect(user1).getFunction('postJob')(
          'ipfs://model-spec-uri',
          ethers.parseEther('1.0'),
          ethers.ZeroAddress,
          86400,
          { value: ethers.parseEther('1.0') }
        )
      ).to.be.revertedWith('Bounty too low');
    });

    it('should pause and unpause contract', async () => {
      // Pause contract
      await computeMarketplace.connect(deployer).getFunction('pause')();
      
      // Try to post job - should fail
      await expect(
        computeMarketplace.connect(user1).getFunction('postJob')(
          'ipfs://model-spec-uri',
          ethers.parseEther('2.0'),
          ethers.ZeroAddress,
          86400,
          { value: ethers.parseEther('2.0') }
        )
      ).to.be.revertedWith('Pausable: paused');
      
      // Unpause contract
      await computeMarketplace.connect(deployer).getFunction('unpause')();
      
      // Now posting should work
      const tx = await computeMarketplace.connect(user1).getFunction('postJob')(
        'ipfs://model-spec-uri',
        ethers.parseEther('2.0'),
        ethers.ZeroAddress,
        86400,
        { value: ethers.parseEther('2.0') }
      );
      await tx.wait();
      
      expect(await computeMarketplace.getFunction('getTotalJobCount')()).to.equal(1);
    });
  });

  describe('View Functions', () => {
    beforeEach(async () => {
      // Setup multiple jobs for testing
      const bounty = ethers.parseEther('2.0');
      
      // Post 3 jobs
      for (let i = 0; i < 3; i++) {
        await computeMarketplace.connect(user1).getFunction('postJob')(
          `ipfs://model-spec-uri-${i}`,
          bounty,
          ethers.ZeroAddress,
          86400,
          { value: bounty }
        );
      }
      
      // Claim 2 jobs
      await computeMarketplace.connect(user2).getFunction('claimJob')(1);
      await computeMarketplace.connect(user2).getFunction('claimJob')(2);
    });

    it('should get active jobs', async () => {
      const activeJobs = await computeMarketplace.getFunction('getActiveJobs')();
      expect(activeJobs.length).to.equal(3); // All jobs are active (posted or claimed)
    });

    it('should get provider jobs', async () => {
      const providerJobs = await computeMarketplace.getFunction('getProviderJobs')(user2.address);
      expect(providerJobs.length).to.equal(2);
      expect(providerJobs[0]).to.equal(1n);
      expect(providerJobs[1]).to.equal(2n);
    });

    it('should get requester jobs', async () => {
      const requesterJobs = await computeMarketplace.getFunction('getRequesterJobs')(user1.address);
      expect(requesterJobs.length).to.equal(3);
    });

    it('should get provider stats', async () => {
      const stats = await computeMarketplace.getFunction('getProviderStats')(user2.address);
      expect(stats.totalJobs).to.equal(2);
      expect(stats.completedJobs).to.equal(0); // None completed yet
    });
  });
}); 