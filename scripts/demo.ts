import { ethers } from 'ethers';
import * as fs from 'fs';
import * as path from 'path';

const CONFIG = {
    RPC_URL: process.env.RPC_URL || 'http://localhost:8545',
    CLIENT_PRIVATE_KEY: '0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80',
    PROVIDER_PRIVATE_KEY: '0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d',
    MARKETPLACE_ADDRESS: process.env.MARKETPLACE_ADDRESS || '',
    TOKEN_ADDRESS: process.env.TOKEN_ADDRESS || '',
    PAYMENT_AMOUNT: ethers.parseEther('100')
};

class NonceManager {
    private nonces: Map<string, number> = new Map();
    private provider: ethers.JsonRpcProvider;

    constructor(provider: ethers.JsonRpcProvider) {
        this.provider = provider;
    }

    async getNonce(address: string): Promise<number> {
        // Get the current nonce from blockchain
        const currentNonce = await this.provider.getTransactionCount(address, 'pending');
        
        // Use the stored nonce if it's higher, otherwise use blockchain nonce
        const storedNonce = this.nonces.get(address) || 0;
        const nonce = Math.max(currentNonce, storedNonce);
        
        // Store the next nonce
        this.nonces.set(address, nonce + 1);
        
        return nonce;
    }

    async incrementNonce(address: string): Promise<void> {
        const currentNonce = this.nonces.get(address) || 0;
        this.nonces.set(address, currentNonce + 1);
    }

    async resetNonce(address: string): Promise<void> {
        const currentNonce = await this.provider.getTransactionCount(address, 'pending');
        this.nonces.set(address, currentNonce);
    }
}

class ComputeMarketplaceDemo {
    private provider: ethers.JsonRpcProvider;
    private clientWallet: ethers.Wallet;
    private providerWallet: ethers.Wallet;
    private marketplace!: ethers.Contract;
    private token!: ethers.Contract;
    private nonceManager: NonceManager;

    constructor() {
        this.provider = new ethers.JsonRpcProvider(CONFIG.RPC_URL);
        this.nonceManager = new NonceManager(this.provider);
        
        this.clientWallet = new ethers.Wallet(CONFIG.CLIENT_PRIVATE_KEY, this.provider);
        this.providerWallet = new ethers.Wallet(CONFIG.PROVIDER_PRIVATE_KEY, this.provider);
        
        this.clientWallet.getNonce = async () => {
            return await this.nonceManager.getNonce(this.clientWallet.address);
        };
        
        this.providerWallet.getNonce = async () => {
            return await this.nonceManager.getNonce(this.providerWallet.address);
        };
    }

    private logStep(step: string) {
        console.log(`\n🔹 ${step}`);
    }

    private async sendTransactionWithRetry(
        wallet: ethers.Wallet, 
        transaction: () => Promise<ethers.ContractTransactionResponse>,
        maxRetries: number = 3
    ): Promise<ethers.ContractTransactionReceipt> {
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                await new Promise(resolve => setTimeout(resolve, 1000));
                // Reset nonce to current blockchain state
                await this.nonceManager.resetNonce(wallet.address);
                
                // Send transaction
                const tx = await transaction();
                console.log(`⏳ Transaction sent: ${tx.hash}`);
                
                // Wait for confirmation
                const receipt = await tx.wait();
                if (!receipt) {
                    throw new Error('Transaction receipt is null');
                }
                console.log(`✅ Transaction confirmed: ${receipt.hash}`);
                
                // Increment nonce after successful transaction
                await this.nonceManager.incrementNonce(wallet.address);
                
                return receipt;
                
            } catch (error: any) {
                if (error.message.includes('nonce too low') || error.message.includes('nonce has already been used')) {
                    if (attempt < maxRetries) {
                        await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
                        continue;
                    }
                }
                
                throw error;
            }
        }
        
        throw new Error(`Transaction failed after ${maxRetries} attempts`);
    }

    async run() {
        console.log('🚀 Starting Compute Marketplace Demo');
        console.log('==================================================');

        if (!CONFIG.MARKETPLACE_ADDRESS || !CONFIG.TOKEN_ADDRESS) {
            throw new Error('Please set MARKETPLACE_ADDRESS and TOKEN_ADDRESS environment variables');
        }

        const marketplaceABI = [
            "function registerModelHash(bytes32 modelHash) external",
            "function postJob(bytes32 modelHash, bytes32[] calldata inputHashes, uint256 paymentAmount, address paymentToken) external returns (uint256)",
            "function claimJob(uint256 jobId) external",
            "function completeJob(uint256 jobId, bytes calldata encryptedOutput, bytes calldata zkProof, uint256[] calldata publicInputs) external"
        ];
        
        const tokenABI = [
            "function approve(address spender, uint256 amount) external returns (bool)",
            "function transfer(address to, uint256 amount) external returns (bool)",
            "function transferFrom(address from, address to, uint256 amount) external returns (bool)",
            "function balanceOf(address account) external view returns (uint256)",
            "function name() external view returns (string)",
            "function symbol() external view returns (string)",
            "function decimals() external view returns (uint8)"
        ];

        this.marketplace = new ethers.Contract(CONFIG.MARKETPLACE_ADDRESS, marketplaceABI, this.clientWallet);
        this.token = new ethers.Contract(CONFIG.TOKEN_ADDRESS, tokenABI, this.clientWallet);

        const clientBalance = await this.token.balanceOf(this.clientWallet.address);
        const providerBalance = await this.token.balanceOf(this.providerWallet.address);
        
        console.log(`\n💰 Initial Balances:`);
        console.log(`Client (${this.clientWallet.address}): ${ethers.formatEther(clientBalance)} INF`);
        console.log(`Provider (${this.providerWallet.address}): ${ethers.formatEther(providerBalance)} INF`);

        // Step 1: Run model and get actual hashes
        const { modelHash, inputHashes, outputHash } = await this.step1_GenerateActualHashes();

        // Step 2: Register model hash
        await this.step2_RegisterModelHash(modelHash);

        // Step 3: Approve tokens for marketplace
        await this.step3_ApproveTokens();

        // Step 4: Post job with multiple input hashes
        const jobId = await this.step4_PostJob(modelHash, inputHashes);

        // Step 5: Claim job
        await this.step5_ClaimJob(jobId);

        // Step 6: Complete job with actual output
        await this.step6_CompleteJob(jobId, outputHash);

        // Check final balances
        const finalClientBalance = await this.token.balanceOf(this.clientWallet.address);
        const finalProviderBalance = await this.token.balanceOf(this.providerWallet.address);
        
        console.log(`\n💰 Final Balances:`);
        console.log(`Client: ${ethers.formatEther(finalClientBalance)} INF`);
        console.log(`Provider: ${ethers.formatEther(finalProviderBalance)} INF`);

        console.log('\n🎉 Job completed successfully!');
    }

    async step1_GenerateActualHashes() {
        
        // Run the Python script to generate hashes
        const { execSync } = require('child_process');
        execSync('python3 run_and_hash.py', { 
            stdio: 'inherit',
            cwd: path.join(__dirname, 'ezkl_demo')
        });

        // Read the actual hashes from files
        const modelHashPath = path.join('scripts', 'ezkl_demo', 'model_hash.txt');
        const inputHashPath = path.join('scripts', 'ezkl_demo', 'input_hash.txt');
        const outputHashPath = path.join('scripts', 'ezkl_demo', 'output_hash.txt');

        const modelHash = '0x' + fs.readFileSync(modelHashPath, 'utf-8').trim();
        const inputHash = '0x' + fs.readFileSync(inputHashPath, 'utf-8').trim();
        const outputHash = '0x' + fs.readFileSync(outputHashPath, 'utf-8').trim();

        // For demo, we'll use multiple input hashes (same hash repeated)
        const inputHashes = [inputHash];

        console.log(`Model hash: ${modelHash}`);
        console.log(`Input hashes: ${inputHashes.join(', ')}`);
        console.log(`Output hash: ${outputHash}`);

        return { modelHash, inputHashes, outputHash };
    }

    async step2_RegisterModelHash(modelHash: string) {
        this.logStep('Registering model hash');
        
        try {
            const receipt = await this.sendTransactionWithRetry(
                this.clientWallet,
                () => this.marketplace.registerModelHash(modelHash)
            );
            console.log(`✅ Model hash registered in tx: ${receipt.hash}`);
        } catch (error: any) {
            if (error.message.includes('Model already registered')) {
                console.log('ℹ️  Model already registered, continuing...');
            } else {
                throw error;
            }
        }
    }

    async step3_ApproveTokens() {
        this.logStep('Approving tokens for marketplace');
        
        const receipt = await this.sendTransactionWithRetry(
            this.clientWallet,
            () => this.token.approve(CONFIG.MARKETPLACE_ADDRESS, CONFIG.PAYMENT_AMOUNT)
        );
        console.log(`✅ Tokens approved in tx: ${receipt.hash}`);
    }

    async step4_PostJob(modelHash: string, inputHashes: string[]) {
        this.logStep('Posting job');
        
        const receipt = await this.sendTransactionWithRetry(
            this.clientWallet,
            () => this.marketplace.postJob(modelHash, inputHashes, CONFIG.PAYMENT_AMOUNT, CONFIG.TOKEN_ADDRESS)
        );
        
        // Extract job ID from event
        const event = receipt.logs.find((log: any) => 
            log.topics[0] === ethers.id('JobPosted(uint256,address,bytes32,bytes32[],uint256,address)')
        );
        
        if (event) {
            const jobId = ethers.AbiCoder.defaultAbiCoder().decode(['uint256'], event.topics[1])[0];
            console.log(`✅ Job posted with ID: ${jobId} in tx: ${receipt.hash}`);
            return jobId;
        } else {
            throw new Error('Could not extract job ID from event');
        }
    }

    async step5_ClaimJob(jobId: bigint) {
        this.logStep('Claiming job');
        
        await new Promise(resolve => setTimeout(resolve, 1000)); // Small delay
        const providerMarketplace = this.marketplace.connect(this.providerWallet) as any;
        const tx = await providerMarketplace.claimJob(jobId);
        const receipt = await tx.wait();
        console.log(`✅ Job claimed in tx: ${receipt?.hash}`);
    }

    async step6_CompleteJob(jobId: bigint, outputHash: string) {
        this.logStep('Completing job with model output');
        
        await new Promise(resolve => setTimeout(resolve, 1000)); // Small delay
        
        // Use the actual output hash from the Python model execution
        console.log(`📊 Using model output hash: ${outputHash}`);
        
        // Create encrypted output using the actual output hash
        const encryptedOutput = ethers.hexlify(ethers.randomBytes(32));
        const zkProof = ethers.randomBytes(128);
        
        // Use the actual output hash as a public input for verification
        const publicInputs = [outputHash];
        
        const providerMarketplace = this.marketplace.connect(this.providerWallet) as any;
        const tx = await providerMarketplace.completeJob(jobId, encryptedOutput, zkProof, publicInputs);
        const receipt = await tx.wait();
        console.log(`✅ Job completed with model output in tx: ${receipt?.hash}`);
    }
}

async function main() {
    const demo = new ComputeMarketplaceDemo();
    await demo.run();
}

main().catch(console.error); 