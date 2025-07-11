import { ethers } from 'ethers';
import * as fs from 'fs';
import * as path from 'path';
import { networks, deploymentConfig, getPrivateKey } from './config';

interface DeploymentResult {
  network: string;
  chainId: number;
  deployer: string;
  contracts: {
    InferenceERC20: {
      address: string;
      transactionHash: string;
    };
    ComputeMarketplace: {
      address: string;
      transactionHash: string;
    };
  };
  timestamp: string;
}

async function loadContractArtifact(contractName: string): Promise<any> {
  const artifactPath = path.join(process.cwd(), 'out', contractName, `${contractName}.json`);
  if (!fs.existsSync(artifactPath)) {
    throw new Error(`Contract artifact not found: ${artifactPath}`);
  }
  return JSON.parse(fs.readFileSync(artifactPath, 'utf8'));
}

async function deploy(): Promise<void> {
  // Parse command line arguments
  const args = process.argv.slice(2);
  const networkArg = args.find((arg: string) => arg.startsWith('--network='))?.split('=')[1];
  const networkName = networkArg || 'localhost';

  if (!networks[networkName]) {
    throw new Error(`Network ${networkName} not found in config`);
  }

  const network = networks[networkName];
  console.log(`🚀 Deploying to ${network.name} (${network.chainId})`);

  // Setup provider and signer
  const provider = new ethers.JsonRpcProvider(network.rpcUrl);
  const signer = new ethers.Wallet(getPrivateKey(), provider);

  console.log(`📝 Deployer address: ${signer.address}`);
  
  // Check balance
  const balance = await provider.getBalance(signer.address);
  console.log(`💰 Deployer balance: ${ethers.formatEther(balance)} ${network.nativeCurrency.symbol}`);

  if (balance === BigInt(0)) {
    throw new Error('Deployer has no balance');
  }

  // Load contract artifacts
  console.log('📦 Loading contract artifacts...');
  const tokenArtifact = await loadContractArtifact('InferenceERC20.sol/InferenceERC20');
  const marketplaceArtifact = await loadContractArtifact('ComputeMarketplace.sol/ComputeMarketplace');

  // Deploy InferenceERC20
  console.log('🏗️  Deploying InferenceERC20...');
  const tokenFactory = new ethers.ContractFactory(
    tokenArtifact.abi,
    tokenArtifact.bytecode,
    signer
  );

  const token = await tokenFactory.deploy(
    deploymentConfig.token.name,
    deploymentConfig.token.symbol,
    deploymentConfig.token.decimals,
    signer.address
  );

  await token.waitForDeployment();
  const tokenAddress = await token.getAddress();
  console.log(`✅ InferenceERC20 deployed to: ${tokenAddress}`);

  // Deploy ComputeMarketplace
  console.log('🏗️  Deploying ComputeMarketplace...');
  const marketplaceFactory = new ethers.ContractFactory(
    marketplaceArtifact.abi,
    marketplaceArtifact.bytecode,
    signer
  );

  const marketplace = await marketplaceFactory.deploy(
    signer.address, // admin
    ethers.ZeroAddress // zkVerifier (can be set later)
  );

  await marketplace.waitForDeployment();
  const marketplaceAddress = await marketplace.getAddress();
  console.log(`✅ ComputeMarketplace deployed to: ${marketplaceAddress}`);

  // Initial configuration
  console.log('⚙️  Setting up initial configuration...');
  
  // Set supported tokens
  const setSupportedTokensTx = await marketplace.getFunction('setSupportedTokens')([tokenAddress]);
  await setSupportedTokensTx.wait();
  console.log('✅ Set supported tokens');

  // Set minimum bounty - parse the string as ether
  const minBountyWei = ethers.parseEther(deploymentConfig.marketplace.minBounty);
  const setMinBountyTx = await marketplace.getFunction('setMinBounty')(minBountyWei);
  await setMinBountyTx.wait();
  console.log('✅ Set minimum bounty');

  // Set default timeout
  const setTimeoutTx = await marketplace.getFunction('setJobTimeout')(deploymentConfig.marketplace.defaultTimeout);
  await setTimeoutTx.wait();
  console.log('✅ Set default timeout');

  // Prepare deployment result
  const deploymentResult: DeploymentResult = {
    network: networkName,
    chainId: network.chainId,
    deployer: signer.address,
    contracts: {
      InferenceERC20: {
        address: tokenAddress,
        transactionHash: token.deploymentTransaction()?.hash || '',
      },
      ComputeMarketplace: {
        address: marketplaceAddress,
        transactionHash: marketplace.deploymentTransaction()?.hash || '',
      },
    },
    timestamp: new Date().toISOString(),
  };

  // Save deployment result
  const deploymentDir = path.join(process.cwd(), 'deployments');
  if (!fs.existsSync(deploymentDir)) {
    fs.mkdirSync(deploymentDir, { recursive: true });
  }

  const deploymentFile = path.join(deploymentDir, `${networkName}-${Date.now()}.json`);
  fs.writeFileSync(deploymentFile, JSON.stringify(deploymentResult, null, 2));

  console.log('\n🎉 Deployment completed successfully!');
  console.log('📄 Deployment summary:');
  console.log(`   Network: ${network.name}`);
  console.log(`   InferenceERC20: ${tokenAddress}`);
  console.log(`   ComputeMarketplace: ${marketplaceAddress}`);
  console.log(`   Deployment file: ${deploymentFile}`);

  if (network.blockExplorer) {
    console.log('\n🔍 View on block explorer:');
    console.log(`   InferenceERC20: ${network.blockExplorer}/address/${tokenAddress}`);
    console.log(`   ComputeMarketplace: ${network.blockExplorer}/address/${marketplaceAddress}`);
  }
}

if (require.main === module) {
  deploy()
    .then(() => process.exit(0))
    .catch((error) => {
      console.error('❌ Deployment failed:', error);
      process.exit(1);
    });
}

export { deploy }; 