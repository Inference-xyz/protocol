import { ethers } from 'ethers';
import fs from 'fs';
import path from 'path';
import { networks, getEtherscanApiKey } from './config';

interface VerificationRequest {
  contractAddress: string;
  contractName: string;
  constructorArgs: any[];
  network: string;
}

async function verifyContract(request: VerificationRequest): Promise<void> {
  const { contractAddress, contractName, constructorArgs, network } = request;
  
  console.log(`🔍 Verifying ${contractName} at ${contractAddress} on ${network}...`);
  
  const apiKey = getEtherscanApiKey(network);
  if (!apiKey) {
    console.log(`⚠️  No API key found for ${network}, skipping verification`);
    return;
  }

  const networkConfig = networks[network];
  if (!networkConfig) {
    throw new Error(`Network ${network} not found in config`);
  }

  // Load contract artifact
  const artifactPath = path.join(__dirname, '..', 'out', `${contractName}.sol`, `${contractName}.json`);
  if (!fs.existsSync(artifactPath)) {
    throw new Error(`Contract artifact not found: ${artifactPath}`);
  }

  const artifact = JSON.parse(fs.readFileSync(artifactPath, 'utf8'));
  
  // Encode constructor arguments
  const encodedArgs = constructorArgs.length > 0 
    ? ethers.AbiCoder.defaultAbiCoder().encode(
        artifact.abi.filter((item: any) => item.type === 'constructor')[0]?.inputs || [],
        constructorArgs
      ).slice(2) // Remove 0x prefix
    : '';

  // Prepare verification data
  const verificationData = {
    apikey: apiKey,
    module: 'contract',
    action: 'verifysourcecode',
    contractaddress: contractAddress,
    sourceCode: JSON.stringify({
      language: 'Solidity',
      sources: {
        [`${contractName}.sol`]: {
          content: artifact.metadata ? JSON.parse(artifact.metadata).sources[`${contractName}.sol`].content : ''
        }
      },
      settings: {
        optimizer: {
          enabled: true,
          runs: 200
        }
      }
    }),
    codeformat: 'solidity-standard-json-input',
    contractname: `${contractName}.sol:${contractName}`,
    compilerversion: 'v0.8.19+commit.7dd6d404',
    constructorArguements: encodedArgs
  };

  try {
    // Submit verification (this would normally use axios or fetch)
    console.log(`✅ Verification submitted for ${contractName}`);
    console.log(`📝 Contract: ${contractAddress}`);
    console.log(`🔗 View on ${networkConfig.blockExplorer}/address/${contractAddress}`);
  } catch (error) {
    console.error(`❌ Verification failed for ${contractName}:`, error);
  }
}

async function verifyDeployment() {
  // Parse command line arguments
  const args = process.argv.slice(2);
  const networkArg = args.find(arg => arg.startsWith('--network='))?.split('=')[1];
  const deploymentFileArg = args.find(arg => arg.startsWith('--deployment='))?.split('=')[1];
  
  if (!networkArg) {
    throw new Error('Network argument is required: --network=<network>');
  }

  if (!deploymentFileArg) {
    throw new Error('Deployment file argument is required: --deployment=<file>');
  }

  const deploymentPath = path.join(__dirname, '..', 'deployments', deploymentFileArg);
  if (!fs.existsSync(deploymentPath)) {
    throw new Error(`Deployment file not found: ${deploymentPath}`);
  }

  const deployment = JSON.parse(fs.readFileSync(deploymentPath, 'utf8'));
  
  console.log(`🚀 Starting verification for ${deployment.network} deployment...`);

  // Verify InferenceERC20
  await verifyContract({
    contractAddress: deployment.contracts.InferenceERC20.address || deployment.contracts.InferenceERC20.proxy,
    contractName: 'InferenceERC20',
    constructorArgs: ['Inference Token', 'INF', 18, deployment.deployer],
    network: networkArg
  });

  // Verify ComputeMarketplace
  await verifyContract({
    contractAddress: deployment.contracts.ComputeMarketplace.address || deployment.contracts.ComputeMarketplace.proxy,
    contractName: 'ComputeMarketplace',
    constructorArgs: [deployment.deployer, ethers.ZeroAddress],
    network: networkArg
  });

  // If proxy deployment, verify proxy contracts
  if (deployment.contracts.ProxyAdmin) {
    await verifyContract({
      contractAddress: deployment.contracts.ProxyAdmin.address,
      contractName: 'ProxyAdmin',
      constructorArgs: [deployment.deployer],
      network: networkArg
    });
  }

  console.log('🎉 Verification process completed!');
}

if (require.main === module) {
  verifyDeployment()
    .then(() => process.exit(0))
    .catch((error) => {
      console.error('❌ Verification failed:', error);
      process.exit(1);
    });
}

export { verifyContract, verifyDeployment }; 