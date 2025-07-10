import { ethers } from 'ethers';
import fs from 'fs';
import path from 'path';
import { networks, deploymentConfig, getPrivateKey } from './config';

interface ProxyDeploymentResult {
  network: string;
  chainId: number;
  deployer: string;
  contracts: {
    ProxyAdmin: {
      address: string;
      transactionHash: string;
    };
    InferenceERC20: {
      implementation: string;
      proxy: string;
      implementationTxHash: string;
      proxyTxHash: string;
    };
    ComputeMarketplace: {
      implementation: string;
      proxy: string;
      implementationTxHash: string;
      proxyTxHash: string;
    };
  };
  timestamp: string;
}

async function loadContractArtifact(contractName: string): Promise<any> {
  const artifactPath = path.join(__dirname, '..', 'out', contractName, `${contractName}.json`);
  if (!fs.existsSync(artifactPath)) {
    throw new Error(`Contract artifact not found: ${artifactPath}`);
  }
  return JSON.parse(fs.readFileSync(artifactPath, 'utf8'));
}

async function deployWithProxy() {
  // Parse command line arguments
  const args = process.argv.slice(2);
  const networkArg = args.find(arg => arg.startsWith('--network='))?.split('=')[1];
  const networkName = networkArg || 'localhost';

  if (!networks[networkName]) {
    throw new Error(`Network ${networkName} not found in config`);
  }

  const network = networks[networkName];
  console.log(`🚀 Deploying with proxy pattern to ${network.name} (${network.chainId})`);

  // Setup provider and signer
  const provider = new ethers.JsonRpcProvider(network.rpcUrl);
  const signer = new ethers.Wallet(getPrivateKey(), provider);

  console.log(`📝 Deployer address: ${signer.address}`);
  
  // Check balance
  const balance = await provider.getBalance(signer.address);
  console.log(`💰 Deployer balance: ${ethers.formatEther(balance)} ${network.nativeCurrency.symbol}`);

  if (balance === 0n) {
    throw new Error('Deployer has no balance');
  }

  // Load contract artifacts
  console.log('📦 Loading contract artifacts...');
  const proxyAdminArtifact = await loadContractArtifact('ProxyAdmin.sol/ProxyAdmin');
  const transparentProxyArtifact = await loadContractArtifact('TransparentUpgradeableProxy.sol/TransparentUpgradeableProxy');
  const tokenArtifact = await loadContractArtifact('InferenceERC20.sol/InferenceERC20');
  const marketplaceArtifact = await loadContractArtifact('ComputeMarketplace.sol/ComputeMarketplace');

  // Deploy ProxyAdmin
  console.log('🏗️  Deploying ProxyAdmin...');
  const proxyAdminFactory = new ethers.ContractFactory(
    proxyAdminArtifact.abi,
    proxyAdminArtifact.bytecode,
    signer
  );

  const proxyAdmin = await proxyAdminFactory.deploy(signer.address);
  await proxyAdmin.waitForDeployment();
  const proxyAdminAddress = await proxyAdmin.getAddress();
  console.log(`✅ ProxyAdmin deployed to: ${proxyAdminAddress}`);

  // Deploy InferenceERC20 Implementation
  console.log('🏗️  Deploying InferenceERC20 implementation...');
  const tokenImplFactory = new ethers.ContractFactory(
    tokenArtifact.abi,
    tokenArtifact.bytecode,
    signer
  );

  const tokenImpl = await tokenImplFactory.deploy(
    deploymentConfig.token.name,
    deploymentConfig.token.symbol,
    deploymentConfig.token.decimals,
    signer.address
  );

  await tokenImpl.waitForDeployment();
  const tokenImplAddress = await tokenImpl.getAddress();
  console.log(`✅ InferenceERC20 implementation deployed to: ${tokenImplAddress}`);

  // Deploy InferenceERC20 Proxy
  console.log('🏗️  Deploying InferenceERC20 proxy...');
  const tokenProxyFactory = new ethers.ContractFactory(
    transparentProxyArtifact.abi,
    transparentProxyArtifact.bytecode,
    signer
  );

  const tokenProxy = await tokenProxyFactory.deploy(
    tokenImplAddress,
    proxyAdminAddress,
    '0x' // Empty initialization data
  );

  await tokenProxy.waitForDeployment();
  const tokenProxyAddress = await tokenProxy.getAddress();
  console.log(`✅ InferenceERC20 proxy deployed to: ${tokenProxyAddress}`);

  // Deploy ComputeMarketplace Implementation
  console.log('🏗️  Deploying ComputeMarketplace implementation...');
  const marketplaceImplFactory = new ethers.ContractFactory(
    marketplaceArtifact.abi,
    marketplaceArtifact.bytecode,
    signer
  );

  const marketplaceImpl = await marketplaceImplFactory.deploy(
    signer.address,
    ethers.ZeroAddress // zkVerifier
  );

  await marketplaceImpl.waitForDeployment();
  const marketplaceImplAddress = await marketplaceImpl.getAddress();
  console.log(`✅ ComputeMarketplace implementation deployed to: ${marketplaceImplAddress}`);

  // Deploy ComputeMarketplace Proxy
  console.log('🏗️  Deploying ComputeMarketplace proxy...');
  const marketplaceProxyFactory = new ethers.ContractFactory(
    transparentProxyArtifact.abi,
    transparentProxyArtifact.bytecode,
    signer
  );

  const marketplaceProxy = await marketplaceProxyFactory.deploy(
    marketplaceImplAddress,
    proxyAdminAddress,
    '0x' // Empty initialization data
  );

  await marketplaceProxy.waitForDeployment();
  const marketplaceProxyAddress = await marketplaceProxy.getAddress();
  console.log(`✅ ComputeMarketplace proxy deployed to: ${marketplaceProxyAddress}`);

  // Initial configuration through proxy
  console.log('⚙️  Setting up initial configuration...');
  
  // Create interface for marketplace proxy
  const marketplaceInterface = new ethers.Contract(
    marketplaceProxyAddress,
    marketplaceArtifact.abi,
    signer
  );

  // Set supported tokens
  const setSupportedTokensTx = await marketplaceInterface.setSupportedTokens([tokenProxyAddress]);
  await setSupportedTokensTx.wait();
  console.log('✅ Set supported tokens');

  // Set minimum bounty
  const setMinBountyTx = await marketplaceInterface.setMinBounty(deploymentConfig.marketplace.minBounty);
  await setMinBountyTx.wait();
  console.log('✅ Set minimum bounty');

  // Set default timeout
  const setTimeoutTx = await marketplaceInterface.setJobTimeout(deploymentConfig.marketplace.defaultTimeout);
  await setTimeoutTx.wait();
  console.log('✅ Set default timeout');

  // Prepare deployment result
  const deploymentResult: ProxyDeploymentResult = {
    network: networkName,
    chainId: network.chainId,
    deployer: signer.address,
    contracts: {
      ProxyAdmin: {
        address: proxyAdminAddress,
        transactionHash: proxyAdmin.deploymentTransaction()?.hash || '',
      },
      InferenceERC20: {
        implementation: tokenImplAddress,
        proxy: tokenProxyAddress,
        implementationTxHash: tokenImpl.deploymentTransaction()?.hash || '',
        proxyTxHash: tokenProxy.deploymentTransaction()?.hash || '',
      },
      ComputeMarketplace: {
        implementation: marketplaceImplAddress,
        proxy: marketplaceProxyAddress,
        implementationTxHash: marketplaceImpl.deploymentTransaction()?.hash || '',
        proxyTxHash: marketplaceProxy.deploymentTransaction()?.hash || '',
      },
    },
    timestamp: new Date().toISOString(),
  };

  // Save deployment result
  const deploymentDir = path.join(__dirname, '..', 'deployments');
  if (!fs.existsSync(deploymentDir)) {
    fs.mkdirSync(deploymentDir, { recursive: true });
  }

  const deploymentFile = path.join(deploymentDir, `${networkName}-proxy-${Date.now()}.json`);
  fs.writeFileSync(deploymentFile, JSON.stringify(deploymentResult, null, 2));

  console.log('\n🎉 Proxy deployment completed successfully!');
  console.log('📄 Deployment summary:');
  console.log(`   Network: ${network.name}`);
  console.log(`   ProxyAdmin: ${proxyAdminAddress}`);
  console.log(`   InferenceERC20 Implementation: ${tokenImplAddress}`);
  console.log(`   InferenceERC20 Proxy: ${tokenProxyAddress}`);
  console.log(`   ComputeMarketplace Implementation: ${marketplaceImplAddress}`);
  console.log(`   ComputeMarketplace Proxy: ${marketplaceProxyAddress}`);
  console.log(`   Deployment file: ${deploymentFile}`);

  if (network.blockExplorer) {
    console.log('\n🔍 View on block explorer:');
    console.log(`   ProxyAdmin: ${network.blockExplorer}/address/${proxyAdminAddress}`);
    console.log(`   InferenceERC20 Proxy: ${network.blockExplorer}/address/${tokenProxyAddress}`);
    console.log(`   ComputeMarketplace Proxy: ${network.blockExplorer}/address/${marketplaceProxyAddress}`);
  }

  console.log('\n⚠️  Important: Use the PROXY addresses for interactions, not the implementation addresses!');
}

if (require.main === module) {
  deployWithProxy()
    .then(() => process.exit(0))
    .catch((error) => {
      console.error('❌ Proxy deployment failed:', error);
      process.exit(1);
    });
}

export { deployWithProxy }; 