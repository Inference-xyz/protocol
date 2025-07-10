// @ts-ignore
import { config } from 'dotenv';

config();

export interface NetworkConfig {
  name: string;
  chainId: number;
  rpcUrl: string;
  blockExplorer: string;
  nativeCurrency: {
    name: string;
    symbol: string;
    decimals: number;
  };
}

export interface DeploymentConfig {
  token: {
    name: string;
    symbol: string;
    decimals: number;
  };
  marketplace: {
    minBounty: string;
    defaultTimeout: number;
  };
}

export const networks: Record<string, NetworkConfig> = {
  localhost: {
    name: 'Localhost',
    chainId: 31337,
    rpcUrl: 'http://localhost:8545',
    blockExplorer: '',
    nativeCurrency: {
      name: 'Ether',
      symbol: 'ETH',
      decimals: 18,
    },
  },
  sepolia: {
    name: 'Sepolia',
    chainId: 11155111,
    rpcUrl: process.env.SEPOLIA_RPC_URL || '',
    blockExplorer: 'https://sepolia.etherscan.io',
    nativeCurrency: {
      name: 'Ether',
      symbol: 'ETH',
      decimals: 18,
    },
  },
  mainnet: {
    name: 'Ethereum Mainnet',
    chainId: 1,
    rpcUrl: process.env.MAINNET_RPC_URL || '',
    blockExplorer: 'https://etherscan.io',
    nativeCurrency: {
      name: 'Ether',
      symbol: 'ETH',
      decimals: 18,
    },
  },
  polygon: {
    name: 'Polygon',
    chainId: 137,
    rpcUrl: process.env.POLYGON_RPC_URL || '',
    blockExplorer: 'https://polygonscan.com',
    nativeCurrency: {
      name: 'Matic',
      symbol: 'MATIC',
      decimals: 18,
    },
  },
  arbitrum: {
    name: 'Arbitrum One',
    chainId: 42161,
    rpcUrl: process.env.ARBITRUM_RPC_URL || '',
    blockExplorer: 'https://arbiscan.io',
    nativeCurrency: {
      name: 'Ether',
      symbol: 'ETH',
      decimals: 18,
    },
  },
};

export const deploymentConfig: DeploymentConfig = {
  token: {
    name: process.env.TOKEN_NAME || 'Inference Token',
    symbol: process.env.TOKEN_SYMBOL || 'INF',
    decimals: parseInt(process.env.TOKEN_DECIMALS || '18'),
  },
  marketplace: {
    minBounty: process.env.MIN_BOUNTY || '1000000000000000000', // 1 token
    defaultTimeout: parseInt(process.env.DEFAULT_TIMEOUT || '86400'), // 24 hours
  },
};

export const getPrivateKey = (): string => {
  const privateKey = process.env.PRIVATE_KEY;
  if (!privateKey) {
    throw new Error('PRIVATE_KEY environment variable is required');
  }
  return privateKey;
};

export const getEtherscanApiKey = (network: string): string => {
  switch (network) {
    case 'mainnet':
    case 'sepolia':
      return process.env.ETHERSCAN_API_KEY || '';
    case 'polygon':
      return process.env.POLYGONSCAN_API_KEY || '';
    case 'arbitrum':
      return process.env.ARBISCAN_API_KEY || '';
    default:
      return '';
  }
}; 