#!/bin/bash

set -e

echo "🚀 Compute Marketplace Demo"
echo "=========================="

# Check if anvil is running
if ! lsof -i :8545 > /dev/null 2>&1; then
    echo "❌ Anvil is not running on port 8545"
    echo "Please start anvil with: anvil"
    exit 1
fi

echo "✅ Anvil is running"

# Build contracts
echo "🔨 Building contracts..."
forge build

# Deploy contracts
echo "🚀 Deploying contracts..."
DEPLOY_OUTPUT=$(forge script scripts/Deploy.s.sol --rpc-url http://localhost:8545 --broadcast --private-key 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80 2>&1)

# Extract contract addresses
MARKETPLACE_ADDRESS=$(echo "$DEPLOY_OUTPUT" | grep "ComputeMarketplace deployed at:" | awk '{print $4}')
TOKEN_ADDRESS=$(echo "$DEPLOY_OUTPUT" | grep "InfToken deployed at:" | awk '{print $4}')

if [ -z "$MARKETPLACE_ADDRESS" ] || [ -z "$TOKEN_ADDRESS" ]; then
    echo "❌ Failed to extract contract addresses"
    exit 1
fi

echo "✅ Contracts deployed:"
echo "  Marketplace: $MARKETPLACE_ADDRESS"
echo "  Token: $TOKEN_ADDRESS"

# Setup demo files
echo "🔧 Setting up demo files..."
cd scripts/ezkl_demo
python3 setup.py
cd ../..

# Run demo
echo "🎬 Running demo..."
MARKETPLACE_ADDRESS=$MARKETPLACE_ADDRESS TOKEN_ADDRESS=$TOKEN_ADDRESS npm run demo

echo "�� Demo completed!" 