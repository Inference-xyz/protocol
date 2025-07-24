#!/bin/bash

set -e

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "📄 Loading environment variables from .env file"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "⚠️  No .env file found. Using default values."
    echo "   Copy config.env.example to .env and update values if needed."
fi

echo "🚀 Compute Marketplace Demo"
echo "=========================="

# Check if anvil is running
ANVIL_PORT=${ANVIL_PORT:-8545}
ANVIL_HOST=${ANVIL_HOST:-localhost}

if ! lsof -i :$ANVIL_PORT > /dev/null 2>&1; then
    echo "❌ Anvil is not running on port $ANVIL_PORT"
    echo "Please start anvil with: anvil --port $ANVIL_PORT"
    exit 1
fi

echo "✅ Anvil is running on port $ANVIL_PORT"

# Build contracts
echo "🔨 Building contracts..."
forge build

# Deploy contracts
echo "🚀 Deploying contracts..."
RPC_URL=${RPC_URL:-http://localhost:8545}
CLIENT_PRIVATE_KEY=${CLIENT_PRIVATE_KEY}

DEPLOY_OUTPUT=$(forge script scripts/Deploy.s.sol --rpc-url $RPC_URL --broadcast --private-key $CLIENT_PRIVATE_KEY 2>&1)

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
export MARKETPLACE_ADDRESS=$MARKETPLACE_ADDRESS
export TOKEN_ADDRESS=$TOKEN_ADDRESS
npm run demo

echo "�� Demo completed!" 