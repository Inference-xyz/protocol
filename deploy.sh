#!/bin/bash

set -e

source .env

forge script Deploy.s.sol:DeployScript \
    --rpc-url $SEPOLIA_RPC_URL \
    --private-key $SEPOLIA_PRIVATE_KEY \
    --broadcast \
    --verify \
    --etherscan-api-key $ETHERSCAN_API_KEY \
    --chain sepolia
