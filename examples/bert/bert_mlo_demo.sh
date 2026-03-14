#!/bin/bash
# Quick test script - matches functionality of old quicktest.sh

set -e

# Set longer timeout for RTL simulation (BERT models can take longer)
export LIVENESS_THRESHOLD=10000000

echo "Running BERT Modern Demo with Loop Rolling Test"
echo "==============================================="

# Change to demo directory
cd "$(dirname "$0")"

# Clean up any existing bert_mlo_demo build directory
if [ -d "${BSMITH_BUILD_DIR}/bert_mlo_demo" ]; then
    echo "Removing existing bert_mlo_demo build directory..."
    rm -rf "${BSMITH_BUILD_DIR}/bert_mlo_demo"
fi

# Generate folding config
echo "Generating folding configuration..."
python gen_folding_config.py \
    --simd 4 \
    --pe 4 \
    --num_layers 2 \
    -t 1 \
    -o ./configs/bert_mlo_demo.json

# Run BERT demo
echo "Running BERT demo with 2 layers..."
python bert_demo.py \
    -o bert_mlo_demo \
    -n 4 \
    -l 2 \
    -z 64 \
    -i 256 \
    -b 8 \
    -q 32 \
    --blueprint ./bert_mlo_demo.yaml

echo "Bert MLO test completed!"
