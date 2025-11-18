#!/bin/bash

PROFILE_DIR="results/profiles"
mkdir -p $PROFILE_DIR

echo "1: Nsight Systems Profiling"

nsys profile \
    --trace=cuda,nvtx \
    --output=$PROFILE_DIR/naive_timeline \
    --force-overwrite=true \
    python benchmarks/profile_naive.py

echo "Saved to: $PROFILE_DIR/naive_timeline.nsys-rep"
echo ""

echo "2: Nsight Compute Profiling (QK Kernel)"

ncu \
    --set detailed \
    --kernel-name regex:.*qk.* \
    --export=$PROFILE_DIR/naive_qk \
    --force-overwrite \
    python benchmarks/profile_naive.py

echo "Saved to: $PROFILE_DIR/naive_qk.ncu-rep"
echo ""

echo "3: Nsight Compute Profiling (Softmax Kernel)"

ncu \
    --set detailed \
    --kernel-name regex:.*softmax.* \
    --export=$PROFILE_DIR/naive_softmax \
    --force-overwrite \
    python benchmarks/profile_naive.py

echo "Saved to: $PROFILE_DIR/naive_softmax.ncu-rep"
echo ""

echo "4: Nsight Compute Profiling (AV Kernel)"

ncu \
    --set detailed \
    --kernel-name regex:.*av.* \
    --export=$PROFILE_DIR/naive_av \
    --force-overwrite \
    python benchmarks/profile_naive.py

echo "Saved to: $PROFILE_DIR/naive_av.ncu-rep"
echo ""

echo "Profiling complete. View results:"
echo "  nsys-ui $PROFILE_DIR/naive_timeline.nsys-rep"
echo "  ncu-ui $PROFILE_DIR/naive_qk.ncu-rep"
