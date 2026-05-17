"""
Profile Bottleneck Metrics - Collects data for Table 2
Metrics: Memory Throughput (GB/s, % Peak), Compute Throughput (TFLOPS, % Peak)

This script runs the kernels and saves profiles that need to be analyzed with:
ncu --csv --metrics <metrics> python profile_bottleneck.py

Or use the generated shell script: run_bottleneck_profile.sh
"""
import torch
import cuda_attention
import math
from pathlib import Path
import subprocess
import sys

# Test configuration
TEST_CONFIG = (4096, 4096, 64)

def reference_attention(Q, K, V):
    """PyTorch reference implementation"""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    seq_len = Q.size(0)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output

def run_p0_pipeline(Q, K, V):
    """p0 (Naive): naive_qk + naive_softmax + naive_av"""
    scale = 1.0 / math.sqrt(Q.size(-1))
    qk_scores = cuda_attention.naive_qk(Q, K, scale)
    attn_weights = cuda_attention.naive_softmax(qk_scores, use_causal_mask=True)
    output = cuda_attention.naive_av(attn_weights, V)
    return output

def run_p1_pipeline(Q, K, V):
    """p1 (Tiled): online_softmax + tiled_av"""
    attn_weights = cuda_attention.online_softmax(Q, K)
    output = cuda_attention.tiled_av(attn_weights, V)
    return output

def run_p2_pipeline(Q, K, V):
    """p2 (FlashLite): flashLite_attention (fused)"""
    output = cuda_attention.flashLite_attention(Q, K, V)
    return output

def profile_kernel(kernel_name, kernel_fn, Q, K, V):
    """Run kernel once for profiling"""
    print(f"\nProfiling {kernel_name}...")

    # Warmup
    for _ in range(3):
        _ = kernel_fn(Q, K, V)
    torch.cuda.synchronize()

    # Profile run
    output = kernel_fn(Q, K, V)
    torch.cuda.synchronize()

    print(f"  ✓ {kernel_name} completed")
    return output

def generate_ncu_script():
    """Generate shell script to run NCU profiling"""

    metrics = [
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",  # Memory throughput %
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",    # Compute throughput %
        "dram__bytes.sum",                                      # Total DRAM bytes
        "gpu__time_duration.sum",                              # GPU time
    ]

    metrics_str = ",".join(metrics)

    # Get absolute path to this script
    script_dir = Path(__file__).parent.absolute()
    results_dir = script_dir.parent / "results" / "profiles"

    script_content = f"""#!/bin/bash
# Auto-generated script to profile bottleneck metrics (Table 2)
# Run this with: bash run_bottleneck_profile.sh

SCRIPT_DIR="{script_dir}"
RESULTS_DIR="{results_dir}"
mkdir -p $RESULTS_DIR

echo "========================================"
echo "Profiling Bottleneck Metrics (Table 2)"
echo "========================================"

# Profile PyTorch Reference
echo "\\n1. Profiling PyTorch Reference..."
ncu --csv \\
    --metrics {metrics_str} \\
    --target-processes all \\
    --export $RESULTS_DIR/pytorch_bottleneck \\
    python $SCRIPT_DIR/profile_bottleneck.py pytorch

# Profile p0 (Naive)
echo "\\n2. Profiling p0 (Naive)..."
ncu --csv \\
    --metrics {metrics_str} \\
    --target-processes all \\
    --export $RESULTS_DIR/p0_bottleneck \\
    python $SCRIPT_DIR/profile_bottleneck.py p0

# Profile p1 (Tiled)
echo "\\n3. Profiling p1 (Tiled)..."
ncu --csv \\
    --metrics {metrics_str} \\
    --target-processes all \\
    --export $RESULTS_DIR/p1_bottleneck \\
    python $SCRIPT_DIR/profile_bottleneck.py p1

# Profile p2 (FlashLite)
echo "\\n4. Profiling p2 (FlashLite)..."
ncu --csv \\
    --metrics {metrics_str} \\
    --target-processes all \\
    --export $RESULTS_DIR/p2_bottleneck \\
    python $SCRIPT_DIR/profile_bottleneck.py p2

echo "\\n========================================"
echo "Profiling complete!"
echo "Results saved to: $RESULTS_DIR"
echo "========================================"
"""

    script_path = Path("run_bottleneck_profile.sh")
    script_path.write_text(script_content)
    script_path.chmod(0o755)  # Make executable

    print(f"\n✓ Generated profiling script: {script_path}")
    print(f"  Run with: bash {script_path}")

def main():
    print("="*80)
    print("BOTTLENECK PROFILING - Table 2")
    print("="*80)

    M, N, d_k = TEST_CONFIG
    print(f"Configuration: M={M}, N={N}, d_k={d_k}\n")

    # Check if running under NCU profiler
    if len(sys.argv) > 1:
        kernel_choice = sys.argv[1]

        # Create test data
        Q = torch.randn(M, d_k, device='cuda', dtype=torch.float32).contiguous()
        K = torch.randn(N, d_k, device='cuda', dtype=torch.float32).contiguous()
        V = torch.randn(N, d_k, device='cuda', dtype=torch.float32).contiguous()

        # Run selected kernel
        if kernel_choice == "pytorch":
            profile_kernel("PyTorch Reference", reference_attention, Q, K, V)
        elif kernel_choice == "p0":
            profile_kernel("p0 (Naive)", run_p0_pipeline, Q, K, V)
        elif kernel_choice == "p1":
            profile_kernel("p1 (Tiled)", run_p1_pipeline, Q, K, V)
        elif kernel_choice == "p2":
            profile_kernel("p2 (FlashLite)", run_p2_pipeline, Q, K, V)
        else:
            print(f"Unknown kernel: {kernel_choice}")
            print("Usage: python profile_bottleneck.py [pytorch|p0|p1|p2]")
            sys.exit(1)

        print("\nProfile data collected. Check NCU output.")
    else:
        # Generate profiling script
        print("Generating NCU profiling script...")
        generate_ncu_script()

        print("\n" + "="*80)
        print("INSTRUCTIONS:")
        print("="*80)
        print("1. Run the generated script:")
        print("   bash run_bottleneck_profile.sh")
        print()
        print("2. Or manually run NCU for each kernel:")
        print("   ncu --csv --metrics <metrics> python profile_bottleneck.py [pytorch|p0|p1|p2]")
        print()
        print("3. Results will be saved to: results/profiles/")
        print("="*80)

if __name__ == "__main__":
    main()
