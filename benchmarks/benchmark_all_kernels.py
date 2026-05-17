"""
Performance Benchmark - Collects data for Table 1 and Table 5
Benchmarks: PyTorch Reference, p0 (Naive), p1 (Tiled), p2 (FlashLite)
Metrics: Execution time, speedup, memory usage, correctness (MAE)
"""
import torch
import cuda_attention
import numpy as np
import pandas as pd
import math
from pathlib import Path
import time

# Configuration
TEST_CONFIG = (4096, 4096, 64)  # (M, N, d_k) - Representative config for thesis
NUM_WARMUP = 10
NUM_RUNS = 100

def reference_attention(Q, K, V):
    """PyTorch reference implementation"""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Causal mask
    seq_len = Q.size(0)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))

    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)

    return output

def benchmark_kernel(kernel_fn, *args, num_warmup=10, num_runs=100):
    """
    Benchmark a kernel function with CUDA events
    Returns: dict with timing statistics
    """
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(num_warmup):
        _ = kernel_fn(*args)
    torch.cuda.synchronize()

    # Measurement
    times = []
    for _ in range(num_runs):
        start_event.record()
        _ = kernel_fn(*args)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        times.append(elapsed_time)

    times_array = np.array(times)
    return {
        'mean_ms': np.mean(times_array),
        'std_ms': np.std(times_array),
        'min_ms': np.min(times_array),
        'max_ms': np.max(times_array),
    }

def measure_memory_usage(kernel_fn, *args):
    """
    Measure peak GPU memory usage
    Returns: peak memory in MB
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Run kernel
    _ = kernel_fn(*args)
    torch.cuda.synchronize()

    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB
    return peak_memory

def compute_mae(pred, target):
    """Compute Mean Absolute Error"""
    return torch.mean(torch.abs(pred - target)).item()

def run_p0_pipeline(Q, K, V):
    """Run p0 (Naive) pipeline: naive_qk + naive_softmax + naive_av"""
    scale = 1.0 / math.sqrt(Q.size(-1))
    qk_scores = cuda_attention.naive_qk(Q, K, scale)
    attn_weights = cuda_attention.naive_softmax(qk_scores, use_causal_mask=True)
    output = cuda_attention.naive_av(attn_weights, V)
    return output

def run_p1_pipeline(Q, K, V):
    """Run p1 (Tiled) pipeline: tiled_qk + online_softmax + tiled_av"""
    attn_weights = cuda_attention.online_softmax(Q, K)
    output = cuda_attention.tiled_av(attn_weights, V)
    return output

def run_p2_pipeline(Q, K, V):
    """Run p2 (FlashLite) pipeline: flashLite_attention (fused)"""
    output = cuda_attention.flashLite_attention(Q, K, V)
    return output

def main():
    print("="*80)
    print("PERFORMANCE BENCHMARK - Tables 1 & 5")
    print("="*80)

    M, N, d_k = TEST_CONFIG
    print(f"\nConfiguration: M={M}, N={N}, d_k={d_k}")
    print(f"Warmup runs: {NUM_WARMUP}, Measurement runs: {NUM_RUNS}\n")

    # Create test data
    print("Creating test data...")
    Q = torch.randn(M, d_k, device='cuda', dtype=torch.float32).contiguous()
    K = torch.randn(N, d_k, device='cuda', dtype=torch.float32).contiguous()
    V = torch.randn(N, d_k, device='cuda', dtype=torch.float32).contiguous()

    # Get reference output for correctness
    print("Computing reference output...")
    ref_output = reference_attention(Q, K, V)

    results = []

    # ========================================================================
    # 1. PyTorch Reference
    # ========================================================================
    print("\n" + "-"*80)
    print("1. Benchmarking PyTorch Reference")
    print("-"*80)

    pytorch_times = benchmark_kernel(
        reference_attention, Q, K, V,
        num_warmup=NUM_WARMUP,
        num_runs=NUM_RUNS
    )
    pytorch_memory = measure_memory_usage(reference_attention, Q, K, V)
    pytorch_mae = 0.0  # Reference is perfect

    print(f"  Mean: {pytorch_times['mean_ms']:.3f} ± {pytorch_times['std_ms']:.3f} ms")
    print(f"  Min:  {pytorch_times['min_ms']:.3f} ms | Max: {pytorch_times['max_ms']:.3f} ms")
    print(f"  Memory: {pytorch_memory:.2f} MB")
    print(f"  MAE: {pytorch_mae:.6e}")

    results.append({
        'kernel': 'PyTorch Reference',
        'mean_ms': pytorch_times['mean_ms'],
        'std_ms': pytorch_times['std_ms'],
        'min_ms': pytorch_times['min_ms'],
        'max_ms': pytorch_times['max_ms'],
        'speedup_vs_pytorch': 1.0,
        'speedup_vs_p0': None,
        'memory_mb': pytorch_memory,
        'mae': pytorch_mae,
    })

    # ========================================================================
    # 2. p0 (Naive Baseline)
    # ========================================================================
    print("\n" + "-"*80)
    print("2. Benchmarking p0 (Naive: 3 kernels)")
    print("-"*80)

    p0_times = benchmark_kernel(
        run_p0_pipeline, Q, K, V,
        num_warmup=NUM_WARMUP,
        num_runs=NUM_RUNS
    )
    p0_memory = measure_memory_usage(run_p0_pipeline, Q, K, V)
    p0_output = run_p0_pipeline(Q, K, V)
    p0_mae = compute_mae(p0_output, ref_output)

    speedup_vs_pytorch = pytorch_times['mean_ms'] / p0_times['mean_ms']

    print(f"  Mean: {p0_times['mean_ms']:.3f} ± {p0_times['std_ms']:.3f} ms")
    print(f"  Min:  {p0_times['min_ms']:.3f} ms | Max: {p0_times['max_ms']:.3f} ms")
    print(f"  Memory: {p0_memory:.2f} MB")
    print(f"  MAE: {p0_mae:.6e}")
    print(f"  Speedup vs PyTorch: {speedup_vs_pytorch:.2f}x")

    results.append({
        'kernel': 'p0 (Naive: 3 kernels)',
        'mean_ms': p0_times['mean_ms'],
        'std_ms': p0_times['std_ms'],
        'min_ms': p0_times['min_ms'],
        'max_ms': p0_times['max_ms'],
        'speedup_vs_pytorch': speedup_vs_pytorch,
        'speedup_vs_p0': 1.0,
        'memory_mb': p0_memory,
        'mae': p0_mae,
    })

    # ========================================================================
    # 3. p1 (Tiled + Online Softmax)
    # ========================================================================
    print("\n" + "-"*80)
    print("3. Benchmarking p1 (Tiled + Online Softmax)")
    print("-"*80)

    p1_times = benchmark_kernel(
        run_p1_pipeline, Q, K, V,
        num_warmup=NUM_WARMUP,
        num_runs=NUM_RUNS
    )
    p1_memory = measure_memory_usage(run_p1_pipeline, Q, K, V)
    p1_output = run_p1_pipeline(Q, K, V)
    p1_mae = compute_mae(p1_output, ref_output)

    speedup_vs_pytorch = pytorch_times['mean_ms'] / p1_times['mean_ms']
    speedup_vs_p0 = p0_times['mean_ms'] / p1_times['mean_ms']

    print(f"  Mean: {p1_times['mean_ms']:.3f} ± {p1_times['std_ms']:.3f} ms")
    print(f"  Min:  {p1_times['min_ms']:.3f} ms | Max: {p1_times['max_ms']:.3f} ms")
    print(f"  Memory: {p1_memory:.2f} MB")
    print(f"  MAE: {p1_mae:.6e}")
    print(f"  Speedup vs PyTorch: {speedup_vs_pytorch:.2f}x")
    print(f"  Speedup vs p0: {speedup_vs_p0:.2f}x")

    results.append({
        'kernel': 'p1 (Tiled + Online Softmax)',
        'mean_ms': p1_times['mean_ms'],
        'std_ms': p1_times['std_ms'],
        'min_ms': p1_times['min_ms'],
        'max_ms': p1_times['max_ms'],
        'speedup_vs_pytorch': speedup_vs_pytorch,
        'speedup_vs_p0': speedup_vs_p0,
        'memory_mb': p1_memory,
        'mae': p1_mae,
    })

    # ========================================================================
    # 4. p2 (FlashLite Fused)
    # ========================================================================
    print("\n" + "-"*80)
    print("4. Benchmarking p2 (FlashLite Fused)")
    print("-"*80)

    p2_times = benchmark_kernel(
        run_p2_pipeline, Q, K, V,
        num_warmup=NUM_WARMUP,
        num_runs=NUM_RUNS
    )
    p2_memory = measure_memory_usage(run_p2_pipeline, Q, K, V)
    p2_output = run_p2_pipeline(Q, K, V)
    p2_mae = compute_mae(p2_output, ref_output)

    speedup_vs_pytorch = pytorch_times['mean_ms'] / p2_times['mean_ms']
    speedup_vs_p0 = p0_times['mean_ms'] / p2_times['mean_ms']

    print(f"  Mean: {p2_times['mean_ms']:.3f} ± {p2_times['std_ms']:.3f} ms")
    print(f"  Min:  {p2_times['min_ms']:.3f} ms | Max: {p2_times['max_ms']:.3f} ms")
    print(f"  Memory: {p2_memory:.2f} MB")
    print(f"  MAE: {p2_mae:.6e}")
    print(f"  Speedup vs PyTorch: {speedup_vs_pytorch:.2f}x")
    print(f"  Speedup vs p0: {speedup_vs_p0:.2f}x")

    results.append({
        'kernel': 'p2 (FlashLite Fused)',
        'mean_ms': p2_times['mean_ms'],
        'std_ms': p2_times['std_ms'],
        'min_ms': p2_times['min_ms'],
        'max_ms': p2_times['max_ms'],
        'speedup_vs_pytorch': speedup_vs_pytorch,
        'speedup_vs_p0': speedup_vs_p0,
        'memory_mb': p2_memory,
        'mae': p2_mae,
    })

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    df = pd.DataFrame(results)

    print("\nTable 1: Performance Metrics and Speedup")
    print("-"*80)
    print(f"{'Kernel':<30} {'Mean (ms)':<12} {'Std (ms)':<12} {'Speedup vs PyTorch':<20} {'Speedup vs p0':<15}")
    print("-"*80)
    for _, row in df.iterrows():
        speedup_p0_str = f"{row['speedup_vs_p0']:.2f}x" if row['speedup_vs_p0'] is not None else "-"
        print(f"{row['kernel']:<30} {row['mean_ms']:<12.3f} {row['std_ms']:<12.3f} {row['speedup_vs_pytorch']:<20.2f}x {speedup_p0_str:<15}")

    print("\nTable 5: Memory and Correctness")
    print("-"*80)
    print(f"{'Kernel':<30} {'Peak Memory (MB)':<20} {'MAE vs PyTorch':<15}")
    print("-"*80)
    for _, row in df.iterrows():
        print(f"{row['kernel']:<30} {row['memory_mb']:<20.2f} {row['mae']:<15.6e}")

    # Save results
    results_dir = Path("results/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "performance_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")

    # Cleanup
    del Q, K, V, ref_output, p0_output, p1_output, p2_output
    torch.cuda.empty_cache()

    return df

if __name__ == "__main__":
    results = main()
