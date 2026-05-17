"""
Test correctness of p0 (Naive Baseline) kernels
Tests: naive_qk + naive_softmax + naive_av
"""
import torch
import cuda_attention
import numpy as np
import math
from pathlib import Path

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

def compute_mae(pred, target):
    """Compute Mean Absolute Error"""
    return torch.mean(torch.abs(pred - target)).item()

def compute_max_error(pred, target):
    """Compute Maximum Absolute Error"""
    return torch.max(torch.abs(pred - target)).item()

def test_p0_naive_kernels():
    """Test p0 (Naive) kernels step by step"""
    print("="*80)
    print("Testing p0 (Naive Baseline): naive_qk + naive_softmax + naive_av")
    print("="*80)

    # Test configurations
    configs = [
        (128, 128, 64),
        (256, 256, 64),
        (512, 512, 64),
        (1024, 1024, 64),
    ]

    results = []

    for M, N, d_k in configs:
        print(f"\nTesting M={M}, N={N}, d_k={d_k}")

        # Create test data
        Q = torch.randn(M, d_k, device='cuda', dtype=torch.float32).contiguous()
        K = torch.randn(N, d_k, device='cuda', dtype=torch.float32).contiguous()
        V = torch.randn(N, d_k, device='cuda', dtype=torch.float32).contiguous()

        # Reference output
        ref_output = reference_attention(Q, K, V)

        # Test Step 1: naive_qk
        print("  Step 1: Testing naive_qk...")
        scale = 1.0 / math.sqrt(d_k)
        qk_scores = cuda_attention.naive_qk(Q, K, scale)

        # Reference QK
        ref_qk = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        qk_mae = compute_mae(qk_scores, ref_qk)
        qk_max_err = compute_max_error(qk_scores, ref_qk)
        print(f"    QK MAE: {qk_mae:.6e}, Max Error: {qk_max_err:.6e}")

        # Test Step 2: naive_softmax
        print("  Step 2: Testing naive_softmax...")
        attn_weights = cuda_attention.naive_softmax(qk_scores, use_causal_mask=True)

        # Reference softmax
        seq_len = qk_scores.size(0)
        mask = torch.triu(torch.ones(seq_len, seq_len, device='cuda'), diagonal=1).bool()
        ref_qk_masked = ref_qk.masked_fill(mask, float('-inf'))
        ref_attn = torch.softmax(ref_qk_masked, dim=-1)

        softmax_mae = compute_mae(attn_weights, ref_attn)
        softmax_max_err = compute_max_error(attn_weights, ref_attn)
        print(f"    Softmax MAE: {softmax_mae:.6e}, Max Error: {softmax_max_err:.6e}")

        # Test Step 3: naive_av
        print("  Step 3: Testing naive_av...")
        output = cuda_attention.naive_av(attn_weights, V)

        # Compare with reference
        av_mae = compute_mae(output, ref_output)
        av_max_err = compute_max_error(output, ref_output)
        print(f"    AV MAE: {av_mae:.6e}, Max Error: {av_max_err:.6e}")

        # End-to-end test
        print("  End-to-end: Testing full p0 pipeline...")
        p0_output = cuda_attention.naive_av(
            cuda_attention.naive_softmax(
                cuda_attention.naive_qk(Q, K, scale),
                use_causal_mask=True
            ),
            V
        )

        e2e_mae = compute_mae(p0_output, ref_output)
        e2e_max_err = compute_max_error(p0_output, ref_output)
        print(f"    End-to-End MAE: {e2e_mae:.6e}, Max Error: {e2e_max_err:.6e}")

        # Check if passed
        tolerance = 1e-4
        passed = e2e_mae < tolerance
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Status: {status} (tolerance: {tolerance})")

        results.append({
            'M': M,
            'N': N,
            'd_k': d_k,
            'qk_mae': qk_mae,
            'qk_max_err': qk_max_err,
            'softmax_mae': softmax_mae,
            'softmax_max_err': softmax_max_err,
            'av_mae': av_mae,
            'av_max_err': av_max_err,
            'e2e_mae': e2e_mae,
            'e2e_max_err': e2e_max_err,
            'passed': passed
        })

        # Cleanup
        del Q, K, V, ref_output, qk_scores, attn_weights, output, p0_output
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - p0 (Naive Baseline)")
    print("="*80)
    passed_count = sum(r['passed'] for r in results)
    print(f"Passed: {passed_count}/{len(results)} tests")

    if passed_count == len(results):
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
        for r in results:
            if not r['passed']:
                print(f"  Failed: M={r['M']}, N={r['N']}, MAE={r['e2e_mae']:.6e}")

    # Print detailed results
    print("\nDetailed Results:")
    print(f"{'Config':<15} {'QK MAE':<12} {'Softmax MAE':<12} {'AV MAE':<12} {'E2E MAE':<12}")
    print("-"*80)
    for r in results:
        config = f"{r['M']}x{r['N']}x{r['d_k']}"
        print(f"{config:<15} {r['qk_mae']:<12.6e} {r['softmax_mae']:<12.6e} {r['av_mae']:<12.6e} {r['e2e_mae']:<12.6e}")

    return results

if __name__ == "__main__":
    results = test_p0_naive_kernels()
