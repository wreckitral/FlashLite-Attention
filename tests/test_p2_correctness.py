"""
Test correctness of p2 (FlashLite Fused) kernel
Tests: flashLite_attention (single fused kernel)
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

def compute_relative_error(pred, target):
    """Compute Mean Relative Error"""
    rel_err = torch.abs(pred - target) / (torch.abs(target) + 1e-8)
    return torch.mean(rel_err).item()

def test_p2_flashlite():
    """Test p2 (FlashLite Fused) kernel"""
    print("="*80)
    print("Testing p2 (FlashLite Fused): flashLite_attention")
    print("="*80)

    # Test configurations
    configs = [
        (128, 128, 64),
        (256, 256, 64),
        (512, 512, 64),
        (1024, 1024, 64),
        (2048, 2048, 64),
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

        # Test FlashLite
        print("  Testing flashLite_attention (fused kernel)...")
        try:
            flash_output = cuda_attention.flashLite_attention(Q, K, V)

            # Compute errors
            mae = compute_mae(flash_output, ref_output)
            max_err = compute_max_error(flash_output, ref_output)
            rel_err = compute_relative_error(flash_output, ref_output)

            print(f"    MAE: {mae:.6e}")
            print(f"    Max Error: {max_err:.6e}")
            print(f"    Relative Error: {rel_err:.6e}")

            # Check if passed
            tolerance = 1e-3  # Slightly relaxed for fused kernel
            passed = mae < tolerance
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  Status: {status} (tolerance: {tolerance})")

            # Additional checks
            print(f"  Additional checks:")
            print(f"    Output contains NaN: {torch.isnan(flash_output).any().item()}")
            print(f"    Output contains Inf: {torch.isinf(flash_output).any().item()}")
            print(f"    Output range: [{flash_output.min().item():.4f}, {flash_output.max().item():.4f}]")
            print(f"    Reference range: [{ref_output.min().item():.4f}, {ref_output.max().item():.4f}]")

            results.append({
                'M': M,
                'N': N,
                'd_k': d_k,
                'mae': mae,
                'max_err': max_err,
                'rel_err': rel_err,
                'passed': passed,
                'has_nan': torch.isnan(flash_output).any().item(),
                'has_inf': torch.isinf(flash_output).any().item(),
            })

        except Exception as e:
            print(f"  ✗ FAILED with exception: {str(e)}")
            results.append({
                'M': M,
                'N': N,
                'd_k': d_k,
                'mae': float('inf'),
                'max_err': float('inf'),
                'rel_err': float('inf'),
                'passed': False,
                'has_nan': False,
                'has_inf': False,
                'error': str(e)
            })

        # Cleanup
        del Q, K, V, ref_output
        if 'flash_output' in locals():
            del flash_output
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - p2 (FlashLite Fused)")
    print("="*80)
    passed_count = sum(r['passed'] for r in results)
    print(f"Passed: {passed_count}/{len(results)} tests")

    if passed_count == len(results):
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
        for r in results:
            if not r['passed']:
                error_msg = r.get('error', f"MAE={r['mae']:.6e}")
                print(f"  Failed: M={r['M']}, N={r['N']} - {error_msg}")

    # Print detailed results
    print("\nDetailed Results:")
    print(f"{'Config':<15} {'MAE':<12} {'Max Error':<12} {'Rel Error':<12} {'Status':<10}")
    print("-"*80)
    for r in results:
        config = f"{r['M']}x{r['N']}x{r['d_k']}"
        status = "PASS" if r['passed'] else "FAIL"
        mae_str = f"{r['mae']:.6e}" if r['mae'] != float('inf') else "ERROR"
        max_err_str = f"{r['max_err']:.6e}" if r['max_err'] != float('inf') else "ERROR"
        rel_err_str = f"{r['rel_err']:.6e}" if r['rel_err'] != float('inf') else "ERROR"
        print(f"{config:<15} {mae_str:<12} {max_err_str:<12} {rel_err_str:<12} {status:<10}")

    return results

if __name__ == "__main__":
    results = test_p2_flashlite()
