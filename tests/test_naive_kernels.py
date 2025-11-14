import torch
import sys
import os

import cuda_attention

# add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

def test_attention_correctness():
    print("\n" + "=" * 70)
    print("CORRECTNESS TEST: CUDA vs PyTorch Reference")
    print("=" * 70)

    # use fixed seed for reproducibility
    torch.manual_seed(42)

    # create test data
    seq_len = 128  # use smaller for faster testing
    d_k = 64

    Q = torch.randn(seq_len, d_k, device='cuda')
    K = torch.randn(seq_len, d_k, device='cuda')
    V = torch.randn(seq_len, d_k, device='cuda')

    print(f"Input shapes: Q={Q.shape}, K={K.shape}, V={V.shape}")

    # Test 1: Individual Kernel (QK)
    print("\n" + "-" * 70)
    print("TEST 1: Q @ K^T with Scaling")
    print("-" * 70)

    scale = 1.0 / (d_k ** 0.5)

    # cuda implementation
    S_cuda = cuda_attention.naive_qk(Q, K, scale)

    # PyTorch reference
    S_ref = (Q @ K.T) / (d_k ** 0.5)

    # compare
    diff = (S_cuda - S_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"CUDA output: {S_cuda.shape}")
    print(f"Reference output: {S_ref.shape}")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")

    assert torch.allclose(S_cuda, S_ref, atol=1e-4, rtol=1e-3), \
        f"QK kernel failed! Max diff: {max_diff}"
    print("QK kernel PASSED")

    # Test 2: Individual Kernel (Softmax with Causal Mask)
    print("\n" + "-" * 70)
    print("TEST 2: Softmax with Causal Masking")
    print("-" * 70)

    # cuda implementation
    A_cuda = cuda_attention.naive_softmax(S_cuda, use_causal_mask=True)

    # PyTorch reference
    mask = torch.triu(torch.ones(seq_len, seq_len, device='cuda') * float('-inf'), diagonal=1)
    S_masked = S_ref + mask
    A_ref = torch.softmax(S_masked, dim=-1)

    # compare
    diff = (A_cuda - A_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"CUDA output: {A_cuda.shape}")
    print(f"Reference output: {A_ref.shape}")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")

    # check row sums
    row_sums = A_cuda.sum(dim=-1)
    print(f"Row sums min: {row_sums.min():.6f}, max: {row_sums.max():.6f}")

    # check causal property (upper triangle should be zero)
    upper_triangle = torch.triu(A_cuda, diagonal=1)
    upper_max = upper_triangle.max().item()
    print(f"Max value in upper triangle (should be ~0): {upper_max:.2e}")

    assert torch.allclose(A_cuda, A_ref, atol=1e-4, rtol=1e-3), \
        f"Softmax kernel failed! Max diff: {max_diff}"
    assert upper_max < 1e-6, "Causal masking not working!"
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        "Softmax rows don't sum to 1!"
    print("Softmax kernel PASSED")

    # Test 3: Individual Kernel - (AV)
    print("\n" + "-" * 70)
    print("TEST 3: A @ V")
    print("-" * 70)

    # cuda implementation
    O_cuda = cuda_attention.naive_av(A_cuda, V)

    # PyTorch reference
    O_ref = A_ref @ V

    # compare
    diff = (O_cuda - O_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"CUDA output: {O_cuda.shape}")
    print(f"Reference output: {O_ref.shape}")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")

    assert torch.allclose(O_cuda, O_ref, atol=1e-4, rtol=1e-3), \
        f"AV kernel failed! Max diff: {max_diff}"
    print("AV kernel PASSED")

    # Test 4: Complete Pipeline
    print("\n" + "-" * 70)
    print("TEST 4: Complete Attention Pipeline")
    print("-" * 70)

    # reset inputs
    torch.manual_seed(42)
    Q = torch.randn(seq_len, d_k, device='cuda')
    K = torch.randn(seq_len, d_k, device='cuda')
    V = torch.randn(seq_len, d_k, device='cuda')

    # cuda implementation
    O_cuda_full = cuda_attention.naive_attention(Q, K, V, use_causal_mask=True)

    # PyTorch reference (full pipeline)
    scale = 1.0 / (d_k ** 0.5)
    S_full = (Q @ K.T) * scale
    mask = torch.triu(torch.ones(seq_len, seq_len, device='cuda') * float('-inf'), diagonal=1)
    A_full = torch.softmax(S_full + mask, dim=-1)
    O_ref_full = A_full @ V

    # compare
    diff = (O_cuda_full - O_ref_full).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"CUDA output: {O_cuda_full.shape}")
    print(f"Reference output: {O_ref_full.shape}")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")

    assert torch.allclose(O_cuda_full, O_ref_full, atol=1e-4, rtol=1e-3), \
        f"Complete pipeline failed! Max diff: {max_diff}"
    print("Complete pipeline PASSED")

    # Test 5: Different Sequence Lengths
    print("\n" + "-" * 70)
    print("TEST 5: Different Sequence Lengths")
    print("-" * 70)

    for test_seq_len in [32, 64, 256, 512]:
        Q_test = torch.randn(test_seq_len, d_k, device='cuda')
        K_test = torch.randn(test_seq_len, d_k, device='cuda')
        V_test = torch.randn(test_seq_len, d_k, device='cuda')

        O_cuda_test = cuda_attention.naive_attention(Q_test, K_test, V_test, True)

        # reference
        S_test = (Q_test @ K_test.T) / (d_k ** 0.5)
        mask_test = torch.triu(torch.ones(test_seq_len, test_seq_len, device='cuda') * float('-inf'), diagonal=1)
        A_test = torch.softmax(S_test + mask_test, dim=-1)
        O_ref_test = A_test @ V_test

        assert torch.allclose(O_cuda_test, O_ref_test, atol=1e-4, rtol=1e-3), \
            f"Failed for seq_len={test_seq_len}"

        print(f"seq_len={test_seq_len:4d} PASSED")

    # Summary
    print("\n" + "=" * 70)
    print("ALL CORRECTNESS TESTS PASSED, UWOOOO LETSGOOO!")
    print("=" * 70)
    print("\nYour CUDA implementation is mathematically correct!")
    print("QK multiplication with scaling")
    print("Softmax with causal masking")
    print("Attention-value multiplication")
    print("Complete pipeline")
    print("Multiple sequence lengths")
    print("\nReady for Phase 4: Flash Attention optimization! 🚀")

if __name__ == "__main__":
    test_attention_correctness()
