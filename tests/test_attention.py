import sys
import os

# add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

import torch
import torch.nn.functional as F
from ref_attention import ReferenceAttention, load_gpt2_attention_weights


def test_1_shapes():
    print("\n" + "=" * 60)
    print("TEST 1: Shape Transformations")
    print("=" * 60)

    model = ReferenceAttention(hidden_size=768, num_heads=12)

    # test input
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 768)

    print(f"Input: {x.shape}")

    # forward pass
    output, attn = model(x, return_attention_weights=True)

    print(f"Output: {output.shape}")
    print(f"Attention: {attn.shape}")

    # verify shapes
    assert output.shape == (batch_size, seq_len, 768)
    assert attn.shape == (batch_size, 12, seq_len, seq_len)

    print("[PASS] All shapes correct!")


def test_2_causal_mask():
    print("\n" + "=" * 60)
    print("TEST 2: Causal Masking")
    print("=" * 60)

    model = ReferenceAttention(hidden_size=768, num_heads=12)

    seq_len = 5
    x = torch.randn(1, seq_len, 768)

    # causal mask
    _, attn = model(x, use_causal_mask=True, return_attention_weights=True)

    print("Attention matrix (head 0):")
    print(attn[0, 0])

    # check upper triangle is zero
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert attn[0, 0, i, j].item() < 1e-6, \
                f"Token {i} attends to future token {j}!"

    print("[PASS] Causal masking works!")


def test_3_attention_sums():
    print("\n" + "=" * 60)
    print("TEST 3: Attention Probabilities")
    print("=" * 60)

    model = ReferenceAttention(hidden_size=768, num_heads=12)

    x = torch.randn(2, 10, 768)
    _, attn = model(x, return_attention_weights=True)

    # each row should sum to 1
    row_sums = attn.sum(dim=-1)

    print(f"Min row sum: {row_sums.min():.6f}")
    print(f"Max row sum: {row_sums.max():.6f}")

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    print("[PASS] All rows sum to 1.0!")


def test_4_different_seq_lengths():
    print("\n" + "=" * 60)
    print("TEST 4: Different Sequence Lengths")
    print("=" * 60)

    model = ReferenceAttention(hidden_size=768, num_heads=12)

    for seq_len in [1, 10, 64, 128, 256]:
        x = torch.randn(1, seq_len, 768)
        output, attn = model(x, return_attention_weights=True)

        assert output.shape == (1, seq_len, 768)
        assert attn.shape == (1, 12, seq_len, seq_len)

        print(f"[] seq_len={seq_len:4d} works")

    print("[PASS] All sequence lengths work!")


def test_5_gpt2_comparison():
    print("\n" + "=" * 60)
    print("TEST 5: GPT-2 Comparison")
    print("=" * 60)

    print("Loading GPT-2...")

    # load our model with GPT-2 weights
    our_model = load_gpt2_attention_weights('gpt2', layer_idx=0)
    our_model.eval()

    # load actual GPT-2 for comparison
    from transformers import GPT2LMHeadModel
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_attn = gpt2.transformer.h[0].attn
    gpt2_attn.eval()

    # test
    torch.manual_seed(42)
    x = torch.randn(2, 10, 768)

    with torch.no_grad():
        our_output, _ = our_model(x)
        gpt2_output = gpt2_attn(x)[0]

    # compare
    is_close = torch.allclose(
        our_output,
        gpt2_output,
        rtol=1e-4,  # Relative tolerance: 0.01%
        atol=1e-5   # Absolute tolerance: 0.00001
    )

    if not is_close:
        diff = (our_output - gpt2_output).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"❌ Max difference: {max_diff:.2e}")
        print(f"❌ Mean difference: {mean_diff:.2e}")
    else:
        diff = (our_output - gpt2_output).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"Max difference: {max_diff:.2e}")
        print(f"Mean difference: {mean_diff:.2e}")

    assert is_close, "Outputs differ beyond acceptable tolerance"

    print("[PASS] Matches GPT-2!")


def test_6_memory_footprint():
    print("\n" + "=" * 60)
    print("TEST 6: Memory Footprint")
    print("=" * 60)

    print(f"{'seq_len':<10} {'Scores (MB)':<15} {'Total (MB)'}")
    print("-" * 40)

    for seq_len in [128, 256, 512, 1024]:
        # scores shape: (1, 12, seq_len, seq_len)
        elements = 1 * 12 * seq_len * seq_len
        mb = elements * 4 / (1024 ** 2)  # FP32
        total = mb * 2  # scores + attention

        print(f"{seq_len:<10} {mb:<15.2f} {total:.2f}")

    print("\nMemory grows O(n²)!")


def run_all_tests():
    print("\n" + "=" * 60)
    print("PHASE 2.2 & 2.3: REFERENCE ATTENTION TESTS")
    print("=" * 60)

    try:
        test_1_shapes()
        test_2_causal_mask()
        test_3_attention_sums()
        test_4_different_seq_lengths()
        test_5_gpt2_comparison()
        test_6_memory_footprint()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("\n")

        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
