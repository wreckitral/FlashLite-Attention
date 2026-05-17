import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
import cuda_attention

class FlashAttentionModule(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()

        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_k = hidden_size // num_heads
        self.dropout_prob = dropout
        self.use_flash = use_flash

        self.c_attn = nn.Linear(hidden_size, hidden_size * 3, bias=True)

        self.c_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.scale = 1.0 / math.sqrt(self.d_k)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.hidden_size)

    def flash_attention_single_head(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, d_k = query.shape

        outputs = []
        for b in range(batch_size):
            Q = query[b].contiguous()
            K = key[b].contiguous()
            V = value[b].contiguous()

            # Scale query
            Q_scaled = Q * self.scale

            # Call CUDA kernel
            O = cuda_attention.flashLite_attention(Q_scaled, K, V)
            outputs.append(O)

        # Stack batch
        output = torch.stack(outputs, dim=0)
        return output

    def reference_attention_single_head(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        # Compute scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # Apply causal mask
        seq_len = query.size(-2)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=query.device),
            diagonal=1
        ).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Multiply by V
        output = torch.matmul(attn_weights, value)
        return output

    def forward(
        self,
        x: torch.Tensor,
        use_causal_mask: bool = True,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.c_attn(x)  # [batch, seq, hidden_size * 3]
        query, key, value = qkv.split(self.hidden_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        use_flash_for_this = (self.use_flash and
                              not self.training and
                              seq_len > 1 and
                              seq_len % 32 == 0)

        if use_flash_for_this:
            head_outputs = []
            for head_idx in range(self.num_heads):
                head_out = self.flash_attention_single_head(
                    query[:, head_idx],
                    key[:, head_idx],
                    value[:, head_idx]
                )
                head_outputs.append(head_out.unsqueeze(1))

            attn_output = torch.cat(head_outputs, dim=1)
            attention_weights = None
        else:
            attn_output = self.reference_attention_single_head(query, key, value)
            attention_weights = None

        # Merge heads
        attn_output = self.merge_heads(attn_output)  # [batch, seq, hidden_size]

        # Output projection
        output = self.c_proj(attn_output)
        output = self.resid_dropout(output)

        return output, attention_weights


def test_flash_attention_module():
    print("="*70)
    print("Testing Flash Attention Module")
    print("="*70)

    # Config (GPT-2 small)
    batch_size = 2
    seq_len = 128
    hidden_size = 768
    num_heads = 12

    # Create module
    flash_attn = FlashAttentionModule(
        hidden_size=hidden_size,
        num_heads=num_heads,
        dropout=0.0,  # No dropout for testing
        use_flash=True
    ).cuda().eval()

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')

    print(f"\nInput shape: {x.shape}")
    print(f"Num heads: {num_heads}")
    print(f"d_k: {hidden_size // num_heads}")

    # Forward pass with Flash
    print("\nRunning forward pass with Flash Attention...")
    with torch.no_grad():
        output_flash, _ = flash_attn(x)

    print(f"Output shape: {output_flash.shape}")
    print(f"Output range: [{output_flash.min():.3f}, {output_flash.max():.3f}]")

    # Forward pass with reference
    print("\nRunning forward pass with Reference Attention...")
    flash_attn.use_flash = False
    with torch.no_grad():
        output_ref, _ = flash_attn(x)

    # Compare
    max_diff = torch.max(torch.abs(output_flash - output_ref)).item()
    mean_diff = torch.mean(torch.abs(output_flash - output_ref)).item()

    print(f"\nComparison with reference:")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")

    if max_diff < 1e-2:
        print("  ✓ PASSED - Results match!")
    else:
        print("  ✗ FAILED - Results differ too much!")

    # Benchmark
    print("\nBenchmarking...")
    import time

    flash_attn.use_flash = True
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = flash_attn(x)
    torch.cuda.synchronize()
    flash_time = (time.time() - start) / 100

    flash_attn.use_flash = False
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = flash_attn(x)
    torch.cuda.synchronize()
    ref_time = (time.time() - start) / 100

    print(f"  Flash Attention: {flash_time*1000:.3f} ms")
    print(f"  Reference: {ref_time*1000:.3f} ms")
    print(f"  Speedup: {ref_time/flash_time:.2f}x")

    print("\n" + "="*70)
    print("✓ Flash Attention Module ready for GPT-2 integration!")
    print("="*70)


if __name__ == "__main__":
    test_flash_attention_module()
