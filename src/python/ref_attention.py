# attempt to make my own multi-head Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ReferenceAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0
    ):
        super().__init__()

        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_k = hidden_size // num_heads
        self.dropout_prob = dropout

        # initialize projection layers for Q, K, V
        self.c_attn = nn.Linear(hidden_size, hidden_size * 3)

        # initialize the output projection
        self.c_proj = nn.Linear(hidden_size, hidden_size)

        # dropout layers for regularization
        # randomly drops some attention links to prevent over-depedency
        self.attn_dropout = nn.Dropout(dropout)
        # randomly drops some output features to prevent overfitting
        self.resid_dropout = nn.Dropout(dropout)

    # split hidden dimension into multiple heads
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        x = x.transpose(1, 2)

        return x

    # merge multiple heads back into hidden dimension
    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, d_k = x.size()

        x = x.transpose(1, 2)
        x = x.contiguous().view(batch_size, seq_len, self.hidden_size)

        return x

    def create_causal_mask(
        self,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )

        return mask

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # compute the attention logits
        scores = torch.matmul(query, key.transpose(-2, -1))

        # scaling
        scores = scores / math.sqrt(self.d_k)

        # check if mask is specified, but it will be better if specified
        if mask is not None:
            scores = scores + mask

        # apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # apply dropout
        attention_weights = self.attn_dropout(attention_weights)

        # multiply attention weights with Value
        output = torch.matmul(attention_weights, value)

        if return_attention_weights:
            return output, attention_weights

        return output, None

    def forward(
        self,
        x: torch.Tensor,
        use_causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        # project to Q, K, V
        qkv = self.c_attn(x)

        # split Q, K, V
        query, key, value = qkv.split(self.hidden_size, dim=2)

        # split heads
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        mask = None
        if use_causal_mask:
            mask = self.create_causal_mask(seq_len, x.device)

        attn_output, attention_weights = self.attention(
            query, key, value, mask, return_attention_weights
        )

        attn_output = self.merge_heads(attn_output)

        output = self.c_proj(attn_output)
        output = self.resid_dropout(output)

        return output, attention_weights


def load_gpt2_attention_weights(
    model_name: str = 'gpt2',
    layer_idx: int = 0
) -> ReferenceAttention:
    from transformers import GPT2LMHeadModel

    # load pretrained GPT-2
    gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
    gpt2_attn = gpt2.transformer.h[layer_idx].attn

    # config
    config = gpt2.config
    hidden_size = config.n_embd
    num_heads = config.n_head

    # create our reference model
    ref_attn = ReferenceAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        dropout=0.0  # no dropout for testing
    )

    # GPT-2 uses Conv1D which stores weights as (in_features, out_features)
    # PyTorch Linear expects (out_features, in_features)
    # so we need to transpose

    with torch.no_grad():
        # c_attn: transpose from (768, 2304) to (2304, 768)
        ref_attn.c_attn.weight.copy_(gpt2_attn.c_attn.weight.t())
        ref_attn.c_attn.bias.copy_(gpt2_attn.c_attn.bias)

        # c_proj: transpose from (768, 768) to (768, 768)
        ref_attn.c_proj.weight.copy_(gpt2_attn.c_proj.weight.t())
        ref_attn.c_proj.bias.copy_(gpt2_attn.c_proj.bias)

    print(f"Loaded GPT-2 attention weights from layer {layer_idx}")
    print(f"Hidden size: {hidden_size}")
    print(f"Num heads: {num_heads}")
    print(f"d_k: {hidden_size // num_heads}")
    print(f"c_attn weight shape: {ref_attn.c_attn.weight.shape}")
    print(f"c_proj weight shape: {ref_attn.c_proj.weight.shape}")

    return ref_attn
