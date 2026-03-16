"""Original grokking transformer from Power et al. (2022).

Faithfully matches the grok repo implementation:
  - Post-norm (Vaswani style)
  - ReLU activation in FFN
  - No bias in any linear layer
  - Sinusoidal positional encoding (fixed, not learned)
  - Causal attention masking
  - Dropout only on FFN output (not attention)
  - Output over full vocab, sliced to p for classification
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_position_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, block_size: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    """Post-norm block matching the grok repo."""

    def __init__(self, d_model: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_head, block_size)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )
        self.ffn_drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn_drop(self.ffn(x)))
        return x


class OriginalGrokTransformer(nn.Module):
    """Exact reproduction of the Power et al. (2022) transformer.

    Args:
        p: Prime modulus.
        d_model: Embedding / hidden dimension (128 in paper).
        n_head: Number of attention heads (4 in paper).
        n_layer: Number of transformer blocks (2 in paper).
        dropout: Dropout rate on FFN output only.
    """

    def __init__(self, p: int, d_model: int = 128, n_head: int = 4,
                 n_layer: int = 2, dropout: float = 0.0):
        super().__init__()
        self.p = p
        self.vocab_size = p + 2  # numbers + op + equals
        block_size = 4

        self.tok_emb = nn.Embedding(self.vocab_size, d_model)
        self.register_buffer("pos_enc", sinusoidal_position_encoding(block_size, d_model))
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        self.head = nn.Linear(d_model, self.vocab_size, bias=False)

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        n_emb = self.tok_emb.weight.numel()
        self.n_non_emb_params = n_params - n_emb

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, tokens):
        """
        Args:
            tokens: (B, 4) long tensor — [a, op_token, b, eq_token]
        Returns:
            logits: (B, p) — prediction over number tokens at last position
        """
        B, T = tokens.size()
        x = self.tok_emb(tokens) + self.pos_enc[:T]
        x = self.blocks(x)
        full_logits = self.head(x[:, -1, :])
        return full_logits[:, :self.p]
