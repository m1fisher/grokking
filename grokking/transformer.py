"""Small decoder-only transformer for grokking experiments.

Matches the architecture from Power et al. (2022):
  - 2 layers, width 128, 4 attention heads
  - Causal attention masking
  - Post-norm (standard Vaswani) — NOT pre-norm
  - Loss/accuracy only on the answer token
  - ReLU activation (standard Vaswani)

Input sequence: [a, op, b, =]  (4 tokens)
Predict: c = a op b (mod p)

Vocabulary: 0..p-1 (numbers), p (operation token), p+1 (equals token)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class TransformerBlock(nn.Module):
    """Post-norm transformer block (standard Vaswani style)."""

    def __init__(self, d_model: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_head, block_size, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.mlp(x))
        return x


class DecoderTransformer(nn.Module):
    """Small decoder-only transformer for modular arithmetic.

    Post-norm architecture matching the original Vaswani (2017) / Power et al. (2022).

    Args:
        p: Prime modulus (determines vocab and output size).
        d_model: Embedding / hidden dimension.
        n_head: Number of attention heads.
        n_layer: Number of transformer blocks.
        dropout: Dropout rate.
    """

    def __init__(self, p: int, d_model: int = 128, n_head: int = 4,
                 n_layer: int = 2, dropout: float = 0.0):
        super().__init__()
        self.p = p
        vocab_size = p + 2  # numbers + op + equals
        block_size = 4      # [a, op, b, =]

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        # No final LayerNorm in post-norm (each block already ends with LN)
        self.head = nn.Linear(d_model, p, bias=False)

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        n_emb = self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        self.n_non_emb_params = n_params - n_emb

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
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
            logits: (B, p) — prediction for the answer token
        """
        B, T = tokens.size()
        pos = torch.arange(T, device=tokens.device)
        x = self.drop(self.tok_emb(tokens) + self.pos_emb(pos))
        x = self.blocks(x)
        logits = self.head(x[:, -1, :])
        return logits
