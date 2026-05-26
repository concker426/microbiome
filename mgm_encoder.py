#!/usr/bin/env python3
"""
MGM-style Transformer Encoder for microbiome genus sequences.

Architecture:
  token_ids → nn.Embedding(vocab=1226, dim=768) → + positional encoding
  → TransformerEncoder × 6 layers (causal mask, 8 heads, 768)
  → attention pooling → (768,) output vector

Two variants:
  - MGMForPretrain: with LM head for next-genus prediction
  - MGMEncoder: without LM head, with pooling for downstream tasks

Based on MGM paper: "pre-trained by predicting the next genera in a sample.
Genera are ordered by abundance in each sample."
"""
import json
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GenusEmbedding(nn.Module):
    """Token embedding + learned positional encoding for genus tokens."""
    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (batch, seq_len) → (batch, seq_len, embed_dim)"""
        batch_size, seq_len = token_ids.shape
        token_emb = self.token_embed(token_ids)
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(positions)
        return self.dropout(token_emb + pos_emb)


class AttentionPooling(nn.Module):
    """Learnable attention-weighted pooling over sequence dimension."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn_vec = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (batch, seq_len, embed_dim), mask: (batch, seq_len) bool → (batch, embed_dim)"""
        # Simple attention: score = tanh(x · v) for each position
        scores = torch.tanh(self.linear(x)).squeeze(-1)  # (batch, seq_len)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
        return (x * weights.unsqueeze(-1)).sum(dim=1)  # (batch, embed_dim)


class TransformerBlock(nn.Module):
    """Single Transformer decoder block with causal masking."""
    def __init__(self, embed_dim: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual.
        # Create causal mask in same dtype as input (handles bf16 vs fp32).
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        # Convert key_padding_mask to same dtype to avoid warning
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(x.dtype)
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm,
                                attn_mask=causal_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)
        x = x + attn_out

        # FFN with residual
        x_norm = self.ln2(x)
        x = x + self.ffn(x_norm)
        return x


class MGMEncoder(nn.Module):
    """
    MGM-style Transformer encoder for genus token sequences.
    Output: pooled representation (batch, embed_dim) for downstream tasks.
    """
    def __init__(self, vocab_size: int = 1226, embed_dim: int = 768,
                 n_layers: int = 6, n_heads: int = 8, ffn_dim: int = 2048,
                 max_seq_len: int = 512, dropout: float = 0.2):
        super().__init__()
        self.embedding = GenusEmbedding(vocab_size, embed_dim, max_seq_len, dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.pooling = AttentionPooling(embed_dim)
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

    def forward(self, token_ids: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        token_ids: (batch, seq_len) — genus token IDs
        mask: (batch, seq_len) — True = valid token, False = PAD
        Returns: (batch, embed_dim) — pooled representation
        """
        seq_len = token_ids.shape[1]
        x = self.embedding(token_ids)  # (batch, seq_len, embed_dim)

        key_padding_mask = None if mask is None else ~mask  # True = mask out

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        return self.pooling(x, mask)


class MGMForPretrain(nn.Module):
    """
    MGM encoder + LM head for next-genus prediction pre-training.
    """
    def __init__(self, vocab_size: int = 1226, embed_dim: int = 768,
                 n_layers: int = 6, n_heads: int = 8, ffn_dim: int = 2048,
                 max_seq_len: int = 512, dropout: float = 0.2):
        super().__init__()
        self.encoder = MGMEncoder(vocab_size, embed_dim, n_layers, n_heads,
                                   ffn_dim, max_seq_len, dropout)
        # LM head with weight tying
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.encoder.embedding.token_embed.weight  # weight tying
        self.vocab_size = vocab_size

    def forward(self, token_ids: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        token_ids: (batch, seq_len)
        mask: (batch, seq_len) — True = valid
        Returns: (batch, seq_len, vocab_size) — logits
        """
        seq_len = token_ids.shape[1]
        x = self.encoder.embedding(token_ids)
        key_padding_mask = None if mask is None else ~mask

        for block in self.encoder.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        return self.lm_head(x)

    def compute_loss(self, logits: torch.Tensor,
                     token_ids: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
        """
        Compute next-genus prediction loss.
        Predict token[i+1] from position i.
        """
        shift_logits = logits[:, :-1, :].contiguous()  # (B, L-1, V)
        shift_labels = token_ids[:, 1:].contiguous()    # (B, L-1)
        shift_mask = mask[:, 1:].contiguous()            # (B, L-1)

        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_logits.shape[:2])

        loss = (loss * shift_mask).sum() / shift_mask.sum().clamp(min=1)
        return loss

    @torch.no_grad()
    def compute_metrics(self, logits: torch.Tensor,
                        token_ids: torch.Tensor,
                        mask: torch.Tensor) -> dict:
        """Compute perplexity and accuracy."""
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = token_ids[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_logits.shape[:2])
        masked_loss = (loss * shift_mask).sum() / shift_mask.sum().clamp(min=1)
        perplexity = torch.exp(masked_loss)

        preds = shift_logits.argmax(dim=-1)
        correct = (preds == shift_labels) & shift_mask
        accuracy = correct.sum().float() / shift_mask.sum().clamp(min=1)

        return {
            "perplexity": perplexity.item(),
            "accuracy": accuracy.item(),
            "loss": masked_loss.item(),
        }


def create_model(vocab_path: str = None, **kwargs) -> MGMForPretrain:
    """Create model, optionally loading vocab info from a vocab JSON file.

    Supports two vocab formats:
      - Old: {"vocab_size": N, "max_seq_len": L, "n_genera": G}
      - New: {"<PAD>": 0, "<UNK>": 1, ...}  (flat dict)
    """
    if vocab_path and os.path.exists(vocab_path):
        with open(vocab_path) as f:
            vocab = json.load(f)
        if "vocab_size" in vocab:
            # Old format
            kwargs.setdefault("vocab_size", vocab["vocab_size"])
            kwargs.setdefault("max_seq_len", vocab["max_seq_len"])
        else:
            # Flat dict format
            kwargs.setdefault("vocab_size", len(vocab))
            # max_seq_len must come from caller or default
    return MGMForPretrain(**kwargs)
