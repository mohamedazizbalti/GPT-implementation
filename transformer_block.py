import torch
import torch.nn as nn
from bpe import BPE
from attention import MultiHeadAttention
from embedding import EmbeddingLayer
from datasets import load_dataset
from torch import Tensor


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out


class EncoderTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, embeddings: torch.Tensor, tokens: dict):
        attention = self.attention(
            embeddings=embeddings, tokens=tokens)
        x1 = embeddings + attention
        x1 = self.norm1(x1)
        x2 = self.feed_forward(x1)
        x3 = x2 + x1
        x3 = self.norm2(x3)
        # print("Output of Transformer block : ", x3.shape)
        return x3


class DecoderTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, token_dim: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.token_dim = token_dim
        self.attention1 = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, causal=True, cross=False)
        self.attention2 = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, causal=False, cross=True)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, embeddings: torch.Tensor, tokens: dict, K: Tensor, V: Tensor, encoded_tokens: dict = None):
        attention: Tensor = self.attention1(
            embeddings=embeddings, tokens=tokens)
        x1: Tensor = embeddings + attention
        x1 = self.norm1(x1)
        attention2: Tensor = self.attention2.forward(
            embeddings=x1, tokens=tokens, K=K, V=V, encoder_tokens=encoded_tokens)
        x2: Tensor = attention2 + x1
        x3: Tensor = self.norm2(x2)
        x4: Tensor = self.feed_forward(x3)
        x5: Tensor = x3 + x4
        x5: Tensor = self.norm3(x5)
        # print("Output of Transformer block : ", x5.shape)
        return x3
