import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from datasets import load_dataset

from bpe import BPE
from embedding import EmbeddingLayer


def is_whole(n):
    return n % 1 == 0


class AttentionHead(nn.Module):
    def __init__(self, d_model: int = 512, d_k: int = 256, causal: bool = False, cross: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_k = int(d_k)
        self.KW = nn.Linear(in_features=self.d_model, out_features=self.d_k)
        self.QW = nn.Linear(in_features=self.d_model, out_features=self.d_k)
        self.VW = nn.Linear(in_features=self.d_model, out_features=self.d_k)
        self.causal = causal
        self.cross = cross

    def forward(self, x: Tensor, tokens: dict, K: Tensor = None, V: Tensor = None, encoder_tokens: dict = None):
        Q = self.QW(x)
        if not self.cross:
            K = self.KW(x)
            V = self.VW(x)
        else:
            if K is None or V is None:
                raise ValueError("Cross Attention with None K & V.")
            K = self.KW(K)
            V = self.VW(V)

        scores: Tensor = torch.matmul(
            Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        batch_size, dec_seq_len, _ = Q.shape
        enc_seq_len = K.shape[1]

        if self.cross:
            mask = encoder_tokens["attention_mask"]
            if not isinstance(mask, Tensor):
                mask = torch.tensor(mask)
            mask = mask.bool().unsqueeze(1).expand(batch_size, dec_seq_len, enc_seq_len)
        else:
            pad_mask = tokens["attention_mask"]
            if not isinstance(pad_mask, Tensor):
                pad_mask = torch.tensor(pad_mask)
            pad_mask = pad_mask.bool().unsqueeze(1).expand(
                batch_size, dec_seq_len, enc_seq_len)
            if self.causal:
                causal_mask = torch.tril(torch.ones(
                    (dec_seq_len, dec_seq_len), dtype=torch.bool, device=x.device))
                causal_mask = causal_mask.unsqueeze(
                    0).expand(batch_size, -1, -1)
                mask = causal_mask & pad_mask
            else:
                mask = pad_mask

        scores = scores.masked_fill(mask == 0, float("-inf"))
        # replace full -inf rows with 0 to avoid nan in softmax
        scores = scores.masked_fill(mask.sum(dim=-1, keepdim=True) == 0, 0.0)
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, causal: bool = False, cross: bool = False):
        super().__init__()
        if not is_whole(d_model / num_heads):
            raise ValueError(
                f"d_model {d_model} must be divisible by num_heads {num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W = nn.Linear(in_features=d_model, out_features=d_model)
        self.attention_heads = nn.ModuleList([
            AttentionHead(d_model=d_model, d_k=self.d_k,
                          causal=causal, cross=cross)
            for _ in range(num_heads)
        ])

    def forward(self, embeddings: Tensor, tokens: dict, K: Tensor = None, V: Tensor = None, encoder_tokens: dict = None):
        outputs = [
            head.forward(embeddings, tokens, K=K, V=V,
                         encoder_tokens=encoder_tokens)
            for head in self.attention_heads
        ]
        combined = torch.cat(outputs, dim=-1)
        return self.W(combined)


def main():
    # Tokenization
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    N = 100
    corpus = "\n".join(dataset["train"]["text"][:N])
    tokenizer = BPE(vocab_size=300)
    tokenizer.train(corpus)
    queries = ["Hello man", "What a beautiful day i can't believe it"]
    tokens = tokenizer.encode(queries)
    # Embedding + Positional Encoding
    embedding = EmbeddingLayer(d_model=1024)
    embeddings: Tensor = embedding(tokens["input_ids"])
    print("Embeddings shape : ", embeddings.shape)
    multihead_attention = MultiHeadAttention(
        d_model=1024, num_heads=1, causal=True)
    output: Tensor = multihead_attention.forward(
        embeddings=embeddings, tokens=tokens)
    print("Final multihead attention output : ", output.shape)


if __name__ == "__main__":
    main()
