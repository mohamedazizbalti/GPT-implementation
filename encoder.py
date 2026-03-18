import torch
import torch.nn as nn
from bpe import BPE
from attention import MultiHeadAttention
from embedding import EmbeddingLayer
from datasets import load_dataset
from torch import Tensor

from transformer_block import EncoderTransformerBlock


class Encoder(nn.Module):
    def __init__(self, N: int, d_model: int, num_heads: int, token_dim: int):
        super().__init__()
        self.N = N
        self.blocks: list[EncoderTransformerBlock] = []
        for _ in range(self.N):
            self.blocks.append(EncoderTransformerBlock(
                d_model=d_model, num_heads=num_heads))
        self.embedding = EmbeddingLayer(d_model=d_model, token_dim=token_dim)

    def forward(self, tokens: dict):
        input_ids = tokens["input_ids"]
        embeddings = self.embedding(input_ids)
        self.previous_output: Tensor
        for i in range(self.N):
            if i >= 1:
                block_output = self.blocks[i].forward(
                    embeddings=self.previous_output, tokens=tokens)
            else:
                block_output = self.blocks[i].forward(
                    embeddings=embeddings, tokens=tokens)
            self.previous_output = block_output
        return self.previous_output


def main():
    # Tokenization
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    N = 100
    corpus = "\n".join(dataset["train"]["text"][:N])
    tokenizer = BPE(vocab_size=300, max_len=512)
    tokenizer.train(corpus)
    queries = ["Hello man", "What a beautiful day i can't believe it"]
    tokens = tokenizer.encode(queries)
    # Encoder forward
    encoder = Encoder(N=6, d_model=1024, num_heads=8, token_dim=512)
    output: Tensor = encoder.forward(tokens)
    print("Final encoder output : ", output.shape)


if __name__ == "__main__":
    main()
