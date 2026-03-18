import torch.nn as nn
import torch
from math import sin, cos
from datasets import load_dataset

from bpe import BPE


class EmbeddingLayer(nn.Module):
    def __init__(self, d_model: int = 256, token_dim: int = 512, add_positional: bool = True):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=token_dim, embedding_dim=d_model)
        self.d_model = d_model
        self.token_dim = token_dim
        self.add_positional = add_positional

    def forward(self, x):
        if not isinstance(x, list) and not isinstance(x, torch.Tensor):
            raise AttributeError("input can only be list or Tensor")
        if isinstance(x, list):
            x = torch.tensor(x)
        print("Input shape :", x.shape)
        print("Expected shape : ", self.embedding.num_embeddings)
        embeddings = self.embedding(x)

        if self.add_positional:

            _, seq_len, d_model = embeddings.shape

            positions = torch.arange(seq_len).unsqueeze(1)

            div_term = torch.exp(
                torch.arange(0, d_model, 2) *
                (-torch.log(torch.tensor(10000.0)) / d_model)
            )

            pe = torch.zeros(seq_len, d_model)

            pe[:, 0::2] = torch.sin(positions * div_term)
            pe[:, 1::2] = torch.cos(positions * div_term)

            embeddings = embeddings + pe.unsqueeze(0)

        return embeddings


def main():
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    N = 100  # number of documents
    corpus = "\n".join(dataset["train"]["text"][:N])
    print("corpus contains : ", len(corpus.split(" ")), " words")
    tokenizer = BPE(vocab_size=300)
    tokenizer.train(corpus)
    queries = ["Hello man", "What a beautiful day i can't believe it"]
    tokens = tokenizer.encode(queries)
    embedding = EmbeddingLayer()
    embedding.forward(tokens["input_ids"])


if __name__ == "__main__":
    main()
