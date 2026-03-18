import torch
from torch import Tensor
import torch.nn as nn
from datasets import load_dataset
from bpe import BPE
from encoder import Encoder
from transformer_block import DecoderTransformerBlock


class Decoder(nn.Module):
    def __init__(self, N: int, d_model: int, num_heads: int, token_dim: int):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.num_heads = num_heads
        self.token_dim = token_dim
        self.nn = nn.Linear(in_features=d_model, out_features=self.token_dim)
        self.softmax = nn.Softmax
        # print("Initiliazing decoder with token dim : ", token_dim)
        self.embedding = nn.Embedding(
            num_embeddings=token_dim, embedding_dim=d_model)
        self.blocks: list[DecoderTransformerBlock] = []
        for _ in range(N):
            self.blocks.append(DecoderTransformerBlock(
                d_model=d_model, num_heads=num_heads, token_dim=token_dim))

    def forward(self, decoder_tokens: dict, encoder_output: Tensor, encoder_tokens: dict = None):
        input_ids, _ = decoder_tokens["input_ids"], decoder_tokens["attention_mask"]
        embeddings = self.embedding(input_ids)
        self.previous_output: Tensor
        for i in range(self.N):
            if i >= 1:
                block_output = self.blocks[i].forward(
                    embeddings=self.previous_output, tokens=decoder_tokens, K=encoder_output, V=encoder_output, encoded_tokens=encoder_tokens)
            else:
                block_output = self.blocks[i].forward(
                    embeddings=embeddings, tokens=decoder_tokens, K=encoder_output, V=encoder_output, encoded_tokens=encoder_tokens)
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
    tokens: dict = tokenizer.encode(queries)
    # ##print(tokens)
    tokens = {
        "input_ids": torch.tensor(tokens["input_ids"]),
        "attention_mask": torch.tensor(tokens["attention_mask"])
    }
    # Encoder forward
    decoder = Decoder(N=6, d_model=1024, num_heads=8, token_dim=512)
    output: Tensor = decoder(tokens)
    # print("Final encoder output : ", output.shape)


if __name__ == "__main__":
    main()
