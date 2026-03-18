import torch
import torch.nn as nn
from datasets import load_dataset
from bpe import BPE
from decoder import Decoder
from encoder import Encoder
from torch import Tensor


class Transformer(nn.Module):
    def __init__(self, N: int, d_model: int, token_dim: int, num_heads: int, vocab_size: int):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.encoder = Encoder(N, d_model, num_heads, token_dim)
        self.decoder = Decoder(N, d_model, num_heads, token_dim-1)
        self.linear = nn.Linear(in_features=d_model, out_features=vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.tokenizer = BPE(vocab_size=vocab_size, max_len=token_dim)

    def train_tokenizer(self, corpus: str):
        print("corpus contains : ", len(corpus.split(" ")), " words")
        self.tokenizer.train(corpus)

    def encode_queries(self, queries: list[str]):
        print("Encoding Queries.")
        tokens = self.tokenizer.encode(queries)
        self.tokens = tokens
        return tokens

    def forward(self, queries, train: bool = False):
        if not hasattr(self, "tokens"):
            print("Encoding Queries at the start of forward.")
            tokens = self.tokenizer.encode(queries)
        else:
            print("Using saved tokens.")
            tokens = self.tokens
        input_ids = torch.tensor(tokens["input_ids"])
        attention_mask = torch.tensor(tokens["attention_mask"])
        decoder_input_ids = input_ids[:, :-1]
        decoder_attention_mask = attention_mask[:, :-1]
        encoder_tokens = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        decoder_tokens = {
            "input_ids": decoder_input_ids,
            "attention_mask": decoder_attention_mask
        }
        # print("decoder attention mask shape : ", decoder_attention_mask.shape)
        labels = input_ids[:, 1:]
        encoder_output = self.encoder(encoder_tokens)
        decoder_output = self.decoder(
            decoder_tokens, encoder_output=encoder_output, encoder_tokens=encoder_tokens)
        x: Tensor = self.linear(decoder_output)
        # print("shape of final linear : ", x.shape)
        if train == False:
            x: Tensor = self.softmax(x)
        return x


def main():
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    N = 100
    corpus = "\n".join(dataset["train"]["text"][:N])
    transformer = Transformer(
        N=6, d_model=1024, token_dim=512, num_heads=8, vocab_size=500)
    transformer.train_tokenizer(corpus=corpus)
    queries = ["Hello man", "What a beautiful day i can't believe it"]
    output: Tensor = transformer.forward(queries)
    # print("output shape : ", output.shape)
    # print("Output : ", output)


if __name__ == "__main__":
    main()
