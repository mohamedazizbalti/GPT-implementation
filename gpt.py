# gpt.py
from enum import Enum
import torch
import torch.nn as nn
from attention import MultiHeadAttention
from bpe import BPE, build_reverse_vocab
from transformer_block import PositionwiseFeedForward
from datasets import load_dataset
from dataclasses import dataclass
import os


@dataclass
class GPTConfig:
    N: int = 12
    d_model: int = 1024
    token_dim: int = 512
    vocab_size: int = 500
    num_heads: int = 8
    epochs: int = 30
    batch_size: int = 8
    lr: float = 1e-4
    min_frequency: int = 2
    checkpoint_path: str = "gpt_checkpoint.pt"

    @property
    def tokenizer_path(self):
        return self.checkpoint_path.replace(".pt", "_tokenizer.pt")


class GPTTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, causal=True)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, embeddings: torch.Tensor, tokens: dict):
        x1 = self.norm1(
            embeddings + self.attention(embeddings=embeddings, tokens=tokens))
        x2 = self.norm2(x1 + self.feed_forward(x1))
        return x2


class TASK(Enum):
    CLASSIFICATION = 1
    ENTAILMENT = 2
    SIMILARITY = 3
    MULTIPLE_CHOICE = 4
    UNSUPERVISED = 5


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.N = config.N
        self.token_dim = config.token_dim
        self.vocab_size = config.vocab_size
        self.tokenizer = BPE(vocab_size=config.vocab_size,
                             max_len=config.token_dim)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_embedding = nn.Embedding(
            config.token_dim, config.d_model)
        self.linear = nn.Linear(
            in_features=config.d_model, out_features=config.vocab_size)
        self.blocks = nn.ModuleList([
            GPTTransformerBlock(d_model=self.d_model,
                                num_heads=config.num_heads)
            for _ in range(self.N)
        ])

    # ── Tokenizer ──────────────────────────────────────────────────────────────

    def train_tokenizer(self, corpus: str):
        self.tokenizer.train(corpus, min_frequency=self.config.min_frequency)
        self.save_tokenizer()

    def save_tokenizer(self):
        torch.save({
            "tokenizer_vocab": self.tokenizer.vocab,
            "tokenizer_rules": self.tokenizer.rules,
            "tokenizer_version": "v3"
        }, self.config.tokenizer_path)
        print(f"Tokenizer saved to {self.config.tokenizer_path}")

    def load_tokenizer(self) -> bool:
        if not os.path.exists(self.config.tokenizer_path):
            return False
        data = torch.load(self.config.tokenizer_path, weights_only=False)
        if data.get("tokenizer_version") != "v3":
            print("Stale tokenizer — deleting.")
            os.remove(self.config.tokenizer_path)
            return False
        self.tokenizer.vocab = data["tokenizer_vocab"]
        self.tokenizer.rules = data["tokenizer_rules"]
        self.tokenizer.current_size = len(self.tokenizer.vocab)
        self.tokenizer.reverse_vocab = build_reverse_vocab(
            self.tokenizer.vocab)
        self.tokenizer._build_trie()
        print(f"Tokenizer loaded from {self.config.tokenizer_path}")
        return True

    # ── Checkpoint ─────────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, optimizer, scheduler, avg_loss: float):
        torch.save({
            "epoch": epoch,
            "model_state": self.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "avg_loss": avg_loss,
        }, self.config.checkpoint_path)
        print(f"Checkpoint saved to {self.config.checkpoint_path}")

    def load_checkpoint(self, optimizer=None) -> tuple[int, dict | None]:
        if not os.path.exists(self.config.checkpoint_path):
            return 0, None
        checkpoint = torch.load(
            self.config.checkpoint_path, weights_only=False)
        self.load_state_dict(checkpoint["model_state"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            # reset lr to config value regardless of saved optimizer state
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.config.lr
        start_epoch = checkpoint["epoch"] + 1
        scheduler_state = checkpoint.get("scheduler_state")
        print(
            f"Model loaded | epoch {start_epoch} | loss {checkpoint['avg_loss']:.4f}")
        return start_epoch, scheduler_state

    # ── Model ──────────────────────────────────────────────────────────────────

    def encode_corpus(self, corpus: str):
        return self.tokenizer.encode(corpus=corpus, padding=False, truncating=False)

    def forward(self, tokens: dict):
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(
            0).expand(batch_size, seq_len)
        x = self.embedding(input_ids) + self.positional_embedding(positions)
        for block in self.blocks:
            x = block(x, {"input_ids": input_ids,
                      "attention_mask": attention_mask})
        return self.linear(x)

    def next_token(self, tokens: dict, temperature: float = 0.8, top_k: int = 40):
        logits = self.forward(tokens)
        next_token_logits = logits[:, -1, :] / temperature
        top_k_logits, top_k_indices = torch.topk(
            next_token_logits, top_k, dim=-1)
        probs = torch.softmax(top_k_logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
        return top_k_indices[torch.arange(top_k_indices.shape[0]), sampled]

    def generate(self, queries: list[str], temperature: float = 0.8, top_k: int = 40):
        self.eval()
        results = []
        with torch.no_grad():
            for query in queries:
                tokens = self.tokenizer.encode(
                    [query], padding=False, truncating=True)
                input_ids = tokens["input_ids"][:, :-1]
                eos_id = self.tokenizer.vocab.get("<EOS>")
                bos_id = self.tokenizer.vocab.get("<BOS>")
                pad_id = self.tokenizer.vocab.get("<PAD>")
                generated = input_ids.clone()
                for _ in range(self.token_dim):
                    current_input = generated[:, -self.token_dim:]
                    attention_mask = torch.ones(
                        current_input.shape, dtype=torch.long
                    )
                    next_token_idx = self.next_token(
                        {"input_ids": current_input,
                            "attention_mask": attention_mask},
                        temperature=temperature,
                        top_k=top_k
                    )
                    if next_token_idx.item() == eos_id:
                        break
                    generated = torch.cat(
                        [generated, next_token_idx.unsqueeze(1)], dim=1)
                seq_ids = generated[0].tolist()
                if eos_id in seq_ids:
                    seq_ids = seq_ids[:seq_ids.index(eos_id)]
                seq_ids = [t for t in seq_ids if t not in (pad_id, bos_id)]
                decoded = self.tokenizer.decode(seq_ids)
                results.append(decoded)
                print("Query:", query)
                print("Generated:", decoded)
                print("------")
        self.train()
        return results

    def check_gradients(self):
        print("\n── Gradient Check ──────────────────────────────")
        no_grad = []
        has_grad = []
        for name, param in self.named_parameters():
            if param.grad is None:
                no_grad.append(name)
            elif param.grad.abs().max().item() == 0:
                no_grad.append(f"{name} (zero grad)")
            else:
                has_grad.append(name)
        print(f"✓ {len(has_grad)} params have gradients")
        if no_grad:
            print(f"✗ {len(no_grad)} params with NO gradient:")
            for n in no_grad:
                print(f"    {n}")
        else:
            print("✓ All params receiving gradients")
        print("────────────────────────────────────────────────\n")

    def train_model(self, corpus: str, task: TASK = TASK.UNSUPERVISED):
        if task != TASK.UNSUPERVISED:
            return
        loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.vocab.get("<PAD>"))
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.lr, weight_decay=0.1)
        start_epoch, scheduler_state = self.load_checkpoint(
            optimizer=optimizer)
        tokens = self.encode_corpus(corpus=corpus)
        input_ids, attention_mask = tokens["input_ids"], tokens["attention_mask"]
        decoder_input_ids = input_ids[:, :-1]
        decoder_attention_mask = attention_mask[:, :-1]
        seq_len = decoder_input_ids.shape[1]
        trimmed_len = (seq_len // self.token_dim) * self.token_dim
        decoder_input_ids = decoder_input_ids[:,
                                              :trimmed_len].reshape(-1, self.token_dim)
        decoder_attention_mask = decoder_attention_mask[:,
                                                        :trimmed_len].reshape(-1, self.token_dim)
        labels = input_ids[:, 1:][:, :trimmed_len].reshape(-1, self.token_dim)
        num_chunks = decoder_input_ids.shape[0]
        steps_per_epoch = -(-num_chunks // self.config.batch_size)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs - start_epoch,
            eta_min=1e-6
        )
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        batch_size = self.config.batch_size
        for epoch in range(start_epoch, self.config.epochs):
            indices = torch.randperm(num_chunks)
            decoder_input_ids = decoder_input_ids[indices]
            decoder_attention_mask = decoder_attention_mask[indices]
            labels = labels[indices]
            print(f"Epoch : {epoch + 1}/{self.config.epochs}")
            epoch_loss = 0.0
            num_batches = 0
            for batch_start in range(0, num_chunks, batch_size):
                batch_end = min(batch_start + batch_size, num_chunks)
                input_batch = decoder_input_ids[batch_start:batch_end]
                mask_batch = decoder_attention_mask[batch_start:batch_end]
                label_batch = labels[batch_start:batch_end]
                optimizer.zero_grad()
                logits = self.forward(
                    {"input_ids": input_batch, "attention_mask": mask_batch})
                logits = logits.reshape(-1, self.vocab_size)
                label_batch = label_batch.reshape(-1)
                batch_loss = loss_fn(logits, label_batch)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += batch_loss.item()
                num_batches += 1
                print(
                    f"  Batch {batch_start // batch_size + 1}/{steps_per_epoch} Loss: {batch_loss.item():.4f}")
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")
            scheduler.step()
            self.save_checkpoint(epoch, optimizer, scheduler, avg_loss)


if __name__ == "__main__":
    N = 10000
    dataset = load_dataset("roneneldan/TinyStories")
    corpus = "\n".join(dataset["train"]["text"][:N])
    config = GPTConfig(N=6, d_model=512, token_dim=200, vocab_size=4000,
                       num_heads=8, epochs=30, batch_size=8, lr=3e-5,
                       min_frequency=10, checkpoint_path="gpt_checkpoint.pt")
    gpt = GPT(config=config)
    queries = ["Once upon a time there was a little girl"]

    tokenizer_ready = gpt.load_tokenizer()
    if not tokenizer_ready:
        gpt.train_tokenizer(corpus=corpus)

    model_ready = os.path.exists(config.checkpoint_path)
    if model_ready:
        print("Loading from pre trained")
        gpt.load_checkpoint()
        gpt.train_model(corpus=corpus)
    else:
        gpt.train_model(corpus=corpus)

    gpt.generate(queries=queries)
