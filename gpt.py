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
from torch import Tensor
from utils import create_classification_dataset, load_ag_news


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
    finetune_checkpoint_path: str = "gpt_finetune_checkpoint.pt"
    finetune_lr: float = 6e-5
    finetune_weight_decay: float = 0.01
    finetune_epochs: int = 3
    num_classes: int = 4

    @property
    def tokenizer_path(self):
        return "/Users/mohamedazizbalti/sites/GPT/gpt_checkpoint_tokenizer.pt"


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
        self.cls_linear = nn.Linear(
            in_features=config.d_model, out_features=config.num_classes)
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

    def save_checkpoint(self, epoch: int, optimizer, scheduler, avg_loss: float, path: str = None):
        save_path = path or self.config.checkpoint_path
        torch.save({
            "epoch": epoch,
            "model_state": self.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "avg_loss": avg_loss,
        }, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, optimizer=None, path: str = None) -> tuple[int, dict | None]:
        load_path = path or self.config.checkpoint_path
        if not os.path.exists(load_path):
            return 0, None
        checkpoint = torch.load(load_path, weights_only=False)
        self.load_state_dict(checkpoint["model_state"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
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

    def forward(self, tokens: dict, pre_training: bool = True):
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(
            0).expand(batch_size, seq_len)
        x = self.embedding(input_ids) + self.positional_embedding(positions)
        for block in self.blocks:
            x = block(x, {"input_ids": input_ids,
                      "attention_mask": attention_mask})
        if pre_training:
            return self.linear(x)
        return x, self.linear(x)

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
                        current_input.shape, dtype=torch.long)
                    next_token_idx = self.next_token(
                        {"input_ids": current_input,
                            "attention_mask": attention_mask},
                        temperature=temperature, top_k=top_k
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

    def classify(self, texts: list[str]) -> list[str]:
        label_names = {v: k for k, v in {
            "animal": 0, "magic": 1, "emotion": 2,
            "adventure": 3, "friendship": 4
        }.items()}
        self.eval()
        with torch.no_grad():
            tokens = self.tokenizer.encode(
                texts, padding=True, truncating=True)
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            hidden, _ = self.forward(
                {"input_ids": input_ids, "attention_mask": attention_mask},
                pre_training=False
            )
            class_logits = self.cls_linear(hidden[:, -1, :])
            preds = class_logits.argmax(dim=-1).tolist()
        self.train()
        return [label_names.get(p, str(p)) for p in preds]

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

    def pre_train_model(self, corpus: str):
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
            optimizer, T_max=self.config.epochs - start_epoch, eta_min=1e-6)
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

    def finetune_model(self, dataset: tuple, task: TASK = TASK.CLASSIFICATION):
        print("Fine-Tuning the model using the task : ", task)
        if task != TASK.CLASSIFICATION:
            return

        # always load pretrained weights fresh before finetuning
        print("Loading pretrained weights for finetuning...")
        self.load_checkpoint()

        dataset, mapping = dataset
        split = int(0.8 * len(dataset))
        train_data = dataset[:split]
        val_data = dataset[split:]

        train_text = [entry["text"] for entry in train_data]
        train_labels = [entry["label"] for entry in train_data]
        val_text = [entry["text"] for entry in val_data]
        val_labels = [entry["label"] for entry in val_data]

        tokens = self.tokenizer.encode(
            corpus=train_text, padding=True, truncating=True)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        lm_input_ids = input_ids[:, :-1]
        lm_attention_mask = attention_mask[:, :-1]
        lm_labels = input_ids[:, 1:]
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
        val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)

        num_samples = len(train_text)
        steps_per_epoch = -(-num_samples // self.config.batch_size)

        loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.vocab.get("<PAD>"))
        finetune_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.finetune_lr,
            weight_decay=self.config.finetune_weight_decay
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=steps_per_epoch)

        print(
            f"Train: {len(train_text)} | Val: {len(val_text)} | Classes: {mapping}")
        batch_size = self.config.batch_size
        best_val_loss = float("inf")

        for epoch in range(self.config.finetune_epochs):
            indices = torch.randperm(num_samples)
            lm_input_ids = lm_input_ids[indices]
            lm_attention_mask = lm_attention_mask[indices]
            lm_labels = lm_labels[indices]
            train_labels_tensor = train_labels_tensor[indices]

            self.train()
            print(f"Epoch : {epoch + 1}/{self.config.finetune_epochs}")
            epoch_loss = 0.0
            epoch_lm_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0

            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                input_batch = lm_input_ids[batch_start:batch_end]
                mask_batch = lm_attention_mask[batch_start:batch_end]
                label_batch = lm_labels[batch_start:batch_end]
                class_label_batch = train_labels_tensor[batch_start:batch_end]

                optimizer.zero_grad()
                hidden, lm_logits = self.forward(
                    {"input_ids": input_batch, "attention_mask": mask_batch},
                    pre_training=False
                )
                lm_loss = loss_fn(
                    lm_logits.reshape(-1, self.vocab_size),
                    label_batch.reshape(-1)
                )
                class_logits = self.cls_linear(hidden[:, -1, :])
                class_loss = finetune_fn(class_logits, class_label_batch)
                total_loss = lm_loss + 0.3 * class_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                preds = class_logits.argmax(dim=-1)
                acc = (preds == class_label_batch).float().mean().item()
                epoch_loss += class_loss.item()
                epoch_lm_loss += lm_loss.item()
                epoch_acc += acc
                num_batches += 1
                print(f"  Batch {num_batches}/{steps_per_epoch} "
                      f"Class: {class_loss.item():.4f} | "
                      f"Acc: {acc:.2%} | "
                      f"LM: {lm_loss.item():.4f}")

            avg_class_loss = epoch_loss / num_batches
            avg_lm_loss = epoch_lm_loss / num_batches
            avg_acc = epoch_acc / num_batches
            print(f"Epoch {epoch + 1} | Class: {avg_class_loss:.4f} | "
                  f"Acc: {avg_acc:.2%} | LM: {avg_lm_loss:.4f}")

            # validation
            self.eval()
            with torch.no_grad():
                val_tokens = self.tokenizer.encode(
                    corpus=val_text, padding=True, truncating=True)
                val_input_ids = val_tokens["input_ids"][:, :-1]
                val_mask = val_tokens["attention_mask"][:, :-1]
                val_hidden, _ = self.forward(
                    {"input_ids": val_input_ids, "attention_mask": val_mask},
                    pre_training=False
                )
                val_logits = self.cls_linear(val_hidden[:, -1, :])
                val_loss = finetune_fn(val_logits, val_labels_tensor).item()
                val_preds = val_logits.argmax(dim=-1)
                val_acc = (
                    val_preds == val_labels_tensor).float().mean().item()
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(
                    epoch, optimizer, scheduler, avg_class_loss,
                    path=self.config.finetune_checkpoint_path
                )
                print(f"  ✓ Best model saved (val loss: {val_loss:.4f})")


if __name__ == "__main__":
    N = 1000
    dataset = load_dataset("roneneldan/TinyStories")
    corpus = "\n".join(dataset["train"]["text"][:N])
    config = GPTConfig(
        N=6,
        d_model=512,
        token_dim=512,
        vocab_size=4000,
        num_heads=8,
        epochs=30,
        batch_size=16,
        lr=1e-4,
        min_frequency=2,
        checkpoint_path="gpt_checkpoint.pt",
        finetune_checkpoint_path="gpt_finetune_checkpoint.pt",
        finetune_epochs=3,
        finetune_lr=1e-5
    )
    gpt = GPT(config=config)
    queries = ["Once upon a time there was a little girl"]

    tokenizer_ready = gpt.load_tokenizer()
    if not tokenizer_ready:
        gpt.train_tokenizer(corpus=corpus)

    model_ready = os.path.exists(config.checkpoint_path)
    if not model_ready:
        gpt.pre_train_model(corpus=corpus)
    else:
        print("Pretrained model found.")
        # gpt.pre_train_model(corpus=corpus)
        # gpt.finetune_model(   dataset=load_ag_news())
        pass
    gpt.load_checkpoint()
    gpt.generate(queries=queries)
    gpt.load_checkpoint(path=config.finetune_checkpoint_path)
    print(gpt.classify(["once upon a time a dog ran through the forest"]))
