# inference.py
import torch
import os
from gpt import GPT, GPTConfig


def load_and_generate(
    queries: list[str],
    checkpoint_path: str = "gpt_checkpoint.pt",
    config: GPTConfig = None,
    temperature: float = 1,
    top_k: int = 0
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)

    if config is None:
        config = GPTConfig(
            N=6,
            d_model=256,
            token_dim=200,
            vocab_size=4000,
            num_heads=8,
            checkpoint_path=checkpoint_path
        )

    print(f"Loading model from {checkpoint_path}")
    print(
        f"Trained for {checkpoint['epoch'] + 1} epochs | Last avg loss: {checkpoint['avg_loss']:.4f}")
    print("-" * 50)

    gpt = GPT(config=config)

    tokenizer_ready = gpt.load_tokenizer()
    if not tokenizer_ready:
        raise RuntimeError(
            f"No tokenizer found at {config.tokenizer_path}. "
            "Run training first to generate the tokenizer file."
        )
    gpt.load_checkpoint()
    print(f"Vocab size: {gpt.tokenizer.current_size} tokens")
    print("-" * 50)

    results = gpt.generate(queries)
    return results


if __name__ == "__main__":
    queries = [
        "Once upon a time"
    ]

    load_and_generate(
        queries=queries,
        checkpoint_path="gpt_checkpoint.pt",
        temperature=0.1,
        top_k=0
    )
