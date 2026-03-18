from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Adam
from torch import Tensor
import argparse
import math
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformer import Transformer


def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]


def build_dataset_texts(dataset, split: str = "train", max_examples: int | None = None):
    ds = dataset[split]
    texts = [t for t in ds["text"] if isinstance(
        t, str) and len(t.strip()) > 0]
    if max_examples is not None:
        texts = texts[:max_examples]
    return texts


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset
    dataset = load_dataset(args.dataset, args.subset)
    texts = build_dataset_texts(
        dataset, split=args.split, max_examples=args.max_examples)
    print(
        f"Loaded {len(texts)} examples from {args.dataset}/{args.subset} ({args.split})")

    # Instantiate model
    model = Transformer(
        N=args.num_layers,
        d_model=args.d_model,
        token_dim=args.token_dim,
        num_heads=args.num_heads,
        vocab_size=args.vocab_size,
    )
    model.to(device)

    # Train tokenizer on a small corpus slice
    corpus_texts = "\n".join(texts[: args.tokenizer_train_docs])
    print(
        f"Training tokenizer on {args.tokenizer_train_docs} docs (approx {len(corpus_texts.split())} words)")
    model.train_tokenizer(corpus=corpus_texts)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    pad_token_id = model.tokenizer.vocab.get("<PAD>", 0)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction="sum")

    model.train()
    step = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        total_loss = 0.0
        total_tokens = 0

        # Simple batching: BPE.encode accepts a list of strings
        for batch_texts in chunked(texts, args.batch_size):
            tokens = model.tokenizer.encode(batch_texts)
            input_ids = torch.tensor(
                tokens["input_ids"], dtype=torch.long, device=device)
            attention_mask = torch.tensor(
                tokens["attention_mask"], dtype=torch.long, device=device)

            # prepare encoder / decoder tokens similar to Transformer.forward
            encoder_tokens = {"input_ids": input_ids,
                              "attention_mask": attention_mask}
            decoder_input_ids = input_ids[:, :-1]
            decoder_attention_mask = attention_mask[:, :-1]
            decoder_tokens = {"input_ids": decoder_input_ids,
                              "attention_mask": decoder_attention_mask}
            labels = input_ids[:, 1:]

            optimizer.zero_grad()

            # forward pass using model components directly
            encoder_output = model.encoder(encoder_tokens)
            decoder_output = model.decoder(
                decoder_tokens, encoder_output=encoder_output, encoder_tokens=encoder_tokens)
            # shape: [B, dec_len, token_dim]
            logits = model.linear(decoder_output)

            B, dec_len, vocab_logits = logits.shape

            logits_flat = logits.reshape(-1, vocab_logits)
            labels_flat = labels.reshape(-1)

            # mask out padding tokens using criterion's ignore_index
            loss_sum = criterion(logits_flat, labels_flat)
            # normalize by number of real tokens in the batch
            num_real = decoder_attention_mask.sum().item()
            if num_real == 0:
                loss = loss_sum
            else:
                loss = loss_sum / num_real

            loss.backward()
            optimizer.step()

            total_loss += loss_sum.item()
            total_tokens += max(1, num_real)
            step += 1

            if step % args.log_interval == 0:
                avg = total_loss / max(1, total_tokens)
                print(f"step={step} avg_loss_per_token={avg:.4f}")

        epoch_loss = total_loss / max(1, total_tokens)
        print(f"Epoch {epoch+1} finished. avg_loss_per_token={epoch_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), args.save_path)
    print("Saved model to", args.save_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--subset", default="wikitext-103-raw-v1")
    p.add_argument("--split", default="train")
    p.add_argument("--max-examples", type=int, default=1000)
    p.add_argument("--tokenizer-train-docs", type=int, default=100)
    p.add_argument("--vocab-size", type=int, default=500)
    p.add_argument("--token-dim", type=int, dest="token_dim", default=512)
    p.add_argument("--d-model", type=int, dest="d_model", default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, dest="num_layers", default=6)
    p.add_argument("--batch-size", type=int, dest="batch_size", default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, dest="lr", default=1e-4)
    p.add_argument("--save-path", default="transformer.pth")
    p.add_argument("--log-interval", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

QUERIES = [
    "Hello, how are you doing today?",
    "What is the weather like in Paris this week?",
    "Explain the difference between supervised and unsupervised learning.",
    "Write a short story about a cat who travels through time.",
    "Translate the following sentence to French: I love programming.",
    "List the main causes of climate change.",
    "What are the top 10 programming languages in 2026?",
    "How do you solve a quadratic equation using the quadratic formula?",
    "Give me a recipe for a vegan chocolate cake.",
    "What is the capital of Brazil?",
    "Summarize the key events of World War II.",
    "Explain the concept of attention in transformers.",
    "Write a poem about the ocean in the style of Shakespeare.",
    "What are the benefits of regular exercise?",
    "Describe the life cycle of a butterfly.",
    "Generate 5 creative ideas for a startup in AI.",
    "How do I set up a Python virtual environment?",
    "What is the difference between TCP and UDP protocols?",
    "Provide a Python snippet to reverse a string.",
    "Explain the difference between deep learning and machine learning.",
    "List 5 famous paintings by Van Gogh.",
    "How do you make homemade pasta from scratch?",
    "Write a letter to a friend inviting them to a birthday party.",
    "Explain the process of photosynthesis.",
    "What is the meaning of life according to philosophy?",
    "Describe the structure of the human brain.",
    "Give 10 tips for improving mental health.",
    "Explain how blockchain technology works.",
    "Write a haiku about spring.",
    "List 5 interesting facts about space exploration.",
    "How do you optimize SQL queries for performance?",
    "Generate a short dialogue between a robot and a human.",
    "Explain Newton's three laws of motion.",
    "Provide an example of recursion in Python.",
    "What are the main differences between Java and C++?",
    "Describe a day in the life of an astronaut on the ISS.",
    "Write a limerick about a clever fox.",
    "Explain the theory of relativity in simple terms.",
    "List 5 common algorithms used in AI.",
    "Give an example of a palindrome in English.",
    "How do you implement a linked list in Python?",
    "Describe the plot of Romeo and Juliet.",
    "Write a short essay on the importance of education.",
    "Provide 5 tips for effective time management.",
    "Explain the concept of entropy in thermodynamics.",
    "List 5 famous landmarks in Europe.",
    "How do you calculate compound interest?",
    "Generate a random joke about programmers.",
    "Describe the differences between HTML and CSS.",
    "What are the symptoms of the common cold?",
    "Explain the significance of the Internet of Things (IoT).",
    "Write a motivational quote about persistence.",
    "Provide a summary of the book '1984' by George Orwell.",
    "Explain how neural networks learn from data.",
    "List 5 programming challenges suitable for beginners.",
    "Describe the water cycle in nature.",
    "Generate a dialogue between Sherlock Holmes and Dr. Watson.",
    "How do you set up a REST API in Python?",
    "Explain the differences between a list and a tuple in Python.",
    "Write a short story about a dragon and a wizard.",
    "List 5 famous mathematicians in history.",
    "Provide 5 examples of metaphors in literature.",
    "Explain the concept of object-oriented programming.",
    "Describe the process of human digestion.",
    "Give 10 ways to reduce plastic usage in daily life.",
    "Write a short poem about autumn leaves.",
    "Explain the difference between AC and DC current.",
    "List 5 popular machine learning libraries in Python.",
    "How do you create a simple chatbot in Python?",
    "Describe the major events of the French Revolution.",
    "Provide an example of a binary search algorithm.",
    "Write a dialogue between two friends planning a trip.",
    "Explain the greenhouse effect and its impact.",
    "List 5 famous scientists and their discoveries.",
    "Generate 3 creative slogans for a tech company.",
    "Describe the anatomy of the human heart.",
    "Explain the differences between classical and quantum physics.",
    "Write a short story set in a futuristic city.",
    "List 10 countries in Africa.",
    "How do you calculate the area of a circle?",
    "Provide a Python function to check if a number is prime.",
    "Describe the plot of 'The Great Gatsby'.",
    "Explain how vaccines work in the human body.",
    "List 5 programming paradigms and their features.",
    "Write a haiku about winter snow.",
    "Generate a dialogue between a teacher and a student about math.",
    "Explain the difference between IPv4 and IPv6.",
    "List 5 famous composers in classical music.",
    "Describe the process of DNA replication.",
    "Write a short story about a lost puppy finding its home.",
    "Explain the principle of superposition in physics.",
    "Provide 5 tips for learning a new language quickly.",
    "List 10 famous inventions of the 20th century.",
    "Write a short motivational speech about achieving goals.",
    "Explain the difference between convolutional and recurrent neural networks.",
    "Describe the lifecycle of a star.",
    "Generate a funny dialogue between two AI assistants.",
    "List 5 important events in American history.",
    "Explain the Pythagorean theorem with an example.",
    "Write a poem about friendship.",
    "Provide a Python snippet to sort a list in ascending order.",
    "Describe the major rivers of the world.",
    "Explain the concept of machine learning overfitting and underfitting.",
    "List 5 famous novels written in the 19th century.",
    "Write a dialogue between a king and a wise advisor.",
    "Explain the differences between renewable and non-renewable energy.",
    "Generate 3 creative product names for a coffee shop.",
    "Describe the structure and function of the human lungs.",
    "Explain the concept of reinforcement learning in AI.",
    "Write a short story about a mysterious island.",
    "List 5 famous philosophers and their main ideas.",
    "Provide tips for effective public speaking.",
    "Explain how photosynthesis contributes to the oxygen supply on Earth.",
    "Write a poem about the night sky.",
    "Describe the main components of a computer system.",
    "Explain the differences between HTTP and HTTPS protocols.",
    "Generate a dialogue between a doctor and a patient about health.",
]


def prepare_corpus():
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    N = 100
    corpus = "\n".join(dataset["train"]["text"][:N])
    return corpus


def chunk_list(lst, size):
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def train(batch: int = 5):
    transformer = Transformer(
        N=6, d_model=1024, token_dim=512, num_heads=8, vocab_size=500)
    corpus = prepare_corpus()
    transformer.train_tokenizer(corpus=corpus)
    batches = chunk_list(QUERIES, batch)
    print("Splitting queries into batches , we have : ", len(batches), " batches")
    loss_fn = nn.CrossEntropyLoss()
    lr = 0.01
    optimizer = Adam(transformer.parameters(), lr=lr,
                     betas=(0.9, .098), eps=1e-9)
    for index, batch in enumerate(batches):
        print("Step n°", index+1)
        tokens = transformer.encode_queries(queries=batch)
        input_ids = torch.tensor(tokens["input_ids"])
        target = input_ids[:, 1:]

        output: Tensor = transformer(batch, train=True)
        output_for_loss = output.permute(0, 2, 1)
        # predicted_token_ids = torch.argmax(output, dim=-1)
        print("Final output shape : ", output.shape)
        # target = target.unsqueeze(dim=-1)
        print("Target shape : ", target.shape)
        optimizer.zero_grad()
        loss = loss_fn(input=output_for_loss, target=target)
        print("loss : ", loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
        optimizer.step()
        total_norm = 0
        for p in transformer.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print("Total gradient norm:", total_norm)


if __name__ == "__main__":
    train(batch=5)
