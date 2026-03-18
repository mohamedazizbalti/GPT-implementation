from datasets import load_dataset
from collections import defaultdict
import heapq
import torch
import numpy as np


def build_reverse_vocab(vocab: dict) -> dict:
    return {v: k for k, v in vocab.items()}


# ── Trie (unchanged, used now for encoding) ──────────────────────────────────

class TrieNode:
    def __init__(self):
        self.children: dict[int, "TrieNode"] = {}
        self.token_id: int | None = None


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, token_str: str, token_id: int):
        node = self.root
        for byte in token_str.encode("utf-8"):
            if byte not in node.children:
                node.children[byte] = TrieNode()
            node = node.children[byte]
        node.token_id = token_id

    def longest_match(self, bytes_list: list[int], start: int) -> tuple[int, int]:
        node = self.root
        last_match_id = None
        last_match_end = start
        i = start
        while i < len(bytes_list):
            byte = bytes_list[i]
            if byte not in node.children:
                break
            node = node.children[byte]
            i += 1
            if node.token_id is not None:
                last_match_id = node.token_id
                last_match_end = i
        return last_match_id, last_match_end


# ── Heap wrapper ─────────────────────────────────────────────────────────────

class PairHeap:
    """Max-heap of (pair → count) backed by a dict.

    We use a lazy-deletion strategy: entries are never removed physically;
    instead we check whether the count in `counts` still matches when we pop.
    """

    def __init__(self):
        self._heap: list[tuple[int, tuple[int, int]]] = []   # (-count, pair)
        self.counts: dict[tuple[int, int], int] = defaultdict(int)

    def push(self, pair: tuple[int, int], count: int):
        self.counts[pair] = count
        heapq.heappush(self._heap, (-count, pair))

    def increment(self, pair: tuple[int, int], delta: int = 1):
        self.counts[pair] += delta
        heapq.heappush(self._heap, (-self.counts[pair], pair))

    def decrement(self, pair: tuple[int, int], delta: int = 1):
        self.counts[pair] -= delta
        if self.counts[pair] <= 0:
            del self.counts[pair]

    def pop_max(self) -> tuple[tuple[int, int], int] | tuple[None, int]:
        while self._heap:
            neg_count, pair = heapq.heappop(self._heap)
            count = -neg_count
            # lazy deletion: skip stale entries
            if self.counts.get(pair, 0) == count:
                return pair, count
        return None, 0

    def peek_max(self) -> tuple[tuple[int, int], int] | tuple[None, int]:
        while self._heap:
            neg_count, pair = self._heap[0]
            count = -neg_count
            if self.counts.get(pair, 0) == count:
                return pair, count
            heapq.heappop(self._heap)          # remove stale
        return None, 0


# ── BPE ──────────────────────────────────────────────────────────────────────

class BPE:
    def __init__(self, vocab_size: int = 500, max_len: int = 512):
        if vocab_size < 257:
            raise ValueError("Vocab size can't be smaller than 256")
        self.current_size = 256
        self.vocab_size = vocab_size
        self.vocab: dict[str, int] = {}
        self.reverse_vocab: dict[int, str] = {}
        self.rules: dict[tuple[int, int], int] = {}
        self.max_length = max_len
        self.trie: Trie | None = None

        # internal corpus state
        self._tokens: list[int] = []          # flat token array
        # doubly-linked list (prev pointers)
        self._prev: list[int] = []
        # doubly-linked list (next pointers)
        self._next: list[int] = []
        self._active: list[bool] = []         # alive positions
        self._pair_positions: dict[tuple[int, int],
                                   set[int]] = defaultdict(set)
        self._heap = PairHeap()

        special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        for i in range(self.current_size):
            self.vocab[chr(i)] = i
        for token in special_tokens:
            self.vocab[token] = self.current_size
            self.current_size += 1
        self.reverse_vocab = build_reverse_vocab(self.vocab)

    # ── text helpers ─────────────────────────────────────────────────────────

    def _normalize(self, text: str) -> str:
        text = text.lower().replace("\n", " ")
        while "  " in text:
            text = text.replace("  ", " ")
        return text.strip()

    # ── corpus initialisation ────────────────────────────────────────────────

    def _init_corpus(self, corpus: str):
        """Build a doubly-linked list over the byte sequence and initialise
        the pair heap in one O(N) pass."""
        corpus = self._normalize(corpus)
        words = corpus.split()

        raw: list[int] = []
        for idx, word in enumerate(words):
            if idx > 0:
                word = " " + word
            raw.extend(word.encode("utf-8"))

        n = len(raw)
        self._tokens = raw[:]
        self._active = [True] * n
        self._prev = list(range(-1, n - 1))   # _prev[0] = -1 (sentinel)
        self._next = list(range(1, n + 1))     # _next[n-1] = n (sentinel)

        # count pairs and build positions
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        self._pair_positions = defaultdict(set)
        for i in range(n - 1):
            pair = (raw[i], raw[i + 1])
            pair_counts[pair] += 1
            self._pair_positions[pair].add(i)

        self._heap = PairHeap()
        for pair, count in pair_counts.items():
            self._heap.push(pair, count)

    # ── O(occurrences) merge ─────────────────────────────────────────────────

    def _merge_pair(self, pair: tuple[int, int], replacement: int):
        """Merge all occurrences of `pair` in the linked-list representation,
        updating only the local neighbourhood of each occurrence."""
        first, second = pair
        positions = list(self._pair_positions.pop(pair, set()))

        for i in positions:
            # guard: position may have been invalidated by an earlier merge in
            # this same call (shouldn't happen for BPE, but be safe)
            if not self._active[i]:
                continue
            j = self._next[i]
            if j >= len(self._tokens) or not self._active[j]:
                continue
            if self._tokens[i] != first or self._tokens[j] != second:
                continue

            # ── left neighbour ───────────────────────────────────────────────
            left = self._prev[i]
            if left >= 0 and self._active[left]:
                old_left_pair = (self._tokens[left], first)
                self._heap.decrement(old_left_pair)
                self._pair_positions[old_left_pair].discard(left)

            # ── right neighbour ──────────────────────────────────────────────
            right = self._next[j]
            if right < len(self._tokens) and self._active[right]:
                old_right_pair = (second, self._tokens[right])
                self._heap.decrement(old_right_pair)
                self._pair_positions[old_right_pair].discard(j)

            # ── perform merge ────────────────────────────────────────────────
            self._tokens[i] = replacement
            self._active[j] = False
            # relink: skip j
            self._next[i] = right
            if right < len(self._tokens):
                self._prev[right] = i

            # ── add new neighbour pairs ──────────────────────────────────────
            if left >= 0 and self._active[left]:
                new_left_pair = (self._tokens[left], replacement)
                self._heap.increment(new_left_pair)
                self._pair_positions[new_left_pair].add(left)

            if right < len(self._tokens) and self._active[right]:
                new_right_pair = (replacement, self._tokens[right])
                self._heap.increment(new_right_pair)
                self._pair_positions[new_right_pair].add(i)

    def train(self, corpus: str, min_frequency: int = 2):
        print("corpus contains:", len(corpus.split()), "words")
        self._init_corpus(corpus)
        max_iterations = self.vocab_size - self.current_size

        while True:
            if self.current_size >= self.vocab_size:
                print("vocab size max reached.")
                break

            pair, frequency = self._heap.peek_max()
            if pair is None or frequency < min_frequency:
                print(f"No more pairs with frequency >= {min_frequency}.")
                break

            # consume from heap
            self._heap.pop_max()

            new_id = self.current_size
            new_token = self.reverse_vocab[pair[0]
                                           ] + self.reverse_vocab[pair[1]]
            self.vocab[new_token] = new_id
            self.reverse_vocab[new_id] = new_token
            self.rules[pair] = new_id
            self.current_size += 1

            print(f"Adding pair: {pair}  frequency: {frequency}")
            self._merge_pair(pair, new_id)

        print("Done training")
        self._build_trie()

    def _build_trie(self):
        self.trie = Trie()
        skip = {"<PAD>", "<BOS>", "<EOS>", "<UNK>"}
        for token_str, token_id in self.vocab.items():
            if token_str not in skip:
                self.trie.insert(token_str, token_id)

    def encode_text(
        self,
        text: str,
        padding: bool = True,
        truncating: bool = True,
        left_pad: bool = False,
    ) -> dict:
        """
        FIX 1: use the Trie for O(text_length) encoding instead of
        re-applying every rule in sequence.
        """
        if self.trie is None:
            raise RuntimeError("Call train() before encode().")

        text = self._normalize(text)
        words = text.split()

        raw: list[int] = []
        for idx, word in enumerate(words):
            if idx > 0:
                word = " " + word
            raw.extend(word.encode("utf-8"))

        unk_id = self.vocab["<UNK>"]
        tokens: list[int] = []
        i = 0
        while i < len(raw):
            token_id, end = self.trie.longest_match(raw, i)
            if token_id is None:
                tokens.append(unk_id)
                i += 1
            else:
                tokens.append(token_id)
                i = end

        if truncating and len(tokens) + 2 > self.max_length:
            tokens = tokens[: self.max_length - 2]

        final = [self.vocab["<BOS>"]] + tokens + [self.vocab["<EOS>"]]

        pad_id = self.vocab["<PAD>"]
        if padding and len(final) < self.max_length:
            pad_count = self.max_length - len(final)
            pad = [pad_id] * pad_count
            final = pad + final if left_pad else final + pad

        attention_mask = [0 if b == pad_id else 1 for b in final]
        return {"input_ids": final, "attention_mask": attention_mask}

    def encode(
        self,
        corpus: str | list[str],
        padding: bool = True,
        truncating: bool = True,
        left_pad: bool = False,
    ) -> dict:
        batch: dict[str, list] = {"input_ids": [], "attention_mask": []}

        if isinstance(corpus, str):
            enc = self.encode_text(
                corpus, padding=padding, truncating=truncating, left_pad=left_pad)
            batch["input_ids"].append(enc["input_ids"])
            batch["attention_mask"].append(enc["attention_mask"])

        elif isinstance(corpus, list):
            for text in corpus:
                if not isinstance(text, str):
                    raise ValueError("Input must be strings.")
                enc = self.encode_text(
                    text, padding=False, truncating=truncating)
                batch["input_ids"].append(enc["input_ids"])
                batch["attention_mask"].append(enc["attention_mask"])

            if padding and len(batch["input_ids"]) > 1:
                pad_id = self.vocab["<PAD>"]
                max_len = max(len(s) for s in batch["input_ids"])
                for i in range(len(batch["input_ids"])):
                    diff = max_len - len(batch["input_ids"][i])
                    pad = [pad_id] * diff
                    if left_pad:
                        batch["input_ids"][i] = pad + batch["input_ids"][i]
                        batch["attention_mask"][i] = [0] * \
                            diff + batch["attention_mask"][i]
                    else:
                        batch["input_ids"][i].extend(pad)
                        batch["attention_mask"][i].extend([0] * diff)

        return {
            "input_ids": torch.tensor(batch["input_ids"]),
            "attention_mask": torch.tensor(batch["attention_mask"]),
        }

    def decode(self, tokenIDs: int | list[int]) -> str:
        if isinstance(tokenIDs, int):
            return self.reverse_vocab.get(tokenIDs, "<UNK>")
        return "".join(self.reverse_vocab.get(tid, "<UNK>") for tid in tokenIDs)


def main():
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    N = 100
    corpus = "\n".join(dataset["train"]["text"][:N])
    tokenizer = BPE(vocab_size=300)
    tokenizer.train(corpus)
    queries = [corpus]
    results = tokenizer.encode(queries, padding=False, truncating=False)
    print(results)


if __name__ == "__main__":
    main()
