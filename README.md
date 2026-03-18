## BPE Transformer (GPT-1 style implementation)

This repository contains an educational / experimental implementation of Transformer building blocks and a GPT-like model. The implementation follows the broad architecture and training objective used in the original GPT paper (radford et al., 2018) and includes a custom byte-level BPE tokenizer with several performance-minded optimizations.

This README summarizes the repository, explains the BPE implementation and optimizations (max-heap + local neighbour merging + trie-based encoding), and gives quickstart instructions for training and inference.

## Relation to GPT-1

The design and training loop in `gpt.py` implements the generative pre-training objective from:

- Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). "Improving Language Understanding by Generative Pre-Training." Available: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

This repository is not an exact reproduction of the OpenAI codebase, but it follows the core idea: train a multi-layer, decoder-only Transformer with a next-token prediction objective on unlabeled text, then use the learned model for downstream tasks or generation.

## Highlights / What this repo implements

- A lightweight GPT-style model with multi-head causal attention and position embeddings (`gpt.py`, `attention.py`, `transformer_block.py`).
- A custom byte-level BPE tokenizer (`bpe.py`) with these optimizations:
	- Pair frequency selection via a max-heap (priority queue) for efficient retrieval of the most frequent pair.
	- A doubly-linked list of byte tokens and per-pair position sets so merges update only local neighbours (an O(occurrences) merge), not the entire corpus.
	- Lazy-deletion in the heap (to avoid costly decrease-key) along with a compact PairHeap wrapper.
	- A Trie built from final vocabulary for O(text_length) encoding at inference time.
- Training and generation harness (`gpt.py`) that saves model checkpoints and the tokenizer state, plus a small `inference.py` helper to load checkpoints and generate text.

## BPE implementation — optimizations explained

The tokenizer is implemented in `bpe.py`. Key implementation points:

- Initialization: the input corpus is normalized and converted into a flat byte list. A doubly-linked list (arrays `_prev` and `_next`) represents adjacent tokens, and `_active` marks alive positions.
- Pair counting: in one pass we count adjacent byte-pair occurrences and populate a PairHeap with (pair -> count).
- Max-heap (PairHeap): a max-heap (via Python's heapq with negated counts) returns the most frequent pair quickly. Because Python's heapq lacks decrease-key the code uses lazy-deletion; counts are kept in a dict and popped entries are validated against the dict to skip stale items.
- Local neighbour merging (O(occurrences)): when we merge a most-frequent pair, we iterate only over positions where that pair occurs (stored in `_pair_positions`) and update the local neighbours and the heap counts for affected pairs. This avoids scanning the whole corpus when a pair changes.
- Trie + encode: after vocabulary is built, we construct a Trie of token strings for fast encoding: longest-prefix match on bytes yields linear-time encoding in the text length (instead of re-applying merge rules repeatedly).

Why these choices matter:
- Max-heap gives O(log V) access to the best pair (V = number of distinct pairs), which is efficient when combined with lazy deletion.
- Local neighbour updates make each merge proportional to the occurrences of that pair rather than corpus size, which is often much smaller and is closer to the textbook "merge all occurrences" operation complexity.
- The Trie enables fast runtime encoding and avoids repeating merge-rule passes per input.

Limitations and tradeoffs:
- The lazy-deletion heap means the heap can accumulate stale entries; the code mitigates this by validating popped entries against the authoritative counts dict. This is simple and robust but not the most memory-efficient.
- The implementation is single-threaded and in-memory; for very large corpora you would want a disk-backed or streaming approach and more careful memory management.

## Files (short descriptions)

- `bpe.py` — byte-level BPE tokenizer, PairHeap, Trie and encode/decode logic.
- `attention.py` — attention head and multi-head causal/cross attention.
- `transformer_block.py` — encoder/decoder transformer blocks and feed-forward layers.
- `encoder.py`, `decoder.py`, `transformer.py` — higher-level encoder/decoder/seq2seq Transformer wiring.
- `gpt.py` — GPT-style decoder-only model class, training loop, tokenizer save/load and generation utilities.
- `inference.py` — helper to load a checkpoint + tokenizer and run generation from queries.
- `embedding.py` — token embedding + sinusoidal positional encoding helper.
- `bert.py`, `train_bert.py` — a BERT-style encoder wrapper and a classification training harness (branch-focused experiments).
- `train_transformer.py`, `main.py` — utilities and examples for training Transformer variants.
- `corpus.txt` — example / sample corpus text.
- `requirements.txt` — Python dependencies (torch, datasets, numpy).

## Quickstart — install & run

1) Create a virtual environment and install dependencies (macOS / zsh):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Train the tokenizer (or let the model training do it):

You can train the tokenizer directly from a corpus using the `BPE` API. The `gpt.py` training harness calls `train_tokenizer` if it cannot load a saved tokenizer.

3) Train GPT (toy example — see `gpt.py` config):

```bash
python gpt.py
```

Notes:
- Model and tokenizer checkpoints: `gpt.py` saves model checkpoints to `gpt_checkpoint.pt` and tokenizer state to `gpt_checkpoint_tokenizer.pt` (configurable in `GPTConfig`).
- Generation: use `inference.py` to load a checkpoint and run `load_and_generate(queries=...)`. Example:

```bash
python inference.py
```

## How to use the code (recommended minimal flow)

1. Prepare a corpus (e.g., using `datasets` or your own text file).
2. Configure `GPTConfig` in `gpt.py` (vocab size, token_dim, d_model, N layers, lr, epochs).
3. Run `python gpt.py` to train; this trains tokenizer if needed and runs next-token prediction training.
4. Run `python inference.py` (or call `load_and_generate`) to generate text from the trained checkpoint.

## Notes / Next steps and improvements

- Add unit tests for the tokenizer (enc/dec correctness) and a small smoke test for model forward/backward.
- Add a streaming tokenizer trainer for large corpora and disk-backed pair counts.
- Replace the lazy-deletion heap with a decrease-key-enabled priority structure (or use a binary indexed tree) for memory efficiency.
- Add proper argument parsing and config files for `gpt.py` training harness.
- adding fine tuning tasks mentionned in the original paper.

## Citation

If you find this repo useful, please cite the original GPT paper:

Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). "Improving Language Understanding by Generative Pre-Training." https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

