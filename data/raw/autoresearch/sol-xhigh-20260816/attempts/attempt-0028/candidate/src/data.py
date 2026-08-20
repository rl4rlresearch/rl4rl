"""
Data pipeline for 10-digit addition.

FORMAT (interleaved reversed operands + reversed output):
  Input:  "a0b0a1b1...a9b9=" where a0/b0 are the least-significant digits
  Output: reversed(C) zero-padded to 11 digits

Example: 123 + 45 = 168
  Input:  "35241000000000000000="
  Output: "86100000000"  (reversed "00000000168")

Why this format:
  1. Column alignment: corresponding operand digits are adjacent and ordered LSB-first.
  2. Reversed output: LSB-first aligns with carry propagation order.
  3. Fixed-length output (11 digits): no stop-signal learning needed.

Tokenization: digits 0-9 + special tokens (=, BOS) = 12 tokens.
"""

import random
import torch
from torch.utils.data import Dataset


# Token vocabulary
TOKENS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '<bos>']
TOKEN_TO_ID = {t: i for i, t in enumerate(TOKENS)}
ID_TO_TOKEN = {i: t for t, i in TOKEN_TO_ID.items()}

PAD_ID = -1  # Compatibility sentinel; fixed-length batches never use padding.
BOS_ID = TOKEN_TO_ID['<bos>']
EOS_ID = None  # Fixed-length decoding emits exactly OUT_DIGITS tokens.

VOCAB_SIZE = len(TOKENS)  # 12

NUM_DIGITS = 10  # Operands padded to this many digits
OUT_DIGITS = 11  # Output length (max sum of two 10-digit numbers = 11 digits)

# Fixed sequence length: BOS + 20(interleaved operands) + 1(=) + 11 = 33
FIXED_SEQ_LEN = 1 + 2 * NUM_DIGITS + 1 + OUT_DIGITS  # 33


def preprocess(a: int, b: int) -> str:
    """Convert two integers to interleaved LSB-first padded operand tokens."""
    a_str = str(a).zfill(NUM_DIGITS)[::-1]
    b_str = str(b).zfill(NUM_DIGITS)[::-1]
    return ''.join(x for pair in zip(a_str, b_str) for x in pair) + '='


def postprocess(model_output: str) -> int:
    """Convert model output (reversed digit string) back to integer."""
    cleaned = ''.join(c for c in model_output if c.isdigit())
    if not cleaned:
        return 0
    normal = cleaned[::-1]
    return int(normal)


def make_target(a: int, b: int) -> str:
    """Create the target string (reversed sum, padded to OUT_DIGITS)."""
    c = a + b
    c_str = str(c).zfill(OUT_DIGITS)
    return c_str[::-1]


def encode(text: str) -> list:
    """Encode a string to token IDs."""
    return [TOKEN_TO_ID[ch] for ch in text]


def decode(ids: list) -> str:
    """Decode token IDs to string."""
    return ''.join(ID_TO_TOKEN[i] for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID))


class AdditionDataset(Dataset):
    """
    Dataset for addition problems. Each example is a fixed-length (35 token)
    sequence: <bos> AAAAAAAAAA+BBBBBBBBBB= reversed(C_padded) <eos>
    Loss is only computed on the answer portion (after '=').
    """

    def __init__(self, num_examples, max_digits=10, seed=None, min_digits=1):
        self.examples = []
        rng = random.Random(seed)

        for _ in range(num_examples):
            d1 = rng.randint(min_digits, max_digits)
            d2 = rng.randint(min_digits, max_digits)
            a = rng.randint(0, 10**d1 - 1)
            b = rng.randint(0, 10**d2 - 1)

            input_str = preprocess(a, b)
            target_str = make_target(a, b)
            full_str = input_str + target_str
            token_ids = [BOS_ID] + encode(full_str)

            # Answer starts after BOS + 20 interleaved operand digits + 1(=).
            answer_start = 1 + 2 * NUM_DIGITS + 1  # 22

            assert len(token_ids) == FIXED_SEQ_LEN, f"Expected {FIXED_SEQ_LEN}, got {len(token_ids)}"
            self.examples.append({
                'token_ids': token_ids,
                'answer_start': answer_start,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    """Collate -- all sequences are same length, no padding needed."""
    input_ids = []
    labels = []

    for ex in batch:
        ids = ex['token_ids']
        ans_start = ex['answer_start']
        input_ids.append(ids)

        # Labels: -100 for input portion (no loss), actual ids for answer portion
        label = [-100] * ans_start + ids[ans_start:]
        labels.append(label)

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
    }


def make_test_set(num_examples=10000, seed=42):
    """Generate a held-out test set. Both operands uniform in [0, 10^10 - 1]."""
    rng = random.Random(seed)
    problems = []
    for _ in range(num_examples):
        a = rng.randint(0, 10**10 - 1)
        b = rng.randint(0, 10**10 - 1)
        problems.append((a, b, a + b))
    return problems
