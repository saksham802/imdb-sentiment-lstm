"""Text cleaning, vocabulary building, encoding and padding."""

import string
from collections import Counter

PAD_IDX = 0  # reserved for padding
UNK_IDX = 1  # reserved for words not seen during training

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def tokenize(text: str) -> list[str]:
    """Lower-case, strip punctuation and split on whitespace."""
    return text.lower().translate(_PUNCT_TABLE).split()


def build_vocab(texts, min_freq: int = 1) -> dict[str, int]:
    """Map every word in the training corpus to a unique integer index.

    Index 0 is reserved for <PAD> and index 1 for <UNK>.
    """
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
    for word, freq in counter.most_common():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int]) -> list[int]:
    """Convert a review into a list of word indices."""
    return [vocab.get(word, UNK_IDX) for word in tokenize(text)]


def pad(sequence: list[int], max_len: int) -> list[int]:
    """Truncate or right-pad a sequence to exactly `max_len` tokens."""
    if len(sequence) >= max_len:
        return sequence[:max_len]
    return sequence + [PAD_IDX] * (max_len - len(sequence))


def encode_and_pad(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    return pad(encode(text, vocab), max_len)
