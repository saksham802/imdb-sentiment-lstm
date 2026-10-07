"""PyTorch Dataset wrapper for the IMDB reviews."""

import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocess import encode_and_pad


class IMDBDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: dict[str, int], max_len: int):
        sequences = [encode_and_pad(t, vocab, max_len) for t in df["text"]]
        self.x = torch.tensor(sequences, dtype=torch.long)
        self.y = torch.tensor(df["label"].values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


def load_split(path: str) -> pd.DataFrame:
    """Load a CSV split and drop duplicate reviews."""
    return pd.read_csv(path).drop_duplicates().reset_index(drop=True)
