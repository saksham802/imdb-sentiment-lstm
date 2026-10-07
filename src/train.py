"""Train and evaluate the LSTM sentiment classifier on the IMDB dataset.

Usage:
    python src/train.py --epochs 5 --batch-size 32
"""

import argparse
import json
import os

import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import IMDBDataset, load_split
from model import SentimentLSTM
from preprocess import build_vocab

KAGGLE_DATASET = "columbine/imdb-dataset-sentiment-analysis-in-csv-format"


def parse_args():
    p = argparse.ArgumentParser(description="Train an LSTM sentiment classifier")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-len", type=int, default=300)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--output-dir", type=str, default="models")
    return p.parse_args()


def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x).argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data
    data_dir = kagglehub.dataset_download(KAGGLE_DATASET)
    train_df = load_split(os.path.join(data_dir, "Train.csv"))
    valid_df = load_split(os.path.join(data_dir, "Valid.csv"))
    test_df = load_split(os.path.join(data_dir, "Test.csv"))
    print(f"Train: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)}")

    # 2. Vocabulary (built on training data only, to avoid leakage)
    vocab = build_vocab(train_df["text"])
    print(f"Vocabulary size: {len(vocab)}")

    train_loader = DataLoader(IMDBDataset(train_df, vocab, args.max_len),
                              batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(IMDBDataset(valid_df, vocab, args.max_len), batch_size=64)
    test_loader = DataLoader(IMDBDataset(test_df, vocab, args.max_len), batch_size=64)

    # 3. Model
    model = SentimentLSTM(len(vocab), args.embed_dim, args.hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4. Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate(model, valid_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | "
              f"Loss: {total_loss / len(train_loader):.4f} | Val Acc: {val_acc * 100:.2f}%")

    # 5. Test
    test_acc = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # 6. Save artifacts
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": vars(args)},
               os.path.join(args.output_dir, "sentiment_lstm.pt"))
    with open(os.path.join(args.output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    print(f"Model and vocabulary saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()
