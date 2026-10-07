"""Predict the sentiment of a custom movie review.

Usage:
    python src/predict.py "This movie was an absolute masterpiece!"
"""

import argparse
import json
import os

import torch

from model import SentimentLSTM
from preprocess import encode_and_pad

LABELS = {0: "Negative", 1: "Positive"}


def main():
    p = argparse.ArgumentParser(description="Predict sentiment of a review")
    p.add_argument("text", type=str, help="Review text to classify")
    p.add_argument("--model-dir", type=str, default="models")
    args = p.parse_args()

    with open(os.path.join(args.model_dir, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    checkpoint = torch.load(os.path.join(args.model_dir, "sentiment_lstm.pt"),
                            map_location="cpu")
    cfg = checkpoint["config"]

    model = SentimentLSTM(len(vocab), cfg["embed_dim"], cfg["hidden_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    x = torch.tensor([encode_and_pad(args.text, vocab, cfg["max_len"])])
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    label = int(probs.argmax())

    print(f"Review     : {args.text}")
    print(f"Sentiment  : {LABELS[label]}")
    print(f"Confidence : {probs[label].item() * 100:.2f}%")


if __name__ == "__main__":
    main()
