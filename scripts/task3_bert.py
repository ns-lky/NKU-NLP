"""Task 3: fine-tune BERT-base-uncased for NYT classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.common import classification_metrics, load_nyt, set_seed, split_nyt, write_json


class EncodedNews(Dataset):
    def __init__(self, encodings: dict[str, list], labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index], dtype=torch.long) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def encode(frame, tokenizer, label_to_id: dict[str, int], max_length: int, max_samples: int | None) -> EncodedNews:
    if max_samples is not None:
        frame = frame.iloc[:max_samples]
    encodings = tokenizer(
        frame["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    labels = [label_to_id[label] for label in frame["label"]]
    return EncodedNews(encodings, labels)


def evaluate(model, loader, device, id_to_label: dict[int, str]) -> dict[str, float]:
    model.eval()
    labels: list[str] = []
    predictions: list[str] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predictions.extend(id_to_label[int(index)] for index in logits.argmax(dim=-1).cpu())
            labels.extend(id_to_label[int(index)] for index in batch["labels"].cpu())
    return classification_metrics(labels, predictions)


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    frame = load_nyt(args.nyt)
    train, validation, test = split_nyt(frame, args.seed)
    labels = sorted(frame["label"].unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_dataset = encode(train, tokenizer, label_to_id, args.max_length, args.max_train_samples)
    val_dataset = encode(validation, tokenizer, label_to_id, args.max_length, args.max_eval_samples)
    test_dataset = encode(test, tokenizer, label_to_id, args.max_length, args.max_eval_samples)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
        # The checkpoint's original classification head usually has a
        # different number of labels; keep the encoder and reinitialize it.
        ignore_mismatched_sizes=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_validation = -np.inf
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
        validation_metrics = evaluate(model, val_loader, device, id_to_label)
        history.append({"epoch": epoch, "train_loss": total_loss / max(1, len(train_loader)), **validation_metrics})
        print(
            f"epoch {epoch}/{args.epochs} | loss={history[-1]['train_loss']:.4f} | "
            f"val acc={validation_metrics['accuracy']:.4f} f1={validation_metrics['macro_f1']:.4f}"
        )
        if validation_metrics["macro_f1"] > best_validation:
            best_validation = validation_metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, id_to_label)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    payload = {
        "task": "Task 3 - BERT fine-tuning",
        "model_name": args.model_name,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
        "labels": labels,
        "history": history,
        "test": test_metrics,
    }
    write_json(args.output, payload)
    print(f"test        | acc={test_metrics['accuracy']:.4f} f1={test_metrics['macro_f1']:.4f}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nyt", type=Path, default=Path("HW-1/nyt.csv"))
    parser.add_argument("--model-name", default="google-bert/bert-base-uncased")
    parser.add_argument("--output", type=Path, default=Path("outputs/task3_bert.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/task3_bert"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda, cpu, or leave unset for automatic selection")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
