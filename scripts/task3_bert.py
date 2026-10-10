"""Task 3: full three-epoch BERT-base-uncased fine-tuning on NYT."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import time

# Keep downloaded models and every auxiliary Hub cache inside this repository.
ROOT = Path(__file__).resolve().parents[1]
HF_HOME = ROOT / "artifacts" / "huggingface"
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
os.environ["HF_XET_CACHE"] = str(HF_HOME / "xet")
os.environ["TORCH_HOME"] = str(ROOT / "artifacts" / "torch")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.common import classification_metrics, prepare_nyt, run_metadata, save_evaluation, set_seed, write_json
from scripts.download_assets import BERT_REVISION

MODEL_NAME = "google-bert/bert-base-uncased"


class EncodedNews(Dataset):
    def __init__(self, frame, tokenizer, label_to_id):
        self.encodings = tokenizer(
            frame.text.tolist(), truncation=True, padding="max_length",
            max_length=64, return_tensors="pt",
        )
        self.labels = torch.tensor([label_to_id[label] for label in frame.label], dtype=torch.long)
        if self.encodings["input_ids"].shape != (len(frame), 64):
            raise ValueError("Every encoded document must have exactly 64 token positions")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {**{key: value[index] for key, value in self.encodings.items()}, "labels": self.labels[index]}


def evaluate(model, loader, device, id_to_label):
    model.eval()
    predictions, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            actual = batch.pop("labels")
            inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            logits = model(**inputs).logits
            predictions.extend(id_to_label[index] for index in logits.argmax(-1).cpu().tolist())
            labels.extend(id_to_label[index] for index in actual.tolist())
    return classification_metrics(labels, predictions), predictions


def run(args):
    if args.model_name != MODEL_NAME or args.max_length != 64 or args.epochs != 3:
        raise ValueError("Formal experiment requires google-bert/bert-base-uncased, max_length=64, epochs=3")
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}; GPU: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'none'}", flush=True)
    frame, train, validation, test, split = prepare_nyt(args.nyt, args.seed)
    labels = sorted(frame.label.unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=BERT_REVISION, cache_dir=HF_HOME / "hub", use_fast=True, local_files_only=True)
    datasets = [EncodedNews(part, tokenizer, label_to_id) for part in (train, validation, test)]
    shuffle_rng = torch.Generator().manual_seed(args.seed)
    loaders = [
        DataLoader(dataset, batch_size=args.batch_size, shuffle=index == 0,
                   generator=shuffle_rng if index == 0 else None,
                   pin_memory=device.type == "cuda")
        for index, dataset in enumerate(datasets)
    ]
    train_loader, val_loader, test_loader = loaders
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, revision=BERT_REVISION, cache_dir=HF_HOME / "hub", local_files_only=True, num_labels=len(labels),
        id2label=id_to_label, label2id=label_to_id,
    ).to(device)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if name.endswith("bias") or "LayerNorm.weight" in name else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.learning_rate,
    )
    total_steps = len(train_loader) * 3
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio), num_training_steps=total_steps,
    )
    use_amp = device.type == "cuda" and not args.fp32
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history, best_validation, best_epoch, start_epoch = [], -1.0, None, 1
    signature = {
        "model": MODEL_NAME, "split_sha256": split["split_sha256"], "dataset_sha256": split["dataset_sha256"],
        "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio, "seed": args.seed, "amp": use_amp,
    }
    if args.resume:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["signature"] != signature:
            raise ValueError("Checkpoint configuration/data does not match this run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        history = checkpoint["history"]
        best_validation, best_epoch = checkpoint["best_validation"], checkpoint["best_epoch"]
        start_epoch = checkpoint["completed_epochs"] + 1
        random.setstate(checkpoint["random_state"])
        np.random.set_state(checkpoint["numpy_state"])
        torch.set_rng_state(checkpoint["torch_state"])
        shuffle_rng.set_state(checkpoint["shuffle_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_state"])
        del checkpoint
        print(f"Resuming after {start_epoch - 1} completed epochs", flush=True)
    print(f"Full dataset: train={len(train)}, validation={len(validation)}, test={len(test)}; steps/epoch={len(train_loader)}", flush=True)
    training_started = time.perf_counter()
    for epoch in range(start_epoch, 4):
        model.train()
        total_loss, seen = 0.0, 0
        epoch_started = time.perf_counter()
        for step, batch in enumerate(train_loader, 1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            count = batch["labels"].size(0)
            total_loss += loss.item() * count
            seen += count
            if step % args.log_every == 0 or step == len(train_loader):
                elapsed = time.perf_counter() - epoch_started
                remaining = elapsed / step * (len(train_loader) - step)
                print(f"epoch {epoch}/3 step {step}/{len(train_loader)} loss={total_loss / seen:.4f} epoch_remaining={remaining:.0f}s", flush=True)
        if seen != len(train):
            raise RuntimeError("An epoch did not visit the entire training set")
        validation_metrics, _ = evaluate(model, val_loader, device, id_to_label)
        history.append({"epoch": epoch, "train_loss": total_loss / seen, "training_examples": seen,
                        "steps": len(train_loader), "seconds": time.perf_counter() - epoch_started,
                        **validation_metrics})
        print(f"epoch {epoch}/3 completed | val acc={validation_metrics['accuracy']:.4f} f1={validation_metrics['macro_f1']:.4f}", flush=True)
        if validation_metrics["macro_f1"] > best_validation:
            best_validation, best_epoch = validation_metrics["macro_f1"], epoch
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
        write_json(args.output.parent / "task3_training_history.json", {"split": split, "history": history, "best_epoch": best_epoch})
        torch.save({
            "signature": signature, "completed_epochs": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "history": history, "best_validation": best_validation, "best_epoch": best_epoch,
            "random_state": random.getstate(), "numpy_state": np.random.get_state(),
            "torch_state": torch.get_rng_state(), "shuffle_state": shuffle_rng.get_state(),
            "cuda_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }, args.checkpoint)
    if [item["epoch"] for item in history] != [1, 2, 3]:
        raise RuntimeError("Formal BERT results require all three complete epochs")
    del optimizer, scheduler, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Select using validation data only; access test predictions once, at the end.
    model = AutoModelForSequenceClassification.from_pretrained(args.output_dir, local_files_only=True).to(device)
    _, val_predictions = evaluate(model, val_loader, device, id_to_label)
    _, test_predictions = evaluate(model, test_loader, device, id_to_label)
    payload = {
        "task": "Task 3 - BERT fine-tuning", "model_name": MODEL_NAME, "max_length": 64, "epochs": 3,
        "seed": args.seed, "device": str(device), "mixed_precision": "fp16" if use_amp else "fp32",
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "labels": labels, "split": split, "run": run_metadata(args), "history": history,
        "best_epoch": best_epoch, "selection_metric": "validation macro_f1",
        "validation": save_evaluation(validation, val_predictions, args.output.parent / "predictions" / "task3_bert_validation.csv"),
        "test": save_evaluation(test, test_predictions, args.output.parent / "predictions" / "task3_bert_test.csv"),
        "training_seconds_this_invocation": time.perf_counter() - training_started,
    }
    write_json(args.output, payload)
    print(f"test | acc={payload['test']['accuracy']:.4f} f1={payload['test']['macro_f1']:.4f}; best epoch={best_epoch}", flush=True)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nyt", type=Path, default=Path("HW-1/nyt.csv"))
    parser.add_argument("--model-name", choices=[MODEL_NAME], default=MODEL_NAME)
    parser.add_argument("--max-length", type=int, choices=[64], default=64)
    parser.add_argument("--epochs", type=int, choices=[3], default=3)
    parser.add_argument("--output", type=Path, default=Path("results/task3_bert.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/task3_bert"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/task3_checkpoint.pt"))
    parser.add_argument("--resume", action="store_true", help="Resume from a completed epoch saved by this script")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--fp32", action="store_true", help="Disable CUDA automatic mixed precision")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
