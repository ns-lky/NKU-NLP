"""Shared, deterministic data preparation for all three experiments."""

from __future__ import annotations

import json
import hashlib
import importlib.metadata
import random
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+(?:\.[0-9]+)?")


def set_seed(seed: int) -> None:
    """Seed Python and NumPy; model-specific libraries seed themselves too."""

    random.seed(seed)
    np.random.seed(seed)


def tokenize(text: str) -> list[str]:
    """Tokenize news text without requiring an external NLTK data download."""

    return TOKEN_RE.findall(str(text).lower())


def load_nyt(path: str | Path) -> pd.DataFrame:
    """Load and validate the NYT CSV used by every classifier."""

    frame = pd.read_csv(path)
    required = {"text", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"NYT CSV is missing columns: {sorted(missing)}")
    frame = frame.loc[:, ["text", "label"]].reset_index(drop=True)
    if frame.isna().any().any():
        raise ValueError("NYT contains missing text/labels; refusing to silently remove rows")
    frame["text"] = frame["text"].astype(str)
    frame["label"] = frame["label"].astype(str)
    if frame.empty:
        raise ValueError("NYT CSV contains no usable rows")
    frame.attrs["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return frame


def load_ag_texts(path: str | Path) -> list[str]:
    """Load AG News text, accepting either a named or a single unnamed column."""

    frame = pd.read_csv(path)
    if "text" in frame.columns:
        series = frame["text"]
    elif len(frame.columns) == 1:
        series = frame.iloc[:, 0]
    else:
        raise ValueError("AG CSV must contain a text column")
    return series.dropna().astype(str).tolist()


def split_nyt(
    frame: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return one deterministic stratified 80/10/10 split.

    The split is performed by row index, so every representation script uses
    exactly the same examples when invoked with the same seed.
    """

    indices = np.arange(len(frame))
    train_idx, remainder_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed,
        stratify=frame["label"].to_numpy(),
    )
    val_idx, test_idx = train_test_split(
        remainder_idx,
        test_size=0.50,
        random_state=seed,
        stratify=frame.iloc[remainder_idx]["label"].to_numpy(),
    )
    return frame.iloc[train_idx].copy(), frame.iloc[val_idx].copy(), frame.iloc[test_idx].copy()


def describe_split(frame, train, validation, test, seed: int) -> dict:
    """Record the exact ordered row IDs, rather than just the nominal ratios."""

    parts = {"train": train, "validation": validation, "test": test}
    indices = {name: part.index.tolist() for name, part in parts.items()}
    sets = [set(ids) for ids in indices.values()]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Overlapping data partitions")
    if set.union(*sets) != set(frame.index):
        raise ValueError("The split does not cover every NYT row")
    fingerprint = hashlib.sha256(
        json.dumps(indices, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "seed": seed,
        "strategy": "stratified shuffled 80/10/10",
        "dataset_sha256": frame.attrs["sha256"],
        "dataset_rows": len(frame),
        "split_sha256": fingerprint,
        "sizes": {name: len(part) for name, part in parts.items()},
        "class_counts": {
            name: {str(k): int(v) for k, v in part.label.value_counts().items()}
            for name, part in parts.items()
        },
        "row_indices": indices,
    }


def prepare_nyt(path: str | Path, seed: int):
    frame = load_nyt(path)
    parts = split_nyt(frame, seed)
    manifest = describe_split(frame, *parts, seed)
    write_json("results/split.json", manifest)
    # Large ordered index arrays are stored once, in split.json.
    summary = {key: value for key, value in manifest.items() if key != "row_indices"}
    return frame, *parts, summary


def run_metadata(args) -> dict:
    """Capture actual arguments and library versions alongside the measurements."""

    versions = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "gensim", "torch", "transformers"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "python_version": sys.version,
        "library_versions": versions,
        "argv": sys.argv,
    }


def save_evaluation(frame, predictions, path: str | Path) -> dict:
    """Save row-level evidence so metrics can be recomputed independently."""

    predictions = list(predictions)
    labels = sorted(frame.label.unique())
    metrics = classification_metrics(frame.label, predictions)
    metrics["labels"] = labels
    metrics["confusion_matrix"] = confusion_matrix(frame.label, predictions, labels=labels).tolist()
    metrics["classification_report"] = classification_report(
        frame.label, predictions, labels=labels, output_dict=True, zero_division=0
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_index": frame.index, "label": frame.label.tolist(), "prediction": predictions}).to_csv(destination, index=False)
    return metrics


def classification_metrics(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, float]:
    """Compute the two metrics required by the assignment."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def write_json(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
