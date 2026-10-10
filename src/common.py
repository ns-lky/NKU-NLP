"""Shared, deterministic data preparation for all three experiments."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
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
    frame = frame.loc[:, ["text", "label"]].dropna().reset_index(drop=True)
    frame["text"] = frame["text"].astype(str)
    frame["label"] = frame["label"].astype(str)
    if frame.empty:
        raise ValueError("NYT CSV contains no usable rows")
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
