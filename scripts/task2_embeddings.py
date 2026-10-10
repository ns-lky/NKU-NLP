"""Task 2: GloVe and Word2Vec document-average representations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.common import classification_metrics, load_ag_texts, load_nyt, set_seed, split_nyt, tokenize, write_json


def load_glove(path: Path, dimension: int = 100) -> dict[str, np.ndarray]:
    """Read a plain-text GloVe file such as ``glove.6B.100d.txt``."""

    vectors: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip().split(" ")
            if len(fields) != dimension + 1:
                continue
            try:
                vectors[fields[0]] = np.asarray(fields[1:], dtype=np.float32)
            except ValueError:
                continue
    if not vectors:
        raise ValueError(f"No {dimension}-dimensional vectors found in {path}")
    return vectors


def mean_vectors(tokenized_documents: Iterable[list[str]], vectors: Mapping[str, np.ndarray], dimension: int) -> np.ndarray:
    output = []
    zero = np.zeros(dimension, dtype=np.float32)
    for tokens in tokenized_documents:
        known = [vectors[token] for token in tokens if token in vectors]
        output.append(np.mean(known, axis=0, dtype=np.float32) if known else zero)
    return np.asarray(output, dtype=np.float32)


def train_word2vec(sentences: list[list[str]], args: argparse.Namespace):
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise RuntimeError("Task 2 Word2Vec requires gensim; install requirements.txt") from exc
    return Word2Vec(
        sentences=sentences,
        vector_size=args.dimension,
        window=5,
        min_count=args.min_count,
        workers=args.workers,
        sg=1,
        seed=args.seed,
        epochs=args.epochs,
    )


def run_logistic(train_x, val_x, test_x, train_y, val_y, test_y, seed: int, max_iter: int) -> dict:
    classifier = LogisticRegression(max_iter=max_iter, solver="lbfgs", random_state=seed)
    classifier.fit(train_x, train_y)
    return {
        "validation": classification_metrics(val_y, classifier.predict(val_x)),
        "test": classification_metrics(test_y, classifier.predict(test_x)),
    }


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    frame = load_nyt(args.nyt)
    train, validation, test = split_nyt(frame, args.seed)
    train_tokens = [tokenize(text) for text in train["text"]]
    val_tokens = [tokenize(text) for text in validation["text"]]
    test_tokens = [tokenize(text) for text in test["text"]]
    results: dict[str, dict] = {}

    if args.method in {"all", "glove"}:
        if args.glove_path is None:
            raise ValueError("--glove-path is required for the GloVe experiment")
        glove = load_glove(args.glove_path, args.dimension)
        results["glove"] = run_logistic(
            mean_vectors(train_tokens, glove, args.dimension),
            mean_vectors(val_tokens, glove, args.dimension),
            mean_vectors(test_tokens, glove, args.dimension),
            train["label"], validation["label"], test["label"], args.seed, args.max_iter,
        )
        print(f"glove        | test acc={results['glove']['test']['accuracy']:.4f} f1={results['glove']['test']['macro_f1']:.4f}")

    if args.method in {"all", "ag-word2vec"}:
        ag_sentences = [tokenize(text) for text in load_ag_texts(args.ag)]
        ag_model = train_word2vec(ag_sentences, args)
        results["word2vec_ag"] = run_logistic(
            mean_vectors(train_tokens, ag_model.wv, args.dimension),
            mean_vectors(val_tokens, ag_model.wv, args.dimension),
            mean_vectors(test_tokens, ag_model.wv, args.dimension),
            train["label"], validation["label"], test["label"], args.seed, args.max_iter,
        )
        print(f"word2vec_ag  | test acc={results['word2vec_ag']['test']['accuracy']:.4f} f1={results['word2vec_ag']['test']['macro_f1']:.4f}")

    if args.method in {"all", "nyt-word2vec"}:
        nyt_model = train_word2vec(train_tokens, args)
        results["word2vec_nyt"] = run_logistic(
            mean_vectors(train_tokens, nyt_model.wv, args.dimension),
            mean_vectors(val_tokens, nyt_model.wv, args.dimension),
            mean_vectors(test_tokens, nyt_model.wv, args.dimension),
            train["label"], validation["label"], test["label"], args.seed, args.max_iter,
        )
        print(f"word2vec_nyt | test acc={results['word2vec_nyt']['test']['accuracy']:.4f} f1={results['word2vec_nyt']['test']['macro_f1']:.4f}")

    payload = {"task": "Task 2 - Word Embeddings", "seed": args.seed, "dimension": args.dimension, "epochs": args.epochs, "results": results}
    write_json(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nyt", type=Path, default=Path("HW-1/nyt.csv"))
    parser.add_argument("--ag", type=Path, default=Path("HW-1/ag.csv"))
    parser.add_argument("--glove-path", type=Path, default=None)
    parser.add_argument("--method", choices=["all", "glove", "ag-word2vec", "nyt-word2vec"], default="all")
    parser.add_argument("--output", type=Path, default=Path("outputs/task2_embeddings.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
