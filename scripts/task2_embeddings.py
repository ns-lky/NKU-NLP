"""Task 2: GloVe and Word2Vec document-average representations."""

from __future__ import annotations

import argparse
import hashlib
import zlib
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

from src.common import load_ag_texts, prepare_nyt, run_metadata, save_evaluation, set_seed, tokenize, write_json


def load_glove(path: Path, dimension: int = 100) -> dict[str, np.ndarray]:
    """Read a plain-text GloVe file such as ``glove.6B.100d.txt``."""

    vectors: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if len(fields) != dimension + 1:
                raise ValueError(f"GloVe row {line_number} does not contain {dimension} dimensions")
            vector = np.asarray(fields[1:], dtype=np.float32)
            if not np.isfinite(vector).all():
                raise ValueError(f"Non-finite GloVe vector on row {line_number}")
            vectors[fields[0]] = vector
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


def stable_hash(word: str) -> int:
    return zlib.crc32(word.encode("utf-8"))


def coverage(documents, vectors) -> dict:
    total = sum(len(tokens) for tokens in documents)
    known = sum(sum(token in vectors for token in tokens) for tokens in documents)
    empty = sum(not any(token in vectors for token in tokens) for tokens in documents)
    return {"total_tokens": total, "known_tokens": known, "token_coverage": known / total if total else 0.0, "zero_vector_documents": empty}


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
        hashfxn=stable_hash,
        epochs=args.epochs,
    )


def run_logistic(train_x, val_x, test_x, train, validation, test, name: str, args) -> dict:
    classifier = LogisticRegression(max_iter=args.max_iter, solver="lbfgs", random_state=args.seed)
    classifier.fit(train_x, train.label)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, args.model_dir / f"{name}_logistic.joblib")
    return {
        "iterations": classifier.n_iter_.tolist(),
        "validation": save_evaluation(validation, classifier.predict(val_x), args.output.parent / "predictions" / f"task2_{name}_validation.csv"),
        "test": save_evaluation(test, classifier.predict(test_x), args.output.parent / "predictions" / f"task2_{name}_test.csv"),
    }


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    if args.dimension != 100:
        raise ValueError("The assignment requires exactly 100-dimensional word vectors")
    frame, train, validation, test, split = prepare_nyt(args.nyt, args.seed)
    args.model_dir.mkdir(parents=True, exist_ok=True)
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
            train, validation, test, "glove", args,
        )
        results["glove"]["vocabulary_size"] = len(glove)
        results["glove"]["source"] = "Stanford glove.6B.100d.txt"
        results["glove"]["source_sha256"] = hashlib.sha256(args.glove_path.read_bytes()).hexdigest()
        results["glove"]["coverage"] = {name: coverage(tokens, glove) for name, tokens in [("train", train_tokens), ("validation", val_tokens), ("test", test_tokens)]}
        print(f"glove        | test acc={results['glove']['test']['accuracy']:.4f} f1={results['glove']['test']['macro_f1']:.4f}")
        del glove

    if args.method in {"all", "ag-word2vec"}:
        ag_sentences = [tokenize(text) for text in load_ag_texts(args.ag)]
        ag_model = train_word2vec(ag_sentences, args)
        ag_model.save(str(args.model_dir / "word2vec_ag.model"))
        results["word2vec_ag"] = run_logistic(
            mean_vectors(train_tokens, ag_model.wv, args.dimension),
            mean_vectors(val_tokens, ag_model.wv, args.dimension),
            mean_vectors(test_tokens, ag_model.wv, args.dimension),
            train, validation, test, "word2vec_ag", args,
        )
        results["word2vec_ag"]["training_documents"] = len(ag_sentences)
        results["word2vec_ag"]["source_sha256"] = hashlib.sha256(args.ag.read_bytes()).hexdigest()
        results["word2vec_ag"]["vocabulary_size"] = len(ag_model.wv)
        results["word2vec_ag"]["coverage"] = {name: coverage(tokens, ag_model.wv) for name, tokens in [("train", train_tokens), ("validation", val_tokens), ("test", test_tokens)]}
        print(f"word2vec_ag  | test acc={results['word2vec_ag']['test']['accuracy']:.4f} f1={results['word2vec_ag']['test']['macro_f1']:.4f}")
        del ag_model, ag_sentences

    if args.method in {"all", "nyt-word2vec"}:
        nyt_model = train_word2vec(train_tokens, args)
        nyt_model.save(str(args.model_dir / "word2vec_nyt.model"))
        results["word2vec_nyt"] = run_logistic(
            mean_vectors(train_tokens, nyt_model.wv, args.dimension),
            mean_vectors(val_tokens, nyt_model.wv, args.dimension),
            mean_vectors(test_tokens, nyt_model.wv, args.dimension),
            train, validation, test, "word2vec_nyt", args,
        )
        results["word2vec_nyt"]["training_documents"] = len(train_tokens)
        results["word2vec_nyt"]["training_scope"] = "NYT training partition only; validation/test excluded"
        results["word2vec_nyt"]["vocabulary_size"] = len(nyt_model.wv)
        results["word2vec_nyt"]["coverage"] = {name: coverage(tokens, nyt_model.wv) for name, tokens in [("train", train_tokens), ("validation", val_tokens), ("test", test_tokens)]}
        print(f"word2vec_nyt | test acc={results['word2vec_nyt']['test']['accuracy']:.4f} f1={results['word2vec_nyt']['test']['macro_f1']:.4f}")

    payload = {"task": "Task 2 - Word Embeddings", "seed": args.seed, "dimension": args.dimension, "epochs": args.epochs, "split": split, "run": run_metadata(args), "results": results}
    write_json(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nyt", type=Path, default=Path("HW-1/nyt.csv"))
    parser.add_argument("--ag", type=Path, default=Path("HW-1/ag.csv"))
    parser.add_argument("--glove-path", type=Path, default=Path("artifacts/glove.6B.100d.txt"))
    parser.add_argument("--method", choices=["all", "glove", "ag-word2vec", "nyt-word2vec"], default="all")
    parser.add_argument("--output", type=Path, default=Path("results/task2_embeddings.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/task2_embeddings"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1, help="Use one worker for deterministic Word2Vec training")
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
