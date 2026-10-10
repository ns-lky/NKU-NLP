"""Task 1: compare binary, frequency, and TF-IDF bag-of-words models."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.common import classification_metrics, load_nyt, set_seed, split_nyt, tokenize, write_json


def build_vectorizers(max_features: int | None) -> OrderedDict[str, object]:
    common = {
        "tokenizer": tokenize,
        "preprocessor": None,
        "token_pattern": None,
        "lowercase": False,
        "max_features": max_features,
    }
    return OrderedDict(
        [
            ("binary", CountVectorizer(binary=True, **common)),
            ("frequency", CountVectorizer(binary=False, **common)),
            ("tfidf", TfidfVectorizer(**common)),
        ]
    )


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    frame = load_nyt(args.nyt)
    train, validation, test = split_nyt(frame, args.seed)
    results: dict[str, dict] = {}

    for name, vectorizer in build_vectorizers(args.max_features).items():
        x_train = vectorizer.fit_transform(train["text"])
        x_validation = vectorizer.transform(validation["text"])
        x_test = vectorizer.transform(test["text"])
        classifier = LogisticRegression(
            max_iter=args.max_iter,
            solver="liblinear",
            random_state=args.seed,
        )
        classifier.fit(x_train, train["label"])
        validation_metrics = classification_metrics(
            validation["label"], classifier.predict(x_validation)
        )
        test_metrics = classification_metrics(test["label"], classifier.predict(x_test))
        results[name] = {
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
            "train_documents": int(len(train)),
            "validation": validation_metrics,
            "test": test_metrics,
        }
        print(
            f"{name:>9} | vocab={results[name]['vocabulary_size']:>7} | "
            f"val acc={validation_metrics['accuracy']:.4f} f1={validation_metrics['macro_f1']:.4f} | "
            f"test acc={test_metrics['accuracy']:.4f} f1={test_metrics['macro_f1']:.4f}"
        )

    payload = {
        "task": "Task 1 - Bag of Words",
        "seed": args.seed,
        "split": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "results": results,
    }
    write_json(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nyt", type=Path, default=Path("HW-1/nyt.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/task1_bow.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--max-features", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
