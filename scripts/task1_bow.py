"""Task 1: compare binary, frequency, and TF-IDF bag-of-words models."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.common import prepare_nyt, run_metadata, save_evaluation, set_seed, tokenize, write_json


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
    frame, train, validation, test, split = prepare_nyt(args.nyt, args.seed)
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
        validation_metrics = save_evaluation(
            validation, classifier.predict(x_validation),
            args.output.parent / "predictions" / f"task1_{name}_validation.csv",
        )
        test_metrics = save_evaluation(
            test, classifier.predict(x_test),
            args.output.parent / "predictions" / f"task1_{name}_test.csv",
        )
        args.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": vectorizer, "classifier": classifier}, args.model_dir / f"{name}.joblib")
        results[name] = {
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
            "train_documents": int(len(train)),
            "iterations": classifier.n_iter_.tolist(),
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
        "split": split,
        "run": run_metadata(args),
        "results": results,
    }
    write_json(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nyt", type=Path, default=Path("HW-1/nyt.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/task1_bow.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/task1_bow"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--max-features", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
