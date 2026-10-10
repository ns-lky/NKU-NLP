"""Independently recompute every metric from the saved full-dataset predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.common import load_nyt, split_nyt, write_json


def verify(nyt: Path, directory: Path):
    frame = load_nyt(nyt)
    split = json.loads((directory / "split.json").read_text(encoding="utf-8"))
    parts = dict(zip(["train", "validation", "test"], split_nyt(frame, split["seed"])))
    fingerprint = hashlib.sha256(json.dumps(split["row_indices"], separators=(",", ":")).encode()).hexdigest()
    assert split["dataset_sha256"] == frame.attrs["sha256"], "Dataset hash mismatch"
    assert split["split_sha256"] == fingerprint, "Split fingerprint mismatch"
    for name, part in parts.items():
        assert part.index.tolist() == split["row_indices"][name], f"Changed {name} split"
    files = {
        "task1": directory / "task1_bow.json",
        "task2": directory / "task2_embeddings.json",
        "task3": directory / "task3_bert.json",
    }
    summary = []
    for task, path in files.items():
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["split"]["split_sha256"] == fingerprint, f"Split mismatch in {task}"
        assert result["split"]["dataset_sha256"] == frame.attrs["sha256"]
        if task == "task3":
            assert result["model_name"] == "google-bert/bert-base-uncased"
            assert result["max_length"] == 64 and result["epochs"] == 3
            assert [epoch["epoch"] for epoch in result["history"]] == [1, 2, 3]
            assert all(epoch["training_examples"] == len(parts["train"]) for epoch in result["history"])
            assert result["best_epoch"] == max(result["history"], key=lambda item: item["macro_f1"])["epoch"]
            methods = {"bert": result}
        else:
            methods = result["results"]
            if task == "task1":
                assert set(methods) == {"binary", "frequency", "tfidf"}
            else:
                assert set(methods) == {"glove", "word2vec_ag", "word2vec_nyt"}
                assert result["dimension"] == 100
                assert methods["word2vec_ag"]["training_documents"] == 90000
                assert methods["word2vec_nyt"]["training_documents"] == len(parts["train"])
        for method, measurements in methods.items():
            for partition in ["validation", "test"]:
                predictions = pd.read_csv(directory / "predictions" / f"{task}_{method}_{partition}.csv")
                part = parts[partition]
                assert predictions.row_index.tolist() == part.index.tolist(), f"Changed prediction order: {task}/{method}/{partition}"
                assert predictions.label.tolist() == part.label.tolist(), "Incorrect true labels"
                assert set(predictions.prediction).issubset(set(frame.label))
                metrics = measurements[partition]
                assert abs(accuracy_score(predictions.label, predictions.prediction) - metrics["accuracy"]) < 1e-12
                assert abs(f1_score(predictions.label, predictions.prediction, average="macro") - metrics["macro_f1"]) < 1e-12
                matrix = confusion_matrix(predictions.label, predictions.prediction, labels=metrics["labels"])
                assert np.array_equal(matrix, metrics["confusion_matrix"])
            summary.append({"task": task, "method": method, "test_documents": len(parts["test"]),
                            "accuracy": measurements["test"]["accuracy"], "macro_f1": measurements["test"]["macro_f1"]})
    assert len(summary) == 7
    pd.DataFrame(summary).to_csv(directory / "metrics_summary.csv", index=False)
    write_json(directory / "verification.json", {
        "all_checks_passed": True, "models_checked": len(summary),
        "prediction_rows_checked": len(summary) * (len(parts["validation"]) + len(parts["test"])),
        "dataset_sha256": frame.attrs["sha256"], "split_sha256": fingerprint,
        "checks": ["exact ordered 80/10/10 split", "true labels", "full test/validation predictions",
                   "independent Accuracy/Macro-F1/confusion-matrix recomputation", "100-dimensional embeddings",
                   "full AG/NYT Word2Vec training corpora", "specified BERT with length 64 and three full epochs",
                   "model selection using validation Macro-F1"],
    })
    print(pd.DataFrame(summary).to_string(index=False))
    print("All seven full experiments verified against saved row-level predictions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nyt", type=Path, default=Path("HW-1/nyt.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    verify(args.nyt, args.results_dir)
