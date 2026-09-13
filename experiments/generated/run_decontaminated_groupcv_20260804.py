"""Leakage-robust classical baseline evaluation for the microbiome project.

This script is intentionally separate from existing experiments. It keeps
identical genus-sequence inputs in the same validation fold and writes all
artifacts to a new dated result directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier


ROOT = Path("/hd/liujx/microbiome_llm_project")
DATA = ROOT / "data/qiita_ibd/clean_2538"
OUT = ROOT / "experiments/results/decontaminated_groupcv_classical_20260804"
VOCAB_SIZE = 1226
SEEDS = [42, 123, 456]


def sequence_hash(sequence: np.ndarray, mask: np.ndarray) -> str:
    packed = np.ascontiguousarray(np.column_stack((sequence, mask)))
    return hashlib.sha256(packed.tobytes()).hexdigest()


def make_presence_features(sequences: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Match the existing project's multi-hot genus-ID baseline representation."""
    features = np.zeros((len(sequences), VOCAB_SIZE), dtype=np.float32)
    for row_index, sequence in enumerate(sequences):
        genus_ids = sequence[masks[row_index] & (sequence > 0)]
        features[row_index, np.unique(genus_ids)] = 1.0
    return features


def score_model(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing result directory: {OUT}")
    OUT.mkdir(parents=True)

    train_rows = [json.loads(line) for line in (DATA / "train_nl.jsonl").read_text().splitlines()]
    test_rows = [json.loads(line) for line in (DATA / "test_nl.jsonl").read_text().splitlines()]
    rows = train_rows + test_rows
    sequences = np.concatenate(
        [np.load(DATA / "train_genus_sequences.npy"), np.load(DATA / "test_genus_sequences.npy")]
    )
    masks = np.concatenate(
        [np.load(DATA / "train_genus_masks.npy"), np.load(DATA / "test_genus_masks.npy")]
    )
    labels = np.array([int(row["label"] == "Disease") for row in rows], dtype=int)
    sample_ids = [row["sample_id"] for row in rows]
    groups = np.array([sequence_hash(sequence, mask) for sequence, mask in zip(sequences, masks)])
    features = make_presence_features(sequences, masks)

    models = {
        "LogisticRegression": lambda seed: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=seed),
        ),
        "LinearSVM": lambda seed: make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, class_weight="balanced", dual=False, max_iter=5000, random_state=seed),
        ),
        "RBFSVM": lambda seed: make_pipeline(
            StandardScaler(),
            SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed),
        ),
        "RandomForest": lambda seed: RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            class_weight="balanced",
            n_jobs=4,
            random_state=seed,
        ),
        "ExtraTrees": lambda seed: ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            n_jobs=4,
            random_state=seed,
        ),
        "XGBoost": lambda seed: XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=4,
            random_state=seed,
        ),
        "MLP": lambda seed: make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(256, 128),
                alpha=1e-4,
                learning_rate_init=1e-3,
                early_stopping=True,
                validation_fraction=0.15,
                max_iter=500,
                random_state=seed,
            ),
        ),
        "KNN": lambda seed: make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5, weights="distance", metric="cosine", n_jobs=4),
        ),
        "BernoulliNB": lambda seed: BernoulliNB(alpha=1.0),
    }

    started = time.time()
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_index, valid_index) in enumerate(splitter.split(features, labels, groups), start=1):
            overlap = set(groups[train_index]) & set(groups[valid_index])
            if overlap:
                raise RuntimeError(f"Group leakage: seed={seed}, fold={fold}, overlap={len(overlap)}")

            for model_name, build_model in models.items():
                model = build_model(seed * 100 + fold)
                started_model = time.time()
                model.fit(features[train_index], labels[train_index])
                predictions = model.predict(features[valid_index])
                scores = score_model(model, features[valid_index])
                metric_row = {
                    "model": model_name,
                    "seed": seed,
                    "fold": fold,
                    "n_train": int(len(train_index)),
                    "n_valid": int(len(valid_index)),
                    "n_groups_train": int(len(set(groups[train_index]))),
                    "n_groups_valid": int(len(set(groups[valid_index]))),
                    "accuracy": float(accuracy_score(labels[valid_index], predictions)),
                    "balanced_accuracy": float(balanced_accuracy_score(labels[valid_index], predictions)),
                    "f1": float(f1_score(labels[valid_index], predictions)),
                    "macro_f1": float(f1_score(labels[valid_index], predictions, average="macro")),
                    "auroc": float(roc_auc_score(labels[valid_index], scores)),
                    "auprc": float(average_precision_score(labels[valid_index], scores)),
                    "runtime_s": round(time.time() - started_model, 3),
                }
                metric_rows.append(metric_row)
                for sample_index, true_label, prediction, score in zip(
                    valid_index, labels[valid_index], predictions, scores
                ):
                    prediction_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "fold": fold,
                            "sample_index": int(sample_index),
                            "sample_id": sample_ids[sample_index],
                            "group_hash": groups[sample_index][:16],
                            "true_label": int(true_label),
                            "predicted_label": int(prediction),
                            "score": float(score),
                        }
                    )
                print(
                    f"{model_name:18s} seed={seed} fold={fold} "
                    f"acc={metric_row['accuracy']:.3f} auc={metric_row['auroc']:.3f}",
                    flush=True,
                )

    summary: dict[str, dict[str, dict[str, float]]] = {}
    metrics = ("accuracy", "balanced_accuracy", "f1", "macro_f1", "auroc", "auprc", "runtime_s")
    for model_name in models:
        selected = [row for row in metric_rows if row["model"] == model_name]
        summary[model_name] = {
            metric: {
                "mean": float(np.mean([float(row[metric]) for row in selected])),
                "std": float(np.std([float(row[metric]) for row in selected], ddof=1)),
            }
            for metric in metrics
        }

    unique_groups = set(groups)
    conflicting_groups = sum(
        len(set(labels[np.where(groups == group)[0]])) > 1 for group in unique_groups
    )
    manifest = {
        "experiment": "decontaminated_groupcv_classical_20260804",
        "dataset": "qiita_ibd/clean_2538 (train + test recombined only for cross-validation)",
        "input": "1226-dimensional multi-hot genus presence features derived from sequence IDs; no language text used",
        "protocol": "3 repeats x StratifiedGroupKFold(5), grouping by exact genus_sequence+mask hash",
        "n_samples": int(len(labels)),
        "n_unique_input_groups": int(len(unique_groups)),
        "n_conflicting_label_groups": int(conflicting_groups),
        "seeds": SEEDS,
        "models": list(models),
        "note": "Fixed baseline hyperparameters; this is leakage-robust comparison, not nested-CV hyperparameter optimization.",
        "runtime_s": round(time.time() - started, 2),
        "summary": summary,
    }
    with (OUT / "metrics_by_fold.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    with (OUT / "predictions_by_fold.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    (OUT / "summary.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"SAVED {OUT}", flush=True)


if __name__ == "__main__":
    main()
