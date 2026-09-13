"""Nested grouped-CV learning curves for the strongest representation variants."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit


ROOT = Path("/hd/liujx/microbiome_llm_project")
DATA = ROOT / "data/qiita_ibd/clean_2538"
REPRESENTATION_SCRIPT = ROOT / "experiments/generated/run_groupcv_representation_ablation_v2_20260804.py"
OUT = ROOT / "experiments/results/decontaminated_groupcv_learning_curve_20260804"
FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS = [42, 123, 456]


def sequence_hash(sequence: np.ndarray, mask: np.ndarray) -> str:
    packed = np.ascontiguousarray(np.column_stack((sequence, mask)))
    return hashlib.sha256(packed.tobytes()).hexdigest()


def load_representation_module():
    spec = importlib.util.spec_from_file_location("representation_ablation", REPRESENTATION_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {REPRESENTATION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def group_labels(groups: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups)
    # Duplicate exact inputs occasionally have inconsistent labels. Majority labels are used only for stratifying groups.
    labels_per_group = np.array([int(labels[groups == group].mean() >= 0.5) for group in unique_groups])
    return unique_groups, labels_per_group


def select_train_indices(train_index: np.ndarray, groups: np.ndarray, labels: np.ndarray, fraction: float, random_state: int) -> np.ndarray:
    if fraction == 1.0:
        return train_index
    unique_groups, labels_per_group = group_labels(groups[train_index], labels[train_index])
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=random_state)
    selected_group_index, _ = next(splitter.split(unique_groups, labels_per_group))
    selected_groups = set(unique_groups[selected_group_index])
    selected = np.array([index for index in train_index if groups[index] in selected_groups], dtype=int)
    if set(groups[selected]) & set(groups[np.setdiff1d(np.arange(len(groups)), selected)]):
        # Overlap with the held-out portion is expected, but not with validation and is checked by caller.
        pass
    return selected


def hgb(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1e-3, random_state=seed)


def evaluate_hgb(features: np.ndarray, labels: np.ndarray, train_index: np.ndarray, valid_index: np.ndarray, seed: int):
    model = hgb(seed)
    started = time.time()
    model.fit(features[train_index], labels[train_index])
    predicted = model.predict(features[valid_index])
    scores = model.predict_proba(features[valid_index])[:, 1]
    # Tree ensembles have no parameter count comparable to neural networks.
    return labels[valid_index], predicted, scores, time.time() - started, 0


def metrics_row(model: str, fraction: float, seed: int, fold: int, n_train: int, target: np.ndarray, predicted: np.ndarray, scores: np.ndarray, runtime: float, n_params: int) -> dict[str, object]:
    return {
        "model": model, "train_fraction": fraction, "seed": seed, "fold": fold, "n_train": n_train, "n_valid": int(len(target)),
        "accuracy": float(accuracy_score(target, predicted)), "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "f1": float(f1_score(target, predicted)), "macro_f1": float(f1_score(target, predicted, average="macro")),
        "auroc": float(roc_auc_score(target, scores)), "auprc": float(average_precision_score(target, scores)), "runtime_s": float(runtime), "n_params": int(n_params),
    }


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing result directory: {OUT}")
    OUT.mkdir(parents=True)
    rep = load_representation_module()
    rows = [json.loads(line) for filename in ("train_nl.jsonl", "test_nl.jsonl") for line in (DATA / filename).read_text().splitlines()]
    sequences = np.concatenate([np.load(DATA / "train_genus_sequences.npy"), np.load(DATA / "test_genus_sequences.npy")])
    masks = np.concatenate([np.load(DATA / "train_genus_masks.npy"), np.load(DATA / "test_genus_masks.npy")])
    labels = np.array([int(row["label"] == "Disease") for row in rows], dtype=np.int64)
    groups = np.array([sequence_hash(sequence, mask) for sequence, mask in zip(sequences, masks)])
    features = rep.make_presence_features(sequences, masks)
    metrics: list[dict[str, object]] = []
    for seed in SEEDS:
        outer_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_index, valid_index) in enumerate(outer_splitter.split(features, labels, groups), start=1):
            if set(groups[train_index]) & set(groups[valid_index]):
                raise RuntimeError(f"Group leakage in outer seed={seed}, fold={fold}")
            for fraction in FRACTIONS:
                subset = select_train_indices(train_index, groups, labels, fraction, random_state=seed * 100 + fold)
                if set(groups[subset]) & set(groups[valid_index]):
                    raise RuntimeError(f"Group leakage at fraction={fraction}, seed={seed}, fold={fold}")
                target, predicted, scores, runtime, leaves = evaluate_hgb(features, labels, subset, valid_index, seed * 100 + fold)
                metrics.append(metrics_row("HistGradientBoosting", fraction, seed, fold, len(subset), target, predicted, scores, runtime, leaves))
                for model_name, model_type in (("Presence_MLP", "presence"), ("Embedding64_MLP", "embedding")):
                    target, predicted, scores, params, runtime = rep.train_fold(model_name, model_type, sequences, masks, features, labels, subset, valid_index, seed * 100 + fold)
                    metrics.append(metrics_row(model_name, fraction, seed, fold, len(subset), target, predicted, scores, runtime, params))
                print(f"seed={seed} fold={fold} fraction={fraction} n_train={len(subset)}", flush=True)
    with (OUT / "metrics_by_fold.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    summary = []
    for model in ("HistGradientBoosting", "Presence_MLP", "Embedding64_MLP"):
        for fraction in FRACTIONS:
            subset = [row for row in metrics if row["model"] == model and row["train_fraction"] == fraction]
            summary.append({"model": model, "train_fraction": fraction, "mean_n_train": float(np.mean([row["n_train"] for row in subset])), **{f"{metric}_{stat}": float(getattr(np, stat)([row[metric] for row in subset], ddof=1) if stat == "std" else getattr(np, stat)([row[metric] for row in subset])) for metric in ("accuracy", "macro_f1", "auroc", "auprc") for stat in ("mean", "std")}})
    with (OUT / "summary_by_fraction.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    (OUT / "manifest.json").write_text(json.dumps({"experiment": "nested_grouped_cv_learning_curve", "outer_protocol": "3 repeats x StratifiedGroupKFold(5), exact genus_sequence+mask groups", "inner_protocol": "stratified sampling of whole training groups only", "fractions": FRACTIONS, "models": ["HistGradientBoosting", "Presence_MLP", "Embedding64_MLP"]}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
