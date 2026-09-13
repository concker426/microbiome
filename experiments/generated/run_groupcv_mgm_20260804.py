"""Group-aware evaluation of the existing pretrained MGM encoder features."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/hd/liujx/microbiome_llm_project")
DATA = ROOT / "data/qiita_ibd/clean_2538"
CHECKPOINT = ROOT / "saved_models/mgm_encoder_pretrained/mgm_encoder.pt"
OUT = ROOT / "experiments/results/decontaminated_groupcv_mgm_20260804"
SEEDS = [42, 123, 456]
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda:0")

sys.path.insert(0, str(ROOT))
from run_v6_merged import MGMEnc  # noqa: E402


def sequence_hash(sequence: np.ndarray, mask: np.ndarray) -> str:
    packed = np.ascontiguousarray(np.column_stack((sequence, mask)))
    return hashlib.sha256(packed.tobytes()).hexdigest()


def extract_features(sequences: np.ndarray, masks: np.ndarray) -> np.ndarray:
    encoder = MGMEnc()
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    encoder.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    encoder.to(DEVICE).eval()
    extracted = []
    with torch.no_grad():
        for start in range(0, len(sequences), 64):
            ids = torch.from_numpy(sequences[start : start + 64].astype(np.int64)).long().to(DEVICE)
            mask = torch.from_numpy(masks[start : start + 64]).bool().to(DEVICE)
            extracted.append(encoder(ids, mask).cpu().numpy())
    del encoder
    torch.cuda.empty_cache()
    return np.concatenate(extracted)


def predict_score(model, features: np.ndarray) -> np.ndarray:
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

    started = time.time()
    features = extract_features(sequences, masks)
    models = {
        "MGM_LogisticRegression": lambda seed: make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=seed)
        ),
        "MGM_MLP": lambda seed: make_pipeline(
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
        "MGM_RandomForest": lambda seed: RandomForestClassifier(
            n_estimators=300, max_depth=15, class_weight="balanced", n_jobs=4, random_state=seed
        ),
    }
    metrics: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for seed in SEEDS:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_index, valid_index) in enumerate(splitter.split(features, labels, groups), start=1):
            if set(groups[train_index]) & set(groups[valid_index]):
                raise RuntimeError(f"Group leakage in seed={seed}, fold={fold}")
            for model_name, build_model in models.items():
                model = build_model(seed * 100 + fold)
                model_started = time.time()
                model.fit(features[train_index], labels[train_index])
                predicted = model.predict(features[valid_index])
                scores = predict_score(model, features[valid_index])
                row = {
                    "model": model_name,
                    "seed": seed,
                    "fold": fold,
                    "n_train": int(len(train_index)),
                    "n_valid": int(len(valid_index)),
                    "accuracy": float(accuracy_score(labels[valid_index], predicted)),
                    "balanced_accuracy": float(balanced_accuracy_score(labels[valid_index], predicted)),
                    "f1": float(f1_score(labels[valid_index], predicted)),
                    "macro_f1": float(f1_score(labels[valid_index], predicted, average="macro")),
                    "auroc": float(roc_auc_score(labels[valid_index], scores)),
                    "auprc": float(average_precision_score(labels[valid_index], scores)),
                    "runtime_s": round(time.time() - model_started, 3),
                }
                metrics.append(row)
                for sample_index, true_label, predicted_label, score in zip(
                    valid_index, labels[valid_index], predicted, scores
                ):
                    predictions.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "fold": fold,
                            "sample_index": int(sample_index),
                            "sample_id": sample_ids[sample_index],
                            "group_hash": groups[sample_index][:16],
                            "true_label": int(true_label),
                            "predicted_label": int(predicted_label),
                            "score": float(score),
                        }
                    )
                print(
                    f"{model_name:24s} seed={seed} fold={fold} "
                    f"acc={row['accuracy']:.3f} auc={row['auroc']:.3f}",
                    flush=True,
                )

    summary = {}
    keys = ("accuracy", "balanced_accuracy", "f1", "macro_f1", "auroc", "auprc", "runtime_s")
    for model_name in models:
        selected = [row for row in metrics if row["model"] == model_name]
        summary[model_name] = {
            key: {
                "mean": float(np.mean([float(row[key]) for row in selected])),
                "std": float(np.std([float(row[key]) for row in selected], ddof=1)),
            }
            for key in keys
        }
    manifest = {
        "experiment": "decontaminated_groupcv_mgm_20260804",
        "encoder": "Existing pretrained MGMEnc checkpoint (6-layer Transformer, 768-dim output)",
        "checkpoint": str(CHECKPOINT),
        "protocol": "3 repeats x StratifiedGroupKFold(5), grouping by exact genus_sequence+mask hash",
        "device": str(DEVICE),
        "n_samples": int(len(labels)),
        "n_unique_input_groups": int(len(set(groups))),
        "models": list(models),
        "summary": summary,
        "runtime_s": round(time.time() - started, 2),
    }
    with (OUT / "metrics_by_fold.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (OUT / "predictions_by_fold.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    (OUT / "summary.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"SAVED {OUT}", flush=True)


if __name__ == "__main__":
    main()
