"""Describe OOF performance by input sparsity and probability calibration."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, roc_auc_score


ROOT = Path("/hd/liujx/microbiome_llm_project")
DATA = ROOT / "data/qiita_ibd/clean_2538"
OUT = ROOT / "experiments/results/decontaminated_groupcv_strata_calibration_20260804"
PREDICTIONS = {
    "HistGradientBoosting": ROOT / "experiments/results/decontaminated_groupcv_classical_no_xgboost_20260804/predictions_by_fold.csv",
    "Presence_MLP": ROOT / "experiments/results/decontaminated_groupcv_representation_ablation_v2_20260804/predictions_by_fold.csv",
    "Embedding64_MLP": ROOT / "experiments/results/decontaminated_groupcv_representation_ablation_v2_20260804/predictions_by_fold.csv",
}


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (scores >= low) & ((scores < high) if high < 1 else (scores <= high))
        if selected.any():
            ece += selected.mean() * abs(scores[selected].mean() - labels[selected].mean())
    return float(ece)


def bucket(length: int) -> str:
    if length <= 4:
        return "1-4"
    if length <= 19:
        return "5-19"
    if length <= 49:
        return "20-49"
    return "50+"


def read_predictions(model: str, path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required_name = {"HistGradientBoosting": "HistGradientBoosting", "Presence_MLP": "Presence_MLP", "Embedding64_MLP": "Embedding64_MLP"}[model]
    return [
        {"sample_index": int(row["sample_index"]), "truth": int(row["true_label"]), "prediction": int(row["predicted_label"]), "score": float(row["score"])}
        for row in rows if row["model"] == required_name
    ]


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing result directory: {OUT}")
    OUT.mkdir(parents=True)
    masks = np.concatenate([np.load(DATA / "train_genus_masks.npy"), np.load(DATA / "test_genus_masks.npy")])
    token_counts = masks.sum(axis=1).astype(int)
    summaries, calibration = [], []
    for model, path in PREDICTIONS.items():
        rows = read_predictions(model, path)
        assert len(rows) == 3 * len(token_counts), (model, len(rows))
        for name in ("1-4", "5-19", "20-49", "50+"):
            subset = [row for row in rows if bucket(token_counts[int(row["sample_index"])]) == name]
            if not subset:
                continue
            truth = np.array([row["truth"] for row in subset])
            predicted = np.array([row["prediction"] for row in subset])
            scores = np.array([row["score"] for row in subset])
            summaries.append({
                "model": model, "token_count_bucket": name, "n_oof_predictions": len(subset), "n_unique_samples": len(subset) // 3,
                "disease_prevalence": float(truth.mean()), "accuracy": float(accuracy_score(truth, predicted)),
                "macro_f1": float(f1_score(truth, predicted, average="macro")),
                "auroc": float(roc_auc_score(truth, scores)) if len(np.unique(truth)) == 2 else None,
                "auprc": float(average_precision_score(truth, scores)) if len(np.unique(truth)) == 2 else None,
                "brier": float(brier_score_loss(truth, scores)), "ece_10": expected_calibration_error(truth, scores),
            })
        truth = np.array([row["truth"] for row in rows])
        scores = np.array([row["score"] for row in rows])
        calibration.append({"model": model, "n_oof_predictions": len(rows), "brier": float(brier_score_loss(truth, scores)), "ece_10": expected_calibration_error(truth, scores)})
    with (OUT / "sparsity_strata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    with (OUT / "calibration_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(calibration[0])); writer.writeheader(); writer.writerows(calibration)
    (OUT / "summary.json").write_text(json.dumps({"protocol": "aggregate OOF predictions from three exact-input grouped 5-fold repeats; descriptive strata only", "bucket_definition": {"1-4": "1 to 4 active tokens", "5-19": "5 to 19", "20-49": "20 to 49", "50+": "50 or more"}, "sparsity_strata": summaries, "calibration": calibration}, indent=2))
    print(json.dumps({"sparsity_strata": summaries, "calibration": calibration}, indent=2), flush=True)


if __name__ == "__main__":
    main()
