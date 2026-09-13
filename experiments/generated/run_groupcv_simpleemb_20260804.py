"""Group-aware CV for the project's SimpleEmb + MLP classification model."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset


ROOT = Path("/hd/liujx/microbiome_llm_project")
DATA = ROOT / "data/qiita_ibd/clean_2538"
OUT = ROOT / "experiments/results/decontaminated_groupcv_simpleemb_mlp_20260804"
VOCAB_SIZE = 1226
EMBED_DIM = 512
HIDDEN_DIM = 256
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEEDS = [42, 123, 456]
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda:0")


def sequence_hash(sequence: np.ndarray, mask: np.ndarray) -> str:
    packed = np.ascontiguousarray(np.column_stack((sequence, mask)))
    return hashlib.sha256(packed.tobytes()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, masks: np.ndarray, labels: np.ndarray) -> None:
        self.sequences = sequences
        self.masks = masks
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            torch.tensor(self.sequences[index], dtype=torch.long),
            torch.tensor(self.masks[index], dtype=torch.bool),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


class SimpleEmbeddingMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM, 2),
        )

    def forward(self, genus_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(genus_ids)
        mask_float = mask.float().unsqueeze(-1)
        pooled = (embeddings * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


def train_and_evaluate(
    sequences: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    seed: int,
):
    set_seed(seed)
    model = SimpleEmbeddingMLP().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    train_loader = DataLoader(
        SequenceDataset(sequences[train_index], masks[train_index], labels[train_index]),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        SequenceDataset(sequences[valid_index], masks[valid_index], labels[valid_index]),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    started = time.time()
    model.train()
    for _ in range(EPOCHS):
        for genus_ids, mask, target in train_loader:
            genus_ids, mask, target = genus_ids.to(DEVICE), mask.to(DEVICE), target.to(DEVICE)
            logits = model(genus_ids, mask)
            sample_weight = torch.where(target == 1, 1.5, 1.0)
            loss = (functional.cross_entropy(logits, target, reduction="none") * sample_weight).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    predictions, scores, targets = [], [], []
    model.eval()
    with torch.no_grad():
        for genus_ids, mask, target in valid_loader:
            logits = model(genus_ids.to(DEVICE), mask.to(DEVICE))
            probability = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            scores.extend(probability.tolist())
            targets.extend(target.numpy().tolist())
    params = sum(parameter.numel() for parameter in model.parameters())
    runtime = time.time() - started
    del model
    torch.cuda.empty_cache()
    return np.array(targets), np.array(predictions), np.array(scores), params, runtime


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

    metrics, predictions = [], []
    started = time.time()
    for seed in SEEDS:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_index, valid_index) in enumerate(splitter.split(sequences, labels, groups), start=1):
            if set(groups[train_index]) & set(groups[valid_index]):
                raise RuntimeError(f"Group leakage in seed={seed}, fold={fold}")
            target, predicted, scores, params, runtime = train_and_evaluate(
                sequences, masks, labels, train_index, valid_index, seed * 100 + fold
            )
            row = {
                "model": "SimpleEmb_MLP",
                "seed": seed,
                "fold": fold,
                "n_train": int(len(train_index)),
                "n_valid": int(len(valid_index)),
                "accuracy": float(accuracy_score(target, predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
                "f1": float(f1_score(target, predicted)),
                "macro_f1": float(f1_score(target, predicted, average="macro")),
                "auroc": float(roc_auc_score(target, scores)),
                "auprc": float(average_precision_score(target, scores)),
                "runtime_s": round(runtime, 3),
                "n_params": int(params),
            }
            metrics.append(row)
            for local_index, true_label, predicted_label, score in zip(valid_index, target, predicted, scores):
                predictions.append(
                    {
                        "model": "SimpleEmb_MLP",
                        "seed": seed,
                        "fold": fold,
                        "sample_index": int(local_index),
                        "sample_id": sample_ids[local_index],
                        "group_hash": groups[local_index][:16],
                        "true_label": int(true_label),
                        "predicted_label": int(predicted_label),
                        "score": float(score),
                    }
                )
            print(
                f"SimpleEmb_MLP seed={seed} fold={fold} "
                f"acc={row['accuracy']:.3f} auc={row['auroc']:.3f}",
                flush=True,
            )

    keys = ("accuracy", "balanced_accuracy", "f1", "macro_f1", "auroc", "auprc", "runtime_s")
    summary = {
        key: {"mean": float(np.mean([row[key] for row in metrics])), "std": float(np.std([row[key] for row in metrics], ddof=1))}
        for key in keys
    }
    manifest = {
        "experiment": "decontaminated_groupcv_simpleemb_mlp_20260804",
        "architecture": "trainable genus embedding (E=512) + masked mean pooling + MLP(256, BatchNorm, ReLU, Dropout=0.3)",
        "protocol": "3 repeats x StratifiedGroupKFold(5), grouping by exact genus_sequence+mask hash",
        "device": str(DEVICE),
        "n_samples": int(len(labels)),
        "n_unique_input_groups": int(len(set(groups))),
        "seeds": SEEDS,
        "epochs": EPOCHS,
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
