"""Controlled representation/head ablation under exact-input grouped CV."""

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
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path("/hd/liujx/microbiome_llm_project")
DATA = ROOT / "data/qiita_ibd/clean_2538"
OUT = ROOT / "experiments/results/decontaminated_groupcv_representation_ablation_v2_20260804"
VOCAB_SIZE = 1226
EMBED_DIM = 64
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


class PresenceLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(VOCAB_SIZE, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class PresenceMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(VOCAB_SIZE, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class EmbeddingLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.classifier = nn.Linear(EMBED_DIM, 2)

    def forward(self, genus_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(genus_ids)
        mask_float = mask.float().unsqueeze(-1)
        pooled = (embedded * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


class EmbeddingMLP(nn.Module):
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
        embedded = self.embedding(genus_ids)
        mask_float = mask.float().unsqueeze(-1)
        pooled = (embedded * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


MODELS = {
    "Presence_Linear": (PresenceLinear, "presence"),
    "Presence_MLP": (PresenceMLP, "presence"),
    "Embedding64_Linear": (EmbeddingLinear, "embedding"),
    "Embedding64_MLP": (EmbeddingMLP, "embedding"),
}


def make_presence_features(sequences: np.ndarray, masks: np.ndarray) -> np.ndarray:
    features = np.zeros((len(sequences), VOCAB_SIZE), dtype=np.float32)
    for row, (sequence, mask) in enumerate(zip(sequences, masks)):
        features[row, np.unique(sequence[mask & (sequence >= 0)])] = 1.0
    return features


def train_fold(model_name: str, model_type: str, sequences: np.ndarray, masks: np.ndarray, features: np.ndarray, labels: np.ndarray, train_index: np.ndarray, valid_index: np.ndarray, seed: int):
    set_seed(seed)
    model = MODELS[model_name][0]().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    if model_type == "presence":
        train_data = TensorDataset(torch.from_numpy(features[train_index]), torch.from_numpy(labels[train_index]))
        valid_data = TensorDataset(torch.from_numpy(features[valid_index]), torch.from_numpy(labels[valid_index]))
    else:
        train_data = TensorDataset(torch.from_numpy(sequences[train_index]).long(), torch.from_numpy(masks[train_index]).bool(), torch.from_numpy(labels[train_index]))
        valid_data = TensorDataset(torch.from_numpy(sequences[valid_index]).long(), torch.from_numpy(masks[valid_index]).bool(), torch.from_numpy(labels[valid_index]))
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    started = time.time()
    model.train()
    for _ in range(EPOCHS):
        for batch in train_loader:
            if model_type == "presence":
                inputs, target = (item.to(DEVICE) for item in batch)
                logits = model(inputs)
            else:
                genus_ids, mask, target = (item.to(DEVICE) for item in batch)
                logits = model(genus_ids, mask)
            weights = torch.where(target == 1, 1.5, 1.0)
            loss = (functional.cross_entropy(logits, target, reduction="none") * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    model.eval()
    target_values, predicted, scores = [], [], []
    with torch.no_grad():
        for batch in valid_loader:
            if model_type == "presence":
                inputs, target = (item.to(DEVICE) for item in batch)
                logits = model(inputs)
            else:
                genus_ids, mask, target = (item.to(DEVICE) for item in batch)
                logits = model(genus_ids, mask)
            probability = torch.softmax(logits, dim=1)[:, 1]
            target_values.extend(target.cpu().numpy().tolist())
            predicted.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            scores.extend(probability.cpu().numpy().tolist())
    params = sum(parameter.numel() for parameter in model.parameters())
    runtime = time.time() - started
    del model
    torch.cuda.empty_cache()
    return np.array(target_values), np.array(predicted), np.array(scores), params, runtime


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing result directory: {OUT}")
    OUT.mkdir(parents=True)
    rows = [json.loads(line) for filename in ("train_nl.jsonl", "test_nl.jsonl") for line in (DATA / filename).read_text().splitlines()]
    sequences = np.concatenate([np.load(DATA / "train_genus_sequences.npy"), np.load(DATA / "test_genus_sequences.npy")])
    masks = np.concatenate([np.load(DATA / "train_genus_masks.npy"), np.load(DATA / "test_genus_masks.npy")])
    labels = np.array([int(row["label"] == "Disease") for row in rows], dtype=np.int64)
    sample_ids = [row["sample_id"] for row in rows]
    groups = np.array([sequence_hash(sequence, mask) for sequence, mask in zip(sequences, masks)])
    features = make_presence_features(sequences, masks)
    metrics, predictions = [], []
    for seed in SEEDS:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_index, valid_index) in enumerate(splitter.split(features, labels, groups), start=1):
            if set(groups[train_index]) & set(groups[valid_index]):
                raise RuntimeError(f"Group leakage in seed={seed}, fold={fold}")
            for model_name, (_, model_type) in MODELS.items():
                # Match the established embedding-dimension experiment's fold seed exactly.
                target, predicted, scores, params, runtime = train_fold(model_name, model_type, sequences, masks, features, labels, train_index, valid_index, seed * 100 + fold)
                metric = {
                    "model": model_name, "seed": seed, "fold": fold, "n_train": int(len(train_index)), "n_valid": int(len(valid_index)),
                    "accuracy": float(accuracy_score(target, predicted)), "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
                    "f1": float(f1_score(target, predicted)), "macro_f1": float(f1_score(target, predicted, average="macro")),
                    "auroc": float(roc_auc_score(target, scores)), "auprc": float(average_precision_score(target, scores)),
                    "runtime_s": float(runtime), "n_params": int(params),
                }
                metrics.append(metric)
                for index, truth, pred, score in zip(valid_index, target, predicted, scores):
                    predictions.append({"model": model_name, "seed": seed, "fold": fold, "sample_index": int(index), "sample_id": sample_ids[index], "true_label": int(truth), "predicted_label": int(pred), "score": float(score)})
                print(f"{model_name} seed={seed} fold={fold} auc={metric['auroc']:.3f}", flush=True)
    fields = list(metrics[0])
    with (OUT / "metrics_by_fold.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(metrics)
    with (OUT / "predictions_by_fold.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    summary = {}
    for model_name in MODELS:
        model_rows = [row for row in metrics if row["model"] == model_name]
        summary[model_name] = {key: {"mean": float(np.mean([row[key] for row in model_rows])), "std": float(np.std([row[key] for row in model_rows], ddof=1))} for key in ("accuracy", "balanced_accuracy", "f1", "macro_f1", "auroc", "auprc", "runtime_s", "n_params")}
    (OUT / "summary.json").write_text(json.dumps({"experiment": "representation_and_head_ablation", "protocol": "3 repeats x StratifiedGroupKFold(5), grouped by exact genus_sequence+mask", "models": {"Presence_Linear": "multi-hot token presence -> linear classifier", "Presence_MLP": "multi-hot token presence -> MLP(256, BN, ReLU, Dropout=0.3)", "Embedding64_Linear": "trainable E=64 embedding -> masked mean pooling -> linear classifier", "Embedding64_MLP": "trainable E=64 embedding -> masked mean pooling -> MLP(256, BN, ReLU, Dropout=0.3)"}, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
