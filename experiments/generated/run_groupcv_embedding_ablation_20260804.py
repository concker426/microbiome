"""Embedding-dimension ablation under the already-established grouped CV protocol."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path("/hd/liujx/microbiome_llm_project")
BASE_SCRIPT = ROOT / "experiments/generated/run_groupcv_simpleemb_20260804.py"
OUT = ROOT / "experiments/results/decontaminated_groupcv_embedding_dim_ablation_20260804"
EMBED_DIMS = [16, 32, 64, 128, 256, 512, 768]
METRICS = ("accuracy", "balanced_accuracy", "f1", "macro_f1", "auroc", "auprc", "runtime_s", "n_params")


def load_base_module():
    spec = importlib.util.spec_from_file_location("simpleemb_groupcv", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing result directory: {OUT}")
    OUT.mkdir(parents=True)
    base = load_base_module()
    all_metrics: list[dict[str, object]] = []

    for dimension in EMBED_DIMS:
        dimension_out = OUT / f"E{dimension}"
        base.EMBED_DIM = dimension
        base.OUT = dimension_out
        print(f"START E={dimension}", flush=True)
        base.main()

        manifest_path = dimension_out / "summary.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["experiment"] = f"decontaminated_groupcv_simpleemb_E{dimension}_20260804"
        manifest["architecture"] = (
            f"trainable genus embedding (E={dimension}) + masked mean pooling + "
            "MLP(256, BatchNorm, ReLU, Dropout=0.3)"
        )
        manifest["ablation_variable"] = "embedding dimension only; all other model and training settings fixed"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        with (dimension_out / "metrics_by_fold.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                row["embedding_dim"] = dimension
                all_metrics.append(row)

    with (OUT / "metrics_by_fold.csv").open("w", newline="") as handle:
        fields = ["embedding_dim"] + [key for key in all_metrics[0] if key != "embedding_dim"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_metrics)

    summary: list[dict[str, object]] = []
    for dimension in EMBED_DIMS:
        rows = [row for row in all_metrics if int(row["embedding_dim"]) == dimension]
        item: dict[str, object] = {"embedding_dim": dimension, "n_folds": len(rows)}
        for metric in METRICS:
            values = np.array([float(row[metric]) for row in rows])
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1))
        summary.append(item)

    with (OUT / "summary_by_dimension.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "experiment": "decontaminated_groupcv_embedding_dim_ablation_20260804",
                "protocol": "3 repeats x StratifiedGroupKFold(5), grouped by exact genus_sequence+mask hash",
                "dimensions": EMBED_DIMS,
                "controlled_variables": "masked mean pooling, MLP hidden=256, BN, ReLU, Dropout=0.3, 50 epochs, AdamW, weighted cross entropy",
                "n_total_folds_per_dimension": 15,
            },
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"SAVED {OUT}", flush=True)


if __name__ == "__main__":
    main()
