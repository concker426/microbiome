#!/usr/bin/env python3
"""
Unified fine-tuning launcher — train all 5 variants with a given encoder.

Usage:
  # With old encoder (AGP-FTP 10k):
  python3 run_finetune_all_variants.py --encoder mgm_encoder_pretrained

  # With new encoder (Qiita 50k):
  python3 run_finetune_all_variants.py --encoder mgm_encoder_qiita_50k

  # Single variant only:
  python3 run_finetune_all_variants.py --encoder mgm_encoder_qiita_50k --only nl

  # Dry run:
  python3 run_finetune_all_variants.py --dry-run

Output per variant: saved_models/procyon_{variant}_{encoder_tag}/
Eval output:        eval_results_procyon_{variant}_{encoder_tag}/
Comparison table:   eval_results_{encoder_tag}_comparison.json
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

BASE = "/hd/liujx/microbiome_llm_project"
DATA_BASE = os.path.join(BASE, "data")

VARIANTS = {
    "nl": {
        "script": "run_microbiome_nl_7b.py",
        "data": {
            "TRAIN_DATA": "data/agp_ftp_processed_nl/train_nl.jsonl",
            "TEST_DATA":  "data/agp_ftp_processed_nl/test_nl.jsonl",
            "TRAIN_SEQUENCES": "data/agp_ftp_processed/train_genus_sequences.npy",
            "TRAIN_MASKS": "data/agp_ftp_processed/train_genus_masks.npy",
            "TEST_SEQUENCES": "data/agp_ftp_processed/test_genus_sequences.npy",
            "TEST_MASKS": "data/agp_ftp_processed/test_genus_masks.npy",
        },
    },
    "nl_aug": {
        "script": "run_microbiome_nl_7b.py",
        "data": {
            "TRAIN_DATA": "data/agp_ftp_processed_nl_aug/train_nl.jsonl",
            "TEST_DATA":  "data/agp_ftp_processed_nl_aug/test_nl.jsonl",
            "TRAIN_SEQUENCES": "data/agp_ftp_processed/train_genus_sequences.npy",
            "TRAIN_MASKS": "data/agp_ftp_processed/train_genus_masks.npy",
            "TEST_SEQUENCES": "data/agp_ftp_processed/test_genus_sequences.npy",
            "TEST_MASKS": "data/agp_ftp_processed/test_genus_masks.npy",
        },
    },
    "qa": {
        "script": "run_microbiome_qa_7b.py",
        "data": {
            "TRAIN_DATA": "data/agp_ftp_processed_qa/train_qa.jsonl",
            "TEST_DATA":  "data/agp_ftp_processed_qa/test_qa.jsonl",
            "TRAIN_SEQUENCES": "data/agp_ftp_processed_qa/train_genus_sequences.npy",
            "TRAIN_MASKS": "data/agp_ftp_processed_qa/train_genus_masks.npy",
            "TEST_SEQUENCES": "data/agp_ftp_processed_qa/test_genus_sequences.npy",
            "TEST_MASKS": "data/agp_ftp_processed_qa/test_genus_masks.npy",
        },
    },
    "subtype": {
        "script": "run_microbiome_subtype_7b.py",
        "data": {},  # subtype defines its own paths internally
    },
    "attribution": {
        "script": "run_microbiome_attribution_7b.py",
        "data": {},  # attribution pulls from run_microbiome_nl_7b
    },
}


def get_encoder_path(encoder_name):
    return os.path.join(BASE, "saved_models", encoder_name, "mgm_encoder.pt")


def run_variant(name, cfg, encoder_path, encoder_tag, gpu_id, dry_run=False):
    """Patch the variant script's config and launch training."""
    script = cfg["script"]
    script_path = os.path.join(BASE, script)
    if not os.path.exists(script_path):
        print(f"  SKIP {name}: script {script} not found")
        return None

    output_dir = os.path.join(BASE, f"saved_models/procyon_{name}_{encoder_tag}")
    eval_dir = os.path.join(BASE, f"eval_results_procyon_{name}_{encoder_tag}")

    # Build a launcher that patches module-level globals before importing
    launcher = f"""
import os, sys
os.chdir("{BASE}")
sys.path.insert(0, "{BASE}")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["TORCH_FLASH_ATTN_ENABLED"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"

import fix_flash_attn
import accelerate.utils.imports as _acc_imports
_acc_imports.is_deepspeed_available = lambda: False
import accelerate.utils.other as _acc_other
_acc_other.is_deepspeed_available = lambda: False

import run_microbiome_nl_7b as nl
"""

    for key, val in cfg["data"].items():
        launcher += f'nl.{key} = os.path.join("{BASE}", "{val}")\n'

    launcher += f"""
nl.OUTPUT_DIR = "{output_dir}"
nl.EVAL_DIR = "{eval_dir}"
nl.PRETRAINED_ENCODER = "{encoder_path}"
nl.EPOCHS = 3
nl.LR = 1e-4
os.makedirs(nl.OUTPUT_DIR, exist_ok=True)
os.makedirs(nl.EVAL_DIR, exist_ok=True)

print("="*60)
print("  Fine-tuning: {name}  |  Encoder: {encoder_tag}  |  GPU: {gpu_id}")
print("  Output: {output_dir}")
print("="*60)
nl.main()
"""

    if dry_run:
        print(f"  [{name}] Would run on GPU {gpu_id} → {output_dir}")
        return {"variant": name, "output": output_dir, "eval": eval_dir}

    tmp = f"/tmp/finetune_{name}_{encoder_tag}.py"
    with open(tmp, "w") as f:
        f.write(launcher)

    print(f"  [{name}] Launching on GPU {gpu_id} → {output_dir}")
    proc = subprocess.Popen(
        ["python3", "-u", tmp],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return {"variant": name, "output": output_dir, "eval": eval_dir, "pid": proc.pid}


def main():
    ap = argparse.ArgumentParser(description="Unified fine-tuning for all 5 variants")
    ap.add_argument("--encoder", required=True,
                    help="Encoder name (subdir of saved_models/, e.g. mgm_encoder_qiita_50k)")
    ap.add_argument("--only", default="", help="Run only this variant (nl/nl_aug/qa/subtype/attribution)")
    ap.add_argument("--gpu", default="0,1,2", help="Comma-separated GPU IDs to use")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    encoder_path = get_encoder_path(args.encoder)
    if not os.path.exists(encoder_path) and not args.dry_run:
        print(f"ERROR: Encoder not found at {encoder_path}")
        return 1

    print(f"Encoder: {encoder_path}")
    print(f"Encoder tag: {args.encoder}")

    gpus = args.gpu.split(",")
    to_run = [args.only] if args.only else list(VARIANTS.keys())

    results = {}
    for i, name in enumerate(to_run):
        if name not in VARIANTS:
            print(f"  SKIP: unknown variant '{name}'")
            continue
        gpu_id = gpus[i % len(gpus)]
        cfg = VARIANTS[name]
        result = run_variant(name, cfg, encoder_path, args.encoder, gpu_id, args.dry_run)
        results[name] = result

    if args.dry_run:
        print(f"\nWould run {len(results)} variants on GPUs {args.gpu}")
        return 0

    print(f"\n{'='*60}")
    print(f"  Launched {len(results)} variants")
    print(f"{'='*60}")
    for name, r in results.items():
        if r:
            print(f"  {name:<15} → {r['output']}")

    meta_path = os.path.join(BASE, f"eval_results_{args.encoder}_launcher.json")
    with open(meta_path, "w") as f:
        json.dump({k: v for k, v in results.items() if v}, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")


if __name__ == "__main__":
    main()
