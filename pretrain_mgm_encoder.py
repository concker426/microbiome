#!/usr/bin/env python3
"""
MGM-style next-genus prediction pre-training.

Usage:
  python3 pretrain_mgm_encoder.py                                    # default AGP-FTP
  python3 pretrain_mgm_encoder.py --data-dir /path/to/qiita_pretrain \
      --output-dir saved_models/mgm_encoder_qiita_50k --epochs 50
"""
import argparse, json, os, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from mgm_encoder import create_model

# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = "/hd/liujx/microbiome_llm_project/data/agp_ftp_processed"
DEFAULT_OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/saved_models/mgm_encoder_pretrained"

BATCH_SIZE = 64
LR = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
GRAD_CLIP = 1.0
DROPOUT = 0.2
N_LAYERS = 6
N_HEADS = 8
EMBED_DIM = 768
FFN_DIM = 2048
VAL_SPLIT = 0.1


class GenusSequenceDataset(Dataset):
    def __init__(self, sequences_path, masks_path):
        self.sequences = np.load(sequences_path).astype(np.int64)
        self.masks = np.load(masks_path).astype(bool)
        print(f"  Loaded: {len(self.sequences)} sequences, "
              f"max_len={self.sequences.shape[1]}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "token_ids": torch.from_numpy(self.sequences[idx]),
            "mask": torch.from_numpy(self.masks[idx]),
        }


def split_train_val(seqs, masks, val_frac=0.1, seed=42):
    n = len(seqs)
    idx = np.random.RandomState(seed).permutation(n)
    n_val = int(n * val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return seqs[train_idx], masks[train_idx], seqs[val_idx], masks[val_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--early-stop", type=int, default=10)
    args = ap.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  MGM Next-Genus Pre-training")
    print(f"  Data: {data_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load data ─────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    vocab_path = os.path.join(data_dir, "genus_vocab.json")

    # Try train/val split files, else create split from single file
    train_seq_path = os.path.join(data_dir, "pretrain_train_sequences.npy")
    train_mask_path = os.path.join(data_dir, "pretrain_train_masks.npy")
    val_seq_path = os.path.join(data_dir, "pretrain_val_sequences.npy")
    val_mask_path = os.path.join(data_dir, "pretrain_val_masks.npy")

    if os.path.exists(train_seq_path) and os.path.exists(val_seq_path):
        train_dataset = GenusSequenceDataset(train_seq_path, train_mask_path)
        val_dataset = GenusSequenceDataset(val_seq_path, val_mask_path)
    else:
        # Split from single file
        seq_path = os.path.join(data_dir, "pretrain_sequences.npy")
        mask_path = os.path.join(data_dir, "pretrain_masks.npy")
        all_seqs = np.load(seq_path).astype(np.int64)
        all_masks = np.load(mask_path).astype(bool)
        print(f"  Loaded: {len(all_seqs)} sequences, max_len={all_seqs.shape[1]}")
        train_seqs, train_masks, val_seqs, val_masks = split_train_val(
            all_seqs, all_masks, VAL_SPLIT)
        # Save splits for reuse
        np.save(os.path.join(data_dir, "pretrain_train_sequences.npy"), train_seqs)
        np.save(os.path.join(data_dir, "pretrain_train_masks.npy"), train_masks)
        np.save(os.path.join(data_dir, "pretrain_val_sequences.npy"), val_seqs)
        np.save(os.path.join(data_dir, "pretrain_val_masks.npy"), val_masks)
        train_dataset = GenusSequenceDataset(
            os.path.join(data_dir, "pretrain_train_sequences.npy"),
            os.path.join(data_dir, "pretrain_train_masks.npy"))
        val_dataset = GenusSequenceDataset(
            os.path.join(data_dir, "pretrain_val_sequences.npy"),
            os.path.join(data_dir, "pretrain_val_masks.npy"))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────
    print("\n[2/4] Building model...")
    with open(vocab_path) as f:
        vocab = json.load(f)

    # Handle both vocab formats
    if isinstance(vocab, dict) and "vocab_size" in vocab:
        vocab_size = vocab["vocab_size"]
        max_seq_len = vocab.get("max_seq_len", 128)
        n_genera = vocab.get("n_genera", vocab_size - 4)
    else:
        # Flat {genus: id} dict
        vocab_size = len(vocab)
        max_seq_len = train_dataset.sequences.shape[1]
        n_genera = vocab_size - 4

    print(f"  Vocab size: {vocab_size}")
    print(f"  Max seq len: {max_seq_len}")
    print(f"  Genera: {n_genera}")

    model = create_model(
        vocab_path=vocab_path,
        n_layers=N_LAYERS, n_heads=N_HEADS,
        embed_dim=EMBED_DIM, ffn_dim=FFN_DIM,
        max_seq_len=max_seq_len, dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ── Optimizer ─────────────────────────────────────────────────
    print("\n[3/4] Setting up optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * args.epochs

    def warmup_cosine_schedule(step):
        if step < WARMUP_STEPS:
            return float(step) / float(max(1, WARMUP_STEPS))
        progress = float(step - WARMUP_STEPS) / float(max(1, total_steps - WARMUP_STEPS))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_schedule)

    # ── Train ─────────────────────────────────────────────────────
    print("\n[4/4] Starting pre-training...")
    print(f"  Batches/epoch: {len(train_loader)} ({len(train_dataset)} samples)")
    print(f"  Total steps: {total_steps}")
    print(f"  Early stop patience: {args.early_stop}")

    start_time = time.time()
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_steps = 0
        for batch in train_loader:
            token_ids = batch["token_ids"].to(device)
            mask = batch["mask"].to(device)
            logits = model(token_ids, mask)
            loss = model.compute_loss(logits, token_ids, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            train_steps += 1

        avg_train_loss = train_loss / max(train_steps, 1)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                token_ids = batch["token_ids"].to(device)
                mask = batch["mask"].to(device)
                logits = model(token_ids, mask)
                metrics = model.compute_metrics(logits, token_ids, mask)
                val_loss += metrics["loss"]
                val_acc += metrics["accuracy"]
                val_steps += 1

        avg_val_loss = val_loss / max(val_steps, 1)
        avg_val_acc = val_acc / max(val_steps, 1)
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.encoder.state_dict(),
                       os.path.join(output_dir, "mgm_encoder.pt"))
            torch.save(model.state_dict(),
                       os.path.join(output_dir, "mgm_pretrain_full.pt"))
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs or patience_counter == 0:
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"train_loss={avg_train_loss:.4f} | "
                  f"val_loss={avg_val_loss:.4f} (best={best_val_loss:.4f}) | "
                  f"val_acc={avg_val_acc:.4f} | lr={lr_now:.2e} | {elapsed:.0f}s")

        if patience_counter >= args.early_stop:
            print(f"  Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Pre-training complete! ({elapsed:.0f}s, {epoch} epochs)")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best val perplexity: {np.exp(best_val_loss):.2f}")
    print(f"  Encoder: {os.path.join(output_dir, 'mgm_encoder.pt')}")

    meta = {
        "method": "mgm_next_genus_prediction",
        "data_dir": data_dir,
        "architecture": {"type": "Transformer", "n_layers": N_LAYERS,
                         "n_heads": N_HEADS, "embed_dim": EMBED_DIM,
                         "ffn_dim": FFN_DIM, "vocab_size": vocab_size,
                         "max_seq_len": max_seq_len, "dropout": DROPOUT},
        "training": {"n_train": len(train_dataset), "n_val": len(val_dataset),
                     "epochs": epoch, "batch_size": args.batch_size,
                     "lr": args.lr, "weight_decay": WEIGHT_DECAY,
                     "warmup_steps": WARMUP_STEPS, "grad_clip": GRAD_CLIP},
        "results": {"best_val_loss": best_val_loss,
                    "best_val_perplexity": float(np.exp(best_val_loss))},
    }
    with open(os.path.join(output_dir, "pretrain_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Metadata: {os.path.join(output_dir, 'pretrain_metadata.json')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
