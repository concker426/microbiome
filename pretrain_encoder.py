#!/usr/bin/env python3
"""
Self-supervised pre-training for MicrobiomeEncoder via Denoising Autoencoder.

Task: mask random genera → encode → decode → reconstruct masked positions
Goal: learn microbial community structure from abundance patterns

Method:
  - Corruption: randomly mask 20% of genera (set to 0) + add small Gaussian noise
  - Encoder: same architecture as MicrobiomeEncoder (1222→512→768)
  - Decoder: symmetric (768→512→1222)
  - Loss: MSE only on masked positions (force meaningful reconstruction)

Output: pretrained encoder weights → loaded into ProCyon pipeline
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Config ──────────────────────────────────────────────────────────
DATA_DIR = "/hd/liujx/microbiome_llm_project/data/agp_ftp_processed"
TRAIN_VECTORS = os.path.join(DATA_DIR, "train_set_vectors.npy")
OUTPUT_PATH = "/hd/liujx/microbiome_llm_project/saved_models/procyon_microbiome_7b/pretrained_encoder.pt"

INPUT_DIM = 1222
EMBED_DIM = 768
MASK_RATIO = 0.20        # fraction of genera to mask
NOISE_STD = 0.01         # small Gaussian noise on non-masked
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-5


class MicrobiomeEncoder(nn.Module):
    """Same architecture as in run_procyon_microbiome_7b.py"""
    def __init__(self, input_dim=INPUT_DIM, embed_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class Autoencoder(nn.Module):
    """Encoder + Decoder for denoising pre-training"""
    def __init__(self, input_dim=INPUT_DIM, embed_dim=EMBED_DIM):
        super().__init__()
        self.encoder = MicrobiomeEncoder(input_dim, embed_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class MaskedAbundanceDataset(Dataset):
    """Apply random masking + noise to abundance vectors"""
    def __init__(self, vectors, mask_ratio=MASK_RATIO, noise_std=NOISE_STD):
        self.vectors = torch.from_numpy(vectors).float()
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        x = self.vectors[idx].clone()
        # Create mask: 1 = masked position
        mask = torch.zeros(INPUT_DIM, dtype=torch.bool)
        n_mask = max(1, int(INPUT_DIM * self.mask_ratio))
        mask_idx = torch.randperm(INPUT_DIM)[:n_mask]
        mask[mask_idx] = True

        # Corrupt: zero out masked + Gaussian noise on rest
        x_corrupt = x.clone()
        x_corrupt[mask] = 0.0
        if self.noise_std > 0:
            noise = torch.randn_like(x_corrupt) * self.noise_std
            x_corrupt = x_corrupt + noise
            x_corrupt = x_corrupt.clamp(min=0.0)  # abundance ≥ 0

        return {"corrupted": x_corrupt, "original": x, "mask": mask}


def main():
    print("=" * 60)
    print("  Self-Supervised Encoder Pre-training")
    print("  Denoising Autoencoder on Abundance Vectors")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print(f"\nLoading training vectors...")
    vectors = np.load(TRAIN_VECTORS).astype(np.float32)
    print(f"  Shape: {vectors.shape}")
    print(f"  Non-zero: {np.count_nonzero(vectors) / vectors.size:.1%}")
    print(f"  Mean sum per sample: {vectors.sum(axis=1).mean():.1f}")

    dataset = MaskedAbundanceDataset(vectors)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Build model
    model = Autoencoder().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nAutoencoder params: {n_params:,}")
    print(f"  Encoder: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  Decoder: {sum(p.numel() for p in model.decoder.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Train
    print(f"\nStarting pre-training: {EPOCHS} epochs, batch={BATCH_SIZE}, mask_ratio={MASK_RATIO}")
    start_time = time.time()
    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_masked = 0

        for batch in dataloader:
            x_corrupt = batch["corrupted"].to(device)
            x_orig = batch["original"].to(device)
            mask = batch["mask"].to(device)  # (B, D)

            x_pred = model(x_corrupt)  # (B, D)

            # MSE only on masked positions
            loss = ((x_pred[mask] - x_orig[mask]) ** 2).sum()
            n_masked = mask.sum().item()
            loss = loss / max(n_masked, 1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * n_masked
            total_masked += n_masked

        scheduler.step()
        avg_loss = total_loss / max(total_masked, 1)
        lr_now = optimizer.param_groups[0]["lr"]

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.encoder.state_dict(), OUTPUT_PATH)

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:3d}/{EPOCHS} | loss={avg_loss:.6f} | best={best_loss:.6f} | lr={lr_now:.2e} | {elapsed:.0f}s")

    elapsed = time.time() - start_time
    print(f"\n✅ Pre-training complete! ({elapsed:.0f}s)")
    print(f"   Best loss: {best_loss:.6f}")
    print(f"   Encoder weights saved to: {OUTPUT_PATH}")

    # Quick verification: reconstruct a test sample
    print(f"\nVerification sample:")
    model.eval()
    with torch.no_grad():
        test_vec = vectors[0:1].copy()
        test_t = torch.from_numpy(test_vec).float().to(device)
        # Mask top genera
        mask_t = torch.zeros_like(test_t, dtype=torch.bool)
        _, top_idx = torch.topk(test_t[0], k=5)
        mask_t[0, top_idx] = True
        test_corrupt = test_t.clone()
        test_corrupt[mask_t] = 0.0
        recon = model(test_corrupt)
        mse = ((recon[mask_t] - test_t[mask_t]) ** 2).mean().item()
        print(f"   Top-5 genera masked reconstruction MSE: {mse:.6f}")
        print(f"   Original top-5 sum: {test_t[0, top_idx].sum().item():.1f}")
        print(f"   Reconstructed top-5 sum: {recon[0, top_idx].sum().item():.1f}")

    # Save training metadata
    meta = {
        "method": "denoising_autoencoder",
        "mask_ratio": MASK_RATIO,
        "noise_std": NOISE_STD,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "best_loss": best_loss,
        "training_samples": len(vectors),
        "encoder_architecture": "Linear(1222,512)->ReLU->Linear(512,768)",
    }
    meta_path = OUTPUT_PATH.replace(".pt", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"   Metadata: {meta_path}")

    print(f"\n{'='*60}")
    print(f"  Next: use pretrained weights in run_procyon_microbiome_7b.py")
    print(f"  Set PRETRAINED_ENCODER = '{OUTPUT_PATH}'")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
