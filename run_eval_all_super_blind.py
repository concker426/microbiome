#!/usr/bin/env python3
"""
Unified super-blind evaluation: run all 5 model variants on the super-blind
held-out test set and produce a single comparison table.

Variants:
  1. NL            (procyon_nl_7b)
  2. NL-aug        (procyon_nl_7b_aug)
  3. QA-balanced   (procyon_qa_balanced_7b)
  4. Subtype-v2    (procyon_subtype_v2_7b)
  5. Attribution   (procyon_attribution_7b)

Run: CUDA_VISIBLE_DEVICES=0 python3 run_eval_all_super_blind.py
"""
import os, json, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, classification_report

from mgm_encoder import MGMEncoder
from run_microbiome_nl_7b import (
    ProjectionLayer, MultimodalNLModel, MAX_SEQ_LEN,
    VOCAB_SIZE, EMBED_DIM, MGM_LAYERS, MGM_HEADS, MGM_FFN_DIM,
    MGM_DROPOUT, MODEL_PATH, MAX_LENGTH,
)

BASE = "/hd/liujx/microbiome_llm_project"
SUPER_BLIND_DIR = os.path.join(BASE, "data/agp_ftp_processed_nl_new_test")
SUPER_BLIND_DATA = os.path.join(SUPER_BLIND_DIR, "new_test_nl.jsonl")
SUPER_BLIND_SEQ = os.path.join(SUPER_BLIND_DIR, "new_test_genus_sequences.npy")
SUPER_BLIND_MSK = os.path.join(SUPER_BLIND_DIR, "new_test_genus_masks.npy")
EVAL_BASE = os.path.join(BASE, "eval_results_super_blind_unified")
os.makedirs(EVAL_BASE, exist_ok=True)

VARIANTS = {
    "NL":           "saved_models/procyon_nl_7b",
    "NL-aug":       "saved_models/procyon_nl_7b_aug",
    "QA-balanced":  "saved_models/procyon_qa_balanced_7b",
    "Subtype-v2":   "saved_models/procyon_subtype_v2_7b",
    "Attribution":  "saved_models/procyon_attribution_7b",
}

ALL_LABELS = ["Healthy", "Disease"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def extract_label(text):
    m = re.search(r'(?:诊断结果)[：:]\s*(Healthy|Disease|CD|UC)', text)
    if m:
        return m.group(1)
    # Fallback: first occurrence
    for lab in ["Healthy", "Disease", "CD", "UC"]:
        if lab in text:
            return lab
    return None


def build_safe_dirname(name):
    return name.lower().replace("-", "_").replace(" ", "_")


@torch.no_grad()
def evaluate_variant(model, tokenizer, test_data, test_seqs, test_msks,
                     device, name, max_new_tokens=128):
    model.eval()
    predictions = []
    for idx, item in enumerate(test_data):
        true_label = item.get("label", item.get("label_detail", "unknown"))
        # Map test_data index → sequence index: 3 entries per sample
        seq_idx = idx // 3
        seq = test_seqs[seq_idx]
        msk = test_msks[seq_idx]
        seq = seq[:MAX_SEQ_LEN]
        msk = msk[:MAX_SEQ_LEN]

        genus_ids = torch.from_numpy(np.asarray(seq).astype(np.int64)).long().unsqueeze(0).to(device)
        genus_mask = torch.from_numpy(np.asarray(msk)).bool().unsqueeze(0).to(device)

        micro_embed = model.encoder(genus_ids, genus_mask)
        micro_embed = micro_embed.to(model.projection.proj.weight.dtype)
        micro_token = model.projection(micro_embed).unsqueeze(1)

        messages = item["messages"]
        prompt = tokenizer.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True,
        )
        prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                   max_length=MAX_LENGTH).to(device)
        text_embeds = model.llm.base_model.model.model.embed_tokens(prompt_inputs["input_ids"])
        combined = torch.cat([micro_token, text_embeds], dim=1)

        seq_len = combined.shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        outputs = model.llm(inputs_embeds=combined, position_ids=position_ids, use_cache=True)

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = [next_token]
        current_len = seq_len
        for _ in range(max_new_tokens):
            pos_id = torch.full((1, 1), current_len, dtype=torch.long, device=device)
            out = model.llm(input_ids=next_token, position_ids=pos_id,
                            past_key_values=outputs.past_key_values, use_cache=True)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            if next_token.item() == tokenizer.eos_token_id:
                break
            generated_ids.append(next_token)
            current_len += 1
            outputs.past_key_values = out.past_key_values

        generated_ids = torch.cat(generated_ids, dim=1)
        generated = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        predicted_label = extract_label(generated)

        predictions.append({
            "sample_idx": idx,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "generated": generated.strip()[:500],
        })

        if (idx + 1) % 100 == 0:
            print(f"    [{name}] {idx+1}/{len(test_data)}", flush=True)

    return predictions


def load_model(checkpoint_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    lora = PeftModel.from_pretrained(base, checkpoint_dir)
    lora.config.use_cache = True

    encoder = MGMEncoder(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
        n_layers=MGM_LAYERS, n_heads=MGM_HEADS, ffn_dim=MGM_FFN_DIM,
        max_seq_len=MAX_SEQ_LEN, dropout=MGM_DROPOUT,
    )
    projection = ProjectionLayer()

    ckpt = torch.load(os.path.join(checkpoint_dir, "multimodal_components.pt"),
                       map_location=device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    projection.load_state_dict(ckpt["projection_state_dict"])
    encoder.to(lora.device, dtype=torch.bfloat16)
    projection.to(lora.device, dtype=torch.bfloat16)

    model = MultimodalNLModel(lora, encoder, projection)
    return model, tokenizer


def main():
    print("=" * 60)
    print("  Unified Super-Blind Evaluation (All 5 Variants)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load super-blind data
    print("\n[1] Loading super-blind test data...")
    test_data = load_jsonl(SUPER_BLIND_DATA)
    test_seqs = np.load(SUPER_BLIND_SEQ).astype(np.int64)
    test_msks = np.load(SUPER_BLIND_MSK)
    print(f"  {len(test_data)} samples, sequences: {test_seqs.shape}")

    # Each sample has 3 entries (diagnosis, marker_analysis, comparison).
    # Sequences array is (N, max_len) where N = len(test_data) // 3.
    # Map diagnosis entries (every 3rd entry starting at 0) to sequence index.
    diag_indices = [i for i, d in enumerate(test_data) if d.get("task_type") == "diagnosis"]
    print(f"  {len(diag_indices)} diagnosis samples for evaluation")

    all_results = {}

    for name, ckpt_rel in VARIANTS.items():
        ckpt_dir = os.path.join(BASE, ckpt_rel)
        if not os.path.exists(os.path.join(ckpt_dir, "multimodal_components.pt")):
            print(f"\n  SKIP {name}: no checkpoint at {ckpt_dir}")
            continue

        print(f"\n{'='*40}")
        print(f"  [{name}] Loading from {ckpt_dir}")
        print(f"{'='*40}")

        model, tokenizer = load_model(ckpt_dir, device)

        # Only evaluate diagnosis items
        diagnosis_items = [test_data[i] for i in diag_indices]
        # Each sample has 3 entries; map to sequence indices
        diagnosis_seqs = test_seqs[[i // 3 for i in diag_indices]]
        diagnosis_msks = test_msks[[i // 3 for i in diag_indices]]

        preds = evaluate_variant(
            model, tokenizer, diagnosis_items,
            diagnosis_seqs, diagnosis_msks, device, name,
        )

        # Clean up model to free GPU memory
        del model
        torch.cuda.empty_cache()

        valid = [p for p in preds if p["predicted_label"] is not None]
        if valid:
            true = [p["true_label"] for p in valid]
            pred = [p["predicted_label"] for p in valid]
            labels = sorted(set(true + pred))
            acc = accuracy_score(true, pred)
            macro_f1 = f1_score(true, pred, labels=labels, average="macro", zero_division=0)
        else:
            acc = 0.0
            macro_f1 = 0.0
            true, pred = [], []
            labels = ALL_LABELS

        print(f"\n  [{name}] Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"  [{name}] Macro F1: {macro_f1:.4f}")

        all_results[name] = {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "n_valid": len(valid),
            "n_total": len(preds),
        }

        safe_name = build_safe_dirname(name)
        out_dir = os.path.join(EVAL_BASE, safe_name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "predictions.json"), "w") as f:
            json.dump(preds, f, indent=2, ensure_ascii=False)
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(all_results[name], f, indent=2)

    # Final comparison table
    print(f"\n\n{'='*70}")
    print("  SUPER-BLIND COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Variant':<18} {'Accuracy':>10} {'Macro F1':>10} {'N valid':>8}")
    print("-" * 50)
    for name, r in all_results.items():
        print(f"{name:<18} {r['accuracy']:>9.4f} {r['macro_f1']:>9.4f} {r['n_valid']:>8}")

    with open(os.path.join(EVAL_BASE, "comparison_table.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone. Results in {EVAL_BASE}/")


if __name__ == "__main__":
    main()
