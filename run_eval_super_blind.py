#!/usr/bin/env python3
"""
Evaluate trained model on super-blind held-out test set.

Usage: python3 run_eval_super_blind.py
"""
import os
import json
import re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from mgm_encoder import MGMEncoder
from run_microbiome_nl_7b import (
    ProjectionLayer, MultimodalNLModel, extract_label, MAX_SEQ_LEN,
    VOCAB_SIZE, EMBED_DIM, MGM_LAYERS, MGM_HEADS, MGM_FFN_DIM,
    MGM_DROPOUT, MODEL_PATH, MAX_LENGTH,
)

BASE = "/hd/liujx/microbiome_llm_project"
NEW_TEST_DIR = os.path.join(BASE, "data/agp_ftp_processed_nl_new_test")
NEW_TEST_DATA = os.path.join(NEW_TEST_DIR, "new_test_nl.jsonl")
NEW_TEST_SEQUENCES = os.path.join(NEW_TEST_DIR, "new_test_genus_sequences.npy")
NEW_TEST_MASKS = os.path.join(NEW_TEST_DIR, "new_test_genus_masks.npy")

# Which trained model to evaluate
TRAINED_DIR = "/hd/liujx/microbiome_llm_project/saved_models/procyon_nl_7b"
EVAL_DIR = "/hd/liujx/microbiome_llm_project/eval_results_procyon_nl_7b_super_blind"
os.makedirs(EVAL_DIR, exist_ok=True)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def evaluate(model, tokenizer, test_data, test_sequences, test_masks, device,
             name="super_blind", max_new_tokens=128):
    model.eval()
    print(f"\n  Evaluating {len(test_data)} NL entries...")

    all_predictions = []

    for idx, item in enumerate(test_data):
        true_label = item["label"]
        messages = item["messages"]
        seq = test_sequences[idx // 3] if idx // 3 < len(test_sequences) else test_sequences[0]
        if isinstance(seq, np.ndarray) and seq.ndim == 2:
            seq = seq[0]
        msk = test_masks[idx // 3] if idx // 3 < len(test_masks) else test_masks[0]
        if isinstance(msk, np.ndarray) and msk.ndim == 2:
            msk = msk[0]

        seq = seq[:MAX_SEQ_LEN]
        msk = msk[:MAX_SEQ_LEN]

        genus_ids = torch.from_numpy(np.asarray(seq).astype(np.int64)).long().unsqueeze(0).to(device)
        genus_mask = torch.from_numpy(np.asarray(msk)).bool().unsqueeze(0).to(device)

        micro_embed = model.encoder(genus_ids, genus_mask)
        micro_embed = micro_embed.to(model.projection.proj.weight.dtype)
        micro_token = model.projection(micro_embed).unsqueeze(1)

        prompt = tokenizer.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True,
        )
        prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                   max_length=MAX_LENGTH).to(device)

        text_embeds = model.llm.base_model.model.model.embed_tokens(
            prompt_inputs["input_ids"]
        )
        combined = torch.cat([micro_token, text_embeds], dim=1)

        seq_len = combined.shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        outputs = model.llm(inputs_embeds=combined, position_ids=position_ids, use_cache=True)

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = [next_token]
        current_len = seq_len

        for _ in range(max_new_tokens):
            pos_id = torch.full((1, 1), current_len, dtype=torch.long, device=device)
            out = model.llm(
                input_ids=next_token, position_ids=pos_id,
                past_key_values=outputs.past_key_values, use_cache=True,
            )
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            if next_token.item() == tokenizer.eos_token_id:
                break
            generated_ids.append(next_token)
            current_len += 1
            outputs.past_key_values = out.past_key_values

        generated_ids = torch.cat(generated_ids, dim=1)
        generated = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        predicted_label = extract_label(generated) if item["task_type"] == "diagnosis" else None

        all_predictions.append({
            "sample_idx": idx,
            "task_type": item["task_type"],
            "true_label": true_label,
            "predicted_label": predicted_label,
            "generated": generated.strip()[:500],
        })

        if (idx + 1) % 200 == 0:
            print(f"    {idx+1}/{len(test_data)}", flush=True)

    return all_predictions


def main():
    from sklearn.metrics import classification_report, accuracy_score, f1_score

    print("=" * 60)
    print("  Super-blind Test Set Evaluation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load data
    print("\n[1/3] Loading super-blind test data...")
    test_data = load_jsonl(NEW_TEST_DATA)
    test_sequences = np.load(NEW_TEST_SEQUENCES).astype(np.int64)
    test_masks = np.load(NEW_TEST_MASKS)
    print(f"  {len(test_data)} NL entries, sequences: {test_sequences.shape}")

    # Load model
    print("\n[2/3] Loading trained model...")
    tokenizer = AutoTokenizer.from_pretrained(TRAINED_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    lora = PeftModel.from_pretrained(base, TRAINED_DIR)
    lora.config.use_cache = True

    encoder = MGMEncoder(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
        n_layers=MGM_LAYERS, n_heads=MGM_HEADS, ffn_dim=MGM_FFN_DIM,
        max_seq_len=MAX_SEQ_LEN, dropout=MGM_DROPOUT,
    )
    projection = ProjectionLayer()

    ckpt = torch.load(os.path.join(TRAINED_DIR, "multimodal_components.pt"),
                       map_location=device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    projection.load_state_dict(ckpt["projection_state_dict"])
    encoder.to(lora.device, dtype=torch.bfloat16)
    projection.to(lora.device, dtype=torch.bfloat16)

    model = MultimodalNLModel(lora, encoder, projection)
    print(f"  Model loaded from {TRAINED_DIR}")

    # Evaluate
    print("\n[3/3] Running evaluation...")
    predictions = evaluate(
        model, tokenizer, test_data,
        test_sequences, test_masks, device,
        name="super_blind", max_new_tokens=128,
    )

    # Report
    print(f"\n{'='*60}")
    print(f"  SUPER-BLIND TEST SET RESULTS")
    print(f"{'='*60}")

    diag_preds = [
        p for p in predictions
        if p["task_type"] == "diagnosis" and p["predicted_label"] is not None
    ]
    if diag_preds:
        true = [p["true_label"] for p in diag_preds]
        pred = [p["predicted_label"] for p in diag_preds]
        labels = ["Healthy", "Disease"]
        acc = accuracy_score(true, pred)
        macro_f1 = f1_score(true, pred, labels=labels, average="macro", zero_division=0)
        print(f"\n  Diagnosis Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Diagnosis Macro F1:  {macro_f1:.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(true, pred, labels=labels, zero_division=0))

    # Save
    with open(os.path.join(EVAL_DIR, "super_blind_predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    results = {
        "model": "ProCyon_NL_7B",
        "test_set": "super_blind_holdout",
        "n_samples": len(predictions),
        "accuracy": float(acc) if diag_preds else None,
        "macro_f1": float(macro_f1) if diag_preds else None,
    }
    with open(os.path.join(EVAL_DIR, "super_blind_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Super-blind evaluation complete!")
    print(f"   Results: {EVAL_DIR}/")


if __name__ == "__main__":
    main()
