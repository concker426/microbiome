#!/usr/bin/env python3
"""
Super-blind evaluation of trained QA model.

- Loads procyon_qa_7b
- Aligns each QA entry to its sample's genus sequence via sample_id
- Subsamples N entries (default 500) for tractable runtime
- Reports diagnostic accuracy on entries where a label can be parsed,
  plus raw qualitative samples
"""
import os, json, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["TORCH_FLASH_ATTN_ENABLED"] = "0"

import fix_flash_attn  # noqa: F401
import accelerate.utils.imports as _acc_imports
_acc_imports.is_deepspeed_available = lambda: False
import accelerate.utils.other as _acc_other
_acc_other.is_deepspeed_available = lambda: False

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from mgm_encoder import MGMEncoder
from run_microbiome_qa_7b import (
    ProjectionLayer, MultimodalQAModel, extract_label, MAX_SEQ_LEN,
    VOCAB_SIZE, EMBED_DIM, MGM_LAYERS, MGM_HEADS, MGM_FFN_DIM,
    MGM_DROPOUT, MODEL_PATH, MAX_LENGTH,
)

BASE = "/hd/liujx/microbiome_llm_project"
QA_TEST_DATA = os.path.join(BASE, "data/agp_ftp_processed_qa/test_qa.jsonl")
QA_TEST_SEQS = os.path.join(BASE, "data/agp_ftp_processed_qa/test_genus_sequences.npy")
QA_TEST_MASKS = os.path.join(BASE, "data/agp_ftp_processed_qa/test_genus_masks.npy")
SAMPLE_ORDER_SRC = os.path.join(BASE, "data/agp_ftp_processed/test_set.jsonl")

TRAINED_DIR = os.environ.get(
    "QA_EVAL_TRAINED_DIR",
    os.path.join(BASE, "saved_models/procyon_qa_7b"),
)
EVAL_DIR = os.environ.get(
    "QA_EVAL_OUTPUT_DIR",
    os.path.join(BASE, "eval_results_procyon_qa_7b"),
)
os.makedirs(EVAL_DIR, exist_ok=True)

N_EVAL = 500
SEED = 42


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def generate_response(model, tokenizer, prompt_text, genus_ids, genus_mask,
                      device, max_new_tokens=128):
    micro_embed = model.encoder(genus_ids, genus_mask)
    micro_embed = micro_embed.to(model.projection.proj.weight.dtype)
    micro_token = model.projection(micro_embed).unsqueeze(1)

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False, add_generation_prompt=True,
    )
    prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=MAX_LENGTH).to(device)

    text_embeds = model.llm.base_model.model.model.embed_tokens(
        prompt_inputs["input_ids"]
    )
    micro_token = micro_token.to(text_embeds.dtype)
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
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


def main():
    print("=" * 60)
    print("  QA Super-blind Evaluation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data + build sample_id → seq_idx map
    print("\n[1/3] Loading data...")
    qa_entries = load_jsonl(QA_TEST_DATA)
    seqs = np.load(QA_TEST_SEQS).astype(np.int64)
    masks = np.load(QA_TEST_MASKS)
    sample_order = load_jsonl(SAMPLE_ORDER_SRC)
    sid_to_idx = {item["sample_id"]: i for i, item in enumerate(sample_order)}
    print(f"  {len(qa_entries)} QA entries / {len(sample_order)} unique samples")
    print(f"  seqs: {seqs.shape}")

    # Subsample
    rng = random.Random(SEED)
    if N_EVAL < len(qa_entries):
        eval_indices = rng.sample(range(len(qa_entries)), N_EVAL)
    else:
        eval_indices = list(range(len(qa_entries)))
    print(f"  Evaluating {len(eval_indices)} entries (seed={SEED})")

    # Load model
    print("\n[2/3] Loading trained QA model...")
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

    model = MultimodalQAModel(lora, encoder, projection)
    model.eval()
    print(f"  Model loaded from {TRAINED_DIR}")

    # Evaluate
    print("\n[3/3] Generating responses...")
    predictions = []
    for n, idx in enumerate(eval_indices):
        item = qa_entries[idx]
        sid = item["sample_id"]
        if sid not in sid_to_idx:
            continue
        seq_idx = sid_to_idx[sid]
        seq = seqs[seq_idx][:MAX_SEQ_LEN]
        msk = masks[seq_idx][:MAX_SEQ_LEN]

        genus_ids = torch.from_numpy(np.asarray(seq).astype(np.int64)).long().unsqueeze(0).to(device)
        genus_mask = torch.from_numpy(np.asarray(msk)).bool().unsqueeze(0).to(device)

        question = item["messages"][0]["content"]
        true_answer = item["messages"][1]["content"]
        gold_label = item["label"]

        try:
            generated = generate_response(
                model, tokenizer, question, genus_ids, genus_mask,
                device, max_new_tokens=128,
            )
        except Exception as e:
            generated = f"<error: {e}>"

        pred_label = extract_label(generated)

        predictions.append({
            "qa_idx": idx,
            "sample_id": sid,
            "task_type": item["task_type"],
            "label": gold_label,
            "label_detail": item.get("label_detail", "None"),
            "question": question,
            "true_answer": true_answer[:300],
            "generated": generated[:300],
            "predicted_label": pred_label,
        })

        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(eval_indices)}", flush=True)

    # Report
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    print("\n" + "=" * 60)
    print("  QA SUPER-BLIND RESULTS")
    print("=" * 60)
    print(f"\n  Total responses: {len(predictions)}")

    diag_preds = [p for p in predictions if p["predicted_label"] is not None]
    print(f"  Parseable diagnoses: {len(diag_preds)} ({100*len(diag_preds)/max(1,len(predictions)):.1f}%)")

    acc = macro_f1 = None
    if diag_preds:
        true = [p["label"] for p in diag_preds]
        pred = [p["predicted_label"] for p in diag_preds]
        labels = ["Healthy", "Disease"]
        acc = accuracy_score(true, pred)
        macro_f1 = f1_score(true, pred, labels=labels, average="macro", zero_division=0)
        print(f"\n  Diagnosis Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Diagnosis Macro F1:  {macro_f1:.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(true, pred, labels=labels, zero_division=0))

    # Save
    with open(os.path.join(EVAL_DIR, "qa_super_blind_predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    with open(os.path.join(EVAL_DIR, "qa_super_blind_results.json"), "w") as f:
        json.dump({
            "model": "ProCyon_QA_7B",
            "test_set": "qa_test_subsample",
            "n_total_entries": len(qa_entries),
            "n_evaluated": len(predictions),
            "n_parseable_diagnoses": len(diag_preds),
            "accuracy": float(acc) if acc is not None else None,
            "macro_f1": float(macro_f1) if macro_f1 is not None else None,
        }, f, indent=2)

    print(f"\n✅ QA evaluation complete. Results: {EVAL_DIR}/")


if __name__ == "__main__":
    main()
