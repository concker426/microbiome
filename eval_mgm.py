#!/usr/bin/env python3
"""Evaluate the trained MGM ProCyon model."""
import os, json, re, sys
import numpy as np
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from mgm_encoder import MGMEncoder

DATA_DIR = "/hd/liujx/microbiome_llm_project/data/agp_ftp_processed"
OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/saved_models/procyon_mgm_7b"
EVAL_DIR = "/hd/liujx/microbiome_llm_project/eval_results_procyon_mgm_7b"
MODEL_PATH = "/hd/gcr/hf_models/Qwen2.5-7B-Instruct"
ALL_LABELS = ["Healthy", "Disease"]
MAX_LENGTH = 1024
MAX_SEQ_LEN = 86

class ProjectionLayer(torch.nn.Module):
    def __init__(self, embed_dim=768, llm_hidden=3584):
        super().__init__()
        self.proj = torch.nn.Linear(embed_dim, llm_hidden)
    def forward(self, x):
        return self.proj(x)

def extract_label(text):
    m = re.search(r'诊断结果[：:]\s*(\S+)', text)
    if m:
        label = m.group(1).strip('。，, \n')
        if label in ALL_LABELS: return label
    for kw in ALL_LABELS:
        if kw in text: return kw
    cn_map = {'健康': 'Healthy', '疾病': 'Disease'}
    for cn, en in cn_map.items():
        if cn in text: return en
    return None

device = torch.device("cuda")

print("Loading data...")
test_data = [json.loads(l) for l in open(f"{DATA_DIR}/test_set.jsonl")]
test_sequences = np.load(f"{DATA_DIR}/test_genus_sequences.npy").astype(np.int64)
test_masks = np.load(f"{DATA_DIR}/test_genus_masks.npy")
print(f"  Test: {len(test_data)} samples, seq shape: {test_sequences.shape}")

print("Loading tokenizer + LLM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

base = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="cuda:0", trust_remote_code=True, torch_dtype=torch.bfloat16)
lora = PeftModel.from_pretrained(base, OUTPUT_DIR)
lora.config.use_cache = True

print("Loading encoder + projection...")
encoder = MGMEncoder(vocab_size=1226, embed_dim=768,
    n_layers=6, n_heads=8, ffn_dim=2048, max_seq_len=MAX_SEQ_LEN, dropout=0.1)
encoder.to(device, dtype=torch.bfloat16)
proj = ProjectionLayer().to(device, dtype=torch.bfloat16)

ckpt = torch.load(f"{OUTPUT_DIR}/multimodal_components.pt", map_location=device)
encoder.load_state_dict(ckpt["encoder_state_dict"])
proj.load_state_dict(ckpt["projection_state_dict"])
print("  Loaded successfully!")

encoder.eval()
proj.eval()
lora.eval()

true_labels, pred_labels = [], []

print("\nEvaluating...")
for idx, item in enumerate(test_data):
    messages = item["messages"]
    true_label = item["label"]
    
    seq = test_sequences[idx][:MAX_SEQ_LEN]
    msk = test_masks[idx][:MAX_SEQ_LEN]
    genus_ids = torch.from_numpy(seq).long().unsqueeze(0).to(device)
    genus_mask = torch.from_numpy(msk).bool().unsqueeze(0).to(device)
    
    with torch.no_grad():
        micro_embed = encoder(genus_ids, genus_mask)
        micro_embed = micro_embed.to(proj.proj.weight.dtype)
        micro_token = proj(micro_embed).unsqueeze(1)
    
    prompt = tokenizer.apply_chat_template(
        [messages[0]], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    
    with torch.no_grad():
        text_embeds = lora.base_model.model.model.embed_tokens(inputs["input_ids"])
        combined = torch.cat([micro_token, text_embeds], dim=1)
        seq_len = combined.shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        outputs = lora(inputs_embeds=combined, position_ids=position_ids, use_cache=True)
        
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = [next_token]
        cur_len = seq_len
        
        for _ in range(64):
            pos_id = torch.full((1, 1), cur_len, dtype=torch.long, device=device)
            out = lora(input_ids=next_token, position_ids=pos_id,
                       past_key_values=outputs.past_key_values, use_cache=True)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            if next_token.item() == tokenizer.eos_token_id:
                break
            generated_ids.append(next_token)
            cur_len += 1
            outputs.past_key_values = out.past_key_values
    
    gen_ids = torch.cat(generated_ids, dim=1)
    generated = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    pred = extract_label(generated) or "UNKNOWN"
    
    true_labels.append(true_label)
    pred_labels.append(pred)
    
    if (idx + 1) % 100 == 0:
        print(f"  {idx+1}/{len(test_data)}", flush=True)

# Results
acc = accuracy_score(true_labels, pred_labels)
report = classification_report(true_labels, pred_labels, labels=ALL_LABELS, zero_division=0)
cm = confusion_matrix(true_labels, pred_labels, labels=ALL_LABELS)
macro_f1 = f1_score(true_labels, pred_labels, labels=ALL_LABELS, average="macro", zero_division=0)

os.makedirs(EVAL_DIR, exist_ok=True)
with open(f"{EVAL_DIR}/results_mgm.json", "w") as f:
    json.dump({"accuracy": float(acc), "macro_f1": float(macro_f1),
               "cm": cm.tolist(), "labels": ALL_LABELS}, f, indent=2)

print(f"\n{'='*60}")
print(f"  ProCyon MGM (7B + Transformer + Focal Loss)")
print(f"{'='*60}")
print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"Macro F1: {macro_f1:.4f}")
print(f"\nClassification Report:")
print(report)
print(f"\nConfusion Matrix:")
header = f"{'':>12}"
for l in ALL_LABELS:
    header += f" {l:>10}"
print(header)
for i, label in enumerate(ALL_LABELS):
    row = f"{label:>10}:"
    for j in range(len(ALL_LABELS)):
        row += f" {cm[i][j]:>10}"
    print(row)
