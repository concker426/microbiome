#!/usr/bin/env python3
"""Quick eval-only script for saved ProCyon-style model."""
import json, os, re, sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

DATA_DIR = "/hd/liujx/microbiome_llm_project/data/agp_ftp_processed"
MODEL_PATH = "/hd/gcr/hf_models/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/saved_models/procyon_microbiome_7b"
EVAL_DIR = "/hd/liujx/microbiome_llm_project/eval_results_procyon_7b"
MAX_LENGTH = 1024
ALL_LABELS = ["Healthy", "Disease"]

class MicrobiomeEncoder(nn.Module):
    def __init__(self, input_dim=1222, embed_dim=768):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, embed_dim))
    def forward(self, x): return self.net(x)

class ProjectionLayer(nn.Module):
    def __init__(self, embed_dim=768, llm_hidden=3584):
        super().__init__()
        self.proj = nn.Linear(embed_dim, llm_hidden)
    def forward(self, x): return self.proj(x)

class MultimodalMicrobiomeModel(nn.Module):
    def __init__(self, llm_peft, encoder, projection):
        super().__init__()
        self.llm = llm_peft
        self.encoder = encoder
        self.projection = projection

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

device = torch.device("cuda:0")
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
eval_base = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map={"": "cuda:0"},
    trust_remote_code=True, torch_dtype=torch.bfloat16,
)

print("Loading LoRA adapter...")
eval_lora = PeftModel.from_pretrained(eval_base, OUTPUT_DIR)
eval_lora.config.use_cache = True

print("Loading encoder + projection...")
eval_encoder = MicrobiomeEncoder().to(device, dtype=torch.bfloat16)
eval_projection = ProjectionLayer().to(device, dtype=torch.bfloat16)
ckpt = torch.load(os.path.join(OUTPUT_DIR, "multimodal_components.pt"), map_location=device)
eval_encoder.load_state_dict(ckpt["encoder_state_dict"])
eval_projection.load_state_dict(ckpt["projection_state_dict"])

eval_model = MultimodalMicrobiomeModel(eval_lora, eval_encoder, eval_projection)
eval_model.to(device)
eval_model.eval()

# Load test data
test_data = []
with open(os.path.join(DATA_DIR, "test_set.jsonl")) as f:
    for line in f: test_data.append(json.loads(line))
test_vectors = np.load(os.path.join(DATA_DIR, "test_set_vectors.npy")).astype(np.float32)
print(f"Test data: {len(test_data)} samples")

predictions = []
for idx, item in enumerate(test_data):
    messages = item["messages"]
    true_label = item["label"]
    vec = test_vectors[idx]

    vec_t = torch.from_numpy(vec).float().unsqueeze(0).to(device)
    if next(eval_model.encoder.parameters()).dtype == torch.bfloat16:
        vec_t = vec_t.bfloat16()

    with torch.no_grad():
        micro_embed = eval_model.encoder(vec_t)
        micro_token = eval_model.projection(micro_embed).unsqueeze(1)

    prompt = tokenizer.apply_chat_template(
        [messages[0]], tokenize=False, add_generation_prompt=True,
    )
    prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    text_len = prompt_inputs["input_ids"].shape[1]

    with torch.no_grad():
        text_embeds = eval_model.llm.base_model.model.model.embed_tokens(prompt_inputs["input_ids"])
        combined = torch.cat([micro_token, text_embeds], dim=1)

        seq_len = combined.shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        outputs = eval_model.llm(inputs_embeds=combined, position_ids=position_ids, use_cache=True)

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = [next_token]
        current_len = seq_len

        for _ in range(64):
            pos_id = torch.full((1, 1), current_len, dtype=torch.long, device=device)
            out = eval_model.llm(input_ids=next_token, position_ids=pos_id,
                                 past_key_values=outputs.past_key_values, use_cache=True)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            if next_token.item() == tokenizer.eos_token_id: break
            generated_ids.append(next_token)
            current_len += 1
            outputs.past_key_values = out.past_key_values

    generated_ids = torch.cat(generated_ids, dim=1)
    generated = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    predicted_label = extract_label(generated)

    predictions.append({
        "sample_id": item.get("sample_id", ""),
        "true_label": true_label,
        "predicted_label": predicted_label or "UNKNOWN",
        "generated": generated.strip()[:200],
    })

    if (idx + 1) % 100 == 0:
        print(f"  {idx+1}/{len(test_data)}", flush=True)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

true_labels = [p["true_label"] for p in predictions]
pred_labels = [p["predicted_label"] for p in predictions]

accuracy = accuracy_score(true_labels, pred_labels)
report = classification_report(true_labels, pred_labels, labels=ALL_LABELS, zero_division=0)
cm = confusion_matrix(true_labels, pred_labels, labels=ALL_LABELS)
macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
weighted_f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

print(f"\n{'='*60}")
print("  ProCyon-style (7B + Encoder + LoRA) — Eval Only")
print(f"{'='*60}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Macro F1: {macro_f1:.4f}")
print(f"\nClassification Report:")
print(report)
print(f"\nConfusion Matrix:")
header = f"{'':>12}"
for l in ALL_LABELS: header += f" {l:>10}"
print(header)
for i, label in enumerate(ALL_LABELS):
    row = f"{label:>10}:"
    for j in range(len(ALL_LABELS)): row += f" {cm[i][j]:>10}"
    print(row)

results = {
    "model": "ProCyon-style_Microbiome_7B",
    "accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
    "report": str(report),
}
with open(os.path.join(EVAL_DIR, "results_eval_only.json"), "w") as f:
    json.dump(results, f, indent=2)
with open(os.path.join(EVAL_DIR, "predictions_eval_only.json"), "w") as f:
    json.dump(predictions, f, indent=2, ensure_ascii=False)
print(f"\n✅ Results: {EVAL_DIR}/")
