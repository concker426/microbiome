#!/usr/bin/env python3
"""
ProCyon-style 多模态微生物组分类训练 — MGM Transformer Encoder version.

Architecture:
  genus_token_sequence → MGMEncoder (Transformer × 6) → pooled embedding (768-dim)
  → Projection (768→3584) → prepend as virtual token → Qwen2.5-7B (LoRA)

Key differences from MLP version:
  - MGMEncoder: Transformer with self-attention over sorted genus tokens
  - Pre-training: next-genus prediction (instead of denoising autoencoder)
  - Class imbalance: Focal Loss + balanced batch sampling
  - Input: genus token IDs (not raw 1222-dim vectors)
"""
import os, re, json, random, sys
from collections import Counter
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from mgm_encoder import MGMEncoder

# ── Config ──────────────────────────────────────────────────────────
DATA_DIR = "/hd/liujx/microbiome_llm_project/data/agp_ftp_processed"
TRAIN_DATA = os.path.join(DATA_DIR, "train_set.jsonl")
TEST_DATA = os.path.join(DATA_DIR, "test_set.jsonl")
TRAIN_SEQUENCES = os.path.join(DATA_DIR, "train_genus_sequences.npy")
TRAIN_MASKS = os.path.join(DATA_DIR, "train_genus_masks.npy")
TEST_SEQUENCES = os.path.join(DATA_DIR, "test_genus_sequences.npy")
TEST_MASKS = os.path.join(DATA_DIR, "test_genus_masks.npy")
VOCAB_PATH = os.path.join(DATA_DIR, "genus_vocab.json")

OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/saved_models/procyon_mgm_7b"
EVAL_DIR = "/hd/liujx/microbiome_llm_project/eval_results_procyon_mgm_7b"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

MODEL_PATH = "/hd/gcr/hf_models/Qwen2.5-7B-Instruct"

# MGM Encoder dimensions
VOCAB_SIZE = 1226       # 1223 genera + 3 special tokens
EMBED_DIM = 768          # microbiome embedding dim
LLM_HIDDEN = 3584        # Qwen2.5-7B hidden size
MAX_SEQ_LEN = 86         # max genus tokens (matches pretrained encoder, covers P99)

# MGM Architecture
MGM_LAYERS = 6
MGM_HEADS = 8
MGM_FFN_DIM = 2048
MGM_DROPOUT = 0.1

# Training
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS = 3
LR = 2e-4
MAX_LENGTH = 1024

# Pretrained MGM encoder (optional)
PRETRAINED_ENCODER = "/hd/liujx/microbiome_llm_project/saved_models/mgm_encoder_pretrained/mgm_encoder.pt"

# Focal Loss
FOCAL_GAMMA = 2.0

ALL_LABELS = ["Healthy", "Disease"]


# ═════════════════════════════════════════════════════════════════════
#  ProCyon-style 组件 (MGM version)
# ═════════════════════════════════════════════════════════════════════

class ProjectionLayer(nn.Module):
    """Microbiome embedding → LLM hidden space (connector)"""
    def __init__(self, embed_dim=EMBED_DIM, llm_hidden=LLM_HIDDEN):
        super().__init__()
        self.proj = nn.Linear(embed_dim, llm_hidden)

    def forward(self, x):
        return self.proj(x)


class MultimodalMicrobiomeModel(nn.Module):
    """
    ProCyon-style: MGMEncoder + Projection + LLM (LoRA)
    """
    def __init__(self, llm_peft: PeftModel,
                 encoder: MGMEncoder,
                 projection: ProjectionLayer):
        super().__init__()
        self.llm = llm_peft
        self.encoder = encoder
        self.projection = projection
        self.config = llm_peft.config

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        genus_ids: Optional[torch.Tensor] = None,
        genus_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        batch_size, seq_len = input_ids.shape

        # 1. Encode genus sequence → pooled embedding → LLM space
        # MGMEncoder returns (B, EMBED_DIM) via attention pooling
        micro_embeds = self.encoder(genus_ids, genus_mask)        # (B, EMBED_DIM)
        micro_embeds = micro_embeds.to(self.projection.proj.weight.dtype)
        micro_tokens = self.projection(micro_embeds)               # (B, LLM_HIDDEN)
        micro_tokens = micro_tokens.unsqueeze(1)                   # (B, 1, LLM_HIDDEN)

        # 2. Get text embeddings from LLM
        text_embeds = self.llm.base_model.model.model.embed_tokens(input_ids)

        # 3. Fuse: prepend micro token to text
        micro_tokens = micro_tokens.to(text_embeds.dtype)
        combined_embeds = torch.cat([micro_tokens, text_embeds], dim=1)

        # 4. Adjust labels: prepend -100 (ignore micro token)
        if labels is not None:
            new_labels = torch.full(
                (batch_size, seq_len + 1), -100,
                device=labels.device, dtype=labels.dtype,
            )
            new_labels[:, 1:] = labels
        else:
            new_labels = None

        # 5. Adjust attention mask
        if attention_mask is not None:
            new_mask = torch.ones(
                batch_size, seq_len + 1,
                device=attention_mask.device, dtype=attention_mask.dtype,
            )
            new_mask[:, 1:] = attention_mask
        else:
            new_mask = None

        # 6. Forward through LLM
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=new_mask,
            **kwargs,
        )
        return outputs

    def generate(self, **kwargs):
        return self.llm.generate(**kwargs)


# ═════════════════════════════════════════════════════════════════════
#  Dataset / Collator / Trainer
# ═════════════════════════════════════════════════════════════════════

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_label(text: str) -> Optional[str]:
    m = re.search(r'诊断结果[：:]\s*(\S+)', text)
    if m:
        label = m.group(1).strip('。，, \n')
        if label in ALL_LABELS:
            return label
    for kw in ALL_LABELS:
        if kw in text:
            return kw
    cn_map = {'健康': 'Healthy', '疾病': 'Disease'}
    for cn, en in cn_map.items():
        if cn in text:
            return en
    return None


def tokenize_chat(tokenizer, messages, max_length=1024, add_generation_prompt=False):
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True,
        add_generation_prompt=add_generation_prompt,
        max_length=max_length, truncation=True,
    )
    return encoded.input_ids if hasattr(encoded, 'input_ids') else encoded


class MultimodalDataset(Dataset):
    """Dataset with text + genus token sequences"""
    def __init__(self, data, sequences, masks, tokenizer, max_length=1024):
        self.data = data
        self.sequences = sequences  # numpy array (N, seq_len)
        self.masks = masks          # numpy array (N, seq_len) bool
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Balanced sample weights (inverse frequency)
        counts = Counter(d["label"] for d in data)
        total = len(data)
        self.sample_weights = []
        for item in data:
            self.sample_weights.append(total / (len(counts) * counts[item["label"]]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item["messages"]
        genus_ids = self.sequences[idx].astype(np.int64)
        genus_msk = self.masks[idx]

        full_ids = tokenize_chat(self.tokenizer, messages, self.max_length, add_generation_prompt=False)
        user_ids = tokenize_chat(self.tokenizer, [messages[0]], self.max_length, add_generation_prompt=True)
        user_len = min(len(user_ids), self.max_length - 5)

        input_ids = full_ids[:self.max_length]
        labels = [-100] * len(input_ids)
        for i in range(user_len, len(input_ids)):
            labels[i] = input_ids[i]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "genus_ids": genus_ids,
            "genus_mask": genus_msk,
            "sample_weight": self.sample_weights[idx],
        }


class MultimodalDataCollator:
    def __init__(self, tokenizer, max_length=1024, pad_genus_ids=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id or 0

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        genus_ids = [item["genus_ids"] for item in batch]
        genus_mask = [item["genus_mask"] for item in batch]
        sample_weights = [item.get("sample_weight", 1.0) for item in batch]

        # Pad text
        max_len = min(max(len(ids) for ids in input_ids), self.max_length)
        padded_ids, padded_mask, padded_labels = [], [], []
        for i in range(len(input_ids)):
            ids = input_ids[i]
            mask = attention_mask[i]
            lbl = labels[i]
            pad_len = max_len - len(ids)
            if pad_len > 0:
                padded_ids.append(ids + [self.pad_token_id] * pad_len)
                padded_mask.append(mask + [0] * pad_len)
                padded_labels.append(lbl + [-100] * pad_len)
            else:
                padded_ids.append(ids[:max_len])
                padded_mask.append(mask[:max_len])
                padded_labels.append(lbl[:max_len])

        # Pad genus sequences (variable length per sample, truncate to MAX_SEQ_LEN)
        truncated_gids = [g[:MAX_SEQ_LEN] for g in genus_ids]
        truncated_gmask = [m[:MAX_SEQ_LEN] for m in genus_mask]
        max_genus_len = max(len(g) for g in truncated_gids)
        padded_gids, padded_gmask = [], []
        for i in range(len(truncated_gids)):
            gids = truncated_gids[i]
            gmask = truncated_gmask[i]
            pad_len = max_genus_len - len(gids)
            if pad_len > 0:
                padded_gids.append(np.pad(gids, (0, pad_len), constant_values=0))
                padded_gmask.append(np.pad(gmask, (0, pad_len), constant_values=False))
            else:
                padded_gids.append(gids)
                padded_gmask.append(gmask)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "genus_ids": torch.tensor(np.stack(padded_gids), dtype=torch.long),
            "genus_mask": torch.tensor(np.stack(padded_gmask), dtype=torch.bool),
            "sample_weight": torch.tensor(sample_weights, dtype=torch.float),
        }


class FocalLossTrainer(Trainer):
    """
    Trainer with Focal Loss for class imbalance.
    FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_weights = inputs.pop("sample_weight", None)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (B, S+1, V)

        # Align labels: prepend -100 to match micro token
        new_labels = torch.full(
            (labels.size(0), labels.size(1) + 1), -100,
            device=labels.device, dtype=labels.dtype,
        )
        new_labels[:, 1:] = labels

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = new_labels[..., 1:].contiguous()

        # Focal Loss
        log_probs = F.log_softmax(shift_logits, dim=-1)
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )
        ce_loss = ce_loss.view(shift_logits.size(0), -1)

        # Compute p_t = exp(-CE) for focal weighting
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** FOCAL_GAMMA

        loss = focal_weight * ce_loss

        # Mask padding positions
        mask = (shift_labels != -100).float()
        loss = loss * mask
        per_sample_loss = loss.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        if sample_weights is not None:
            per_sample_loss = per_sample_loss * sample_weights.to(per_sample_loss.device)

        return (per_sample_loss.mean(), outputs) if return_outputs else per_sample_loss.mean()


# ═════════════════════════════════════════════════════════════════════
#  Evaluation
# ═════════════════════════════════════════════════════════════════════

def evaluate_multimodal(model, tokenizer, test_data, test_sequences, test_masks, device,
                        name="model", max_new_tokens=64):
    """Evaluate MGM-based multimodal model"""
    model.eval()
    predictions = []
    true_labels = []
    pred_labels = []

    for idx, item in enumerate(test_data):
        messages = item["messages"]
        true_label = item["label"]

        # 1. Encode genus sequence (truncate to MAX_SEQ_LEN)
        seq = test_sequences[idx].astype(np.int64)[:MAX_SEQ_LEN]
        msk = test_masks[idx][:MAX_SEQ_LEN]
        genus_ids = torch.from_numpy(seq).long().unsqueeze(0).to(device)
        genus_mask = torch.from_numpy(msk).bool().unsqueeze(0).to(device)

        with torch.no_grad():
            micro_embed = model.encoder(genus_ids, genus_mask)  # (1, EMBED_DIM)
            micro_embed = micro_embed.to(model.projection.proj.weight.dtype)
            micro_token = model.projection(micro_embed).unsqueeze(1)  # (1, 1, H)

        # 2. Tokenize prompt
        prompt = tokenizer.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True,
        )
        prompt_inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)
        text_len = prompt_inputs["input_ids"].shape[1]

        # 3. Generate
        with torch.no_grad():
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
        predicted_label = extract_label(generated)

        predictions.append({
            "sample_id": item.get("sample_id", ""),
            "true_label": true_label,
            "predicted_label": predicted_label or "UNKNOWN",
            "generated": generated.strip()[:200],
        })
        true_labels.append(true_label)
        pred_labels.append(predicted_label or "UNKNOWN")

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(test_data)}", flush=True)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, labels=ALL_LABELS, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels, labels=ALL_LABELS)
    macro_f1 = f1_score(true_labels, pred_labels, labels=ALL_LABELS, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_labels, pred_labels, labels=ALL_LABELS, average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
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

    return {"accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
            "predictions": predictions, "report": report}


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  ProCyon MGM 多模态训练 (Transformer Encoder)")
    print("  MGMEncoder + Projection + Qwen2.5-7B (LoRA)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    torch.cuda.empty_cache()

    # ── 1. Load data ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  [1/6] 加载 MGM 多模态数据")
    print(f"{'='*60}")
    train_data = load_jsonl(TRAIN_DATA)
    test_data = load_jsonl(TEST_DATA)
    train_sequences = np.load(TRAIN_SEQUENCES).astype(np.int64)
    train_masks = np.load(TRAIN_MASKS)
    test_sequences = np.load(TEST_SEQUENCES).astype(np.int64)
    test_masks = np.load(TEST_MASKS)

    train_dist = Counter(d["label"] for d in train_data)
    test_dist = Counter(d["label"] for d in test_data)

    print(f"  训练集: {len(train_data)} 样本, 分布: {dict(train_dist)}")
    print(f"  测试集: {len(test_data)} 样本, 分布: {dict(test_dist)}")
    print(f"  序列维度: {train_sequences.shape} (P99 len: {np.percentile([m.sum() for m in train_masks], 99):.0f})")

    # ── 2. Load tokenizer + LLM ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  [2/6] 加载 Qwen2.5-7B")
    print(f"{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map={"": f"cuda:0"},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    base_model.config.use_cache = False
    print(f"  基座参数量: {sum(p.numel() for p in base_model.parameters())/1e9:.2f}B")

    # ── 3. Build ProCyon MGM model ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  [3/6] 构建 ProCyon MGM 多模态模型")
    print(f"{'='*60}")

    # LoRA on LLM
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    lora_model = get_peft_model(base_model, lora_config)

    # MGM Encoder
    encoder = MGMEncoder(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
        n_layers=MGM_LAYERS, n_heads=MGM_HEADS, ffn_dim=MGM_FFN_DIM,
        max_seq_len=MAX_SEQ_LEN, dropout=MGM_DROPOUT,
    )

    # Load pretrained MGM encoder weights if available
    pretrained_path = PRETRAINED_ENCODER
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"  加载 MGM 预训练编码器: {pretrained_path}")
        pretrained_state = torch.load(pretrained_path, map_location=device)
        # Filter to matching keys
        encoder_state = encoder.state_dict()
        matched = {k: v for k, v in pretrained_state.items() if k in encoder_state}
        if len(matched) == len(encoder_state):
            encoder.load_state_dict(matched)
            print(f"    ✓ 成功加载所有 {len(matched)} 个参数")
        else:
            print(f"    ⚠ 部分匹配: {len(matched)}/{len(encoder_state)}")
            # Load what we can
            encoder.load_state_dict(matched, strict=False)
    else:
        print(f"  随机初始化 MGM Encoder (未找到预训练权重)")
        if pretrained_path:
            print(f"    (期望路径: {pretrained_path})")

    projection = ProjectionLayer()

    # Convert encoder + projection to bfloat16
    encoder.to(device, dtype=torch.bfloat16)
    projection.to(device, dtype=torch.bfloat16)

    multimodal_model = MultimodalMicrobiomeModel(lora_model, encoder, projection)
    multimodal_model.to(device)

    # Count trainable params
    trainable = sum(p.numel() for p in multimodal_model.parameters() if p.requires_grad)
    enc_params = sum(p.numel() for p in encoder.parameters())
    proj_params = sum(p.numel() for p in projection.parameters())
    lora_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"  可训练参数总数: {trainable:,}")
    print(f"    MGM Encoder: {enc_params:,}")
    print(f"    Projection: {proj_params:,}")
    print(f"    LoRA: {lora_params:,}")

    # ── 4. Train ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  [4/6] MGM 多模态 LoRA 微调 (Focal Loss)")
    print(f"{'='*60}")

    train_dataset = MultimodalDataset(
        train_data, train_sequences, train_masks, tokenizer, max_length=MAX_LENGTH,
    )

    # Balanced sampler
    labels_arr = [d["label"] for d in train_data]
    class_counts = Counter(labels_arr)
    weights = [1.0 / class_counts[l] for l in labels_arr]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_data), replacement=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        report_to="none",
    )
    data_collator = MultimodalDataCollator(tokenizer, max_length=MAX_LENGTH)
    trainer = FocalLossTrainer(
        model=multimodal_model, args=training_args,
        train_dataset=train_dataset, data_collator=data_collator,
    )

    print("\n开始训练 (Focal Loss γ=2.0, 平衡采样)...")
    train_result = trainer.train()
    final_loss = train_result.training_loss
    print(f"  训练完成！Loss: {final_loss:.4f}")

    # Save
    multimodal_model.llm.save_pretrained(OUTPUT_DIR)
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "projection_state_dict": projection.state_dict(),
    }, os.path.join(OUTPUT_DIR, "multimodal_components.pt"))
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  模型保存至: {OUTPUT_DIR}/")

    # ── 5. Evaluate ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  [5/6] MGM 多模态评估")
    print(f"{'='*60}")

    # Reload
    del multimodal_model, trainer
    torch.cuda.empty_cache()

    eval_base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="cuda:0",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    eval_lora = PeftModel.from_pretrained(eval_base, OUTPUT_DIR)
    eval_lora.config.use_cache = True

    eval_encoder = MGMEncoder(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
        n_layers=MGM_LAYERS, n_heads=MGM_HEADS, ffn_dim=MGM_FFN_DIM,
        max_seq_len=MAX_SEQ_LEN, dropout=MGM_DROPOUT,
    ).to(device, dtype=torch.bfloat16)
    eval_projection = ProjectionLayer().to(device, dtype=torch.bfloat16)
    ckpt = torch.load(os.path.join(OUTPUT_DIR, "multimodal_components.pt"), map_location=device)
    eval_encoder.load_state_dict(ckpt["encoder_state_dict"])
    eval_projection.load_state_dict(ckpt["projection_state_dict"])

    eval_multimodal = MultimodalMicrobiomeModel(eval_lora, eval_encoder, eval_projection)
    eval_multimodal.to(device)

    results_after = evaluate_multimodal(
        eval_multimodal, tokenizer, test_data, test_sequences, test_masks, device,
        name="ProCyon MGM (7B + MGMEncoder + FocalLoss)", max_new_tokens=64,
    )

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ProCyon MGM 多模态训练完成!")
    print(f"{'='*60}")

    results = {
        "model": "ProCyon_MGM_7B",
        "task": "multimodal_healthy_vs_disease",
        "architecture": {
            "encoder": f"MGMEncoder(Transformer_{MGM_LAYERS}layers_{MGM_HEADS}heads_{EMBED_DIM}dim)",
            "projection": f"Linear({EMBED_DIM}->{LLM_HIDDEN})",
            "llm": "Qwen2.5-7B(LoRA)",
            "pretrained_encoder": os.path.exists(pretrained_path) if pretrained_path else False,
        },
        "training": {
            "loss_fn": f"FocalLoss(gamma={FOCAL_GAMMA})",
            "sampling": "weighted_random",
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "lr": LR,
            "epochs": EPOCHS,
        },
        "train_dist": dict(train_dist),
        "test_dist": dict(test_dist),
        "training_loss": float(final_loss),
        "results_after": {k: results_after[k] for k in ["accuracy", "macro_f1", "weighted_f1"]},
    }
    with open(os.path.join(EVAL_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(EVAL_DIR, "predictions.json"), "w") as f:
        json.dump(results_after["predictions"], f, indent=2, ensure_ascii=False)

    print(f"\n✅ 完成！结果: {EVAL_DIR}/")


if __name__ == "__main__":
    main()
