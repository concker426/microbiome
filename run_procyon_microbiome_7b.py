#!/usr/bin/env python3
"""
ProCyon-style 多模态微生物组分类训练
Architecture: MicrobiomeEncoder + Projection + Qwen2.5-7B (LoRA)

Forward:
  abundance_vector(1222-dim) → MicrobiomeEncoder → embedding(768-dim)
  → Projection → LLM hidden(3584-dim) → prepend as virtual token to text → Qwen2.5

Based on google.txt roadmap (Step 3): LLM+Microbiome encoder (like ProCyon)
"""
import os, re, json, random, sys
from collections import Counter
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ── Config ──────────────────────────────────────────────────────────
DATA_DIR = "/hd/liujx/microbiome_llm_project/data/agp_ftp_processed"
TRAIN_DATA = os.path.join(DATA_DIR, "train_set.jsonl")
TEST_DATA = os.path.join(DATA_DIR, "test_set.jsonl")
TRAIN_VECTORS = os.path.join(DATA_DIR, "train_set_vectors.npy")
TEST_VECTORS = os.path.join(DATA_DIR, "test_set_vectors.npy")

OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/saved_models/procyon_microbiome_7b"
EVAL_DIR = "/hd/liujx/microbiome_llm_project/eval_results_procyon_7b"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

MODEL_PATH = "/hd/gcr/hf_models/Qwen2.5-7B-Instruct"

# Encoder dimensions
INPUT_DIM = 1222       # genus-level abundance vector
EMBED_DIM = 768        # microbiome embedding dim
LLM_HIDDEN = 3584      # Qwen2.5-7B hidden size

# Training
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS = 3
LR = 2e-4
MAX_LENGTH = 1024

# Pretrained encoder (optional — if set, loads weights instead of random init)
PRETRAINED_ENCODER = "/hd/liujx/microbiome_llm_project/saved_models/procyon_microbiome_7b/pretrained_encoder.pt"

ALL_LABELS = ["Healthy", "Disease"]


# ═════════════════════════════════════════════════════════════════════
#  ProCyon-style 组件
# ═════════════════════════════════════════════════════════════════════

class MicrobiomeEncoder(nn.Module):
    """Microbiome abundance vector → embedding (modality-specific encoder)"""
    def __init__(self, input_dim=INPUT_DIM, embed_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class ProjectionLayer(nn.Module):
    """Microbiome embedding → LLM hidden space (connector)"""
    def __init__(self, embed_dim=EMBED_DIM, llm_hidden=LLM_HIDDEN):
        super().__init__()
        self.proj = nn.Linear(embed_dim, llm_hidden)

    def forward(self, x):
        return self.proj(x)


class MultimodalMicrobiomeModel(nn.Module):
    """
    ProCyon-style: Encoder + Projection + LLM (LoRA)

    Forward: prepend micro virtual token to text, then run LLM.
    """
    def __init__(self, llm_peft: PeftModel,
                 encoder: MicrobiomeEncoder,
                 projection: ProjectionLayer):
        super().__init__()
        self.llm = llm_peft
        self.encoder = encoder
        self.projection = projection
        # Track the config for HF compatibility
        self.config = llm_peft.config

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        microbiome_vector: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        batch_size, seq_len = input_ids.shape

        # 1. Encode microbiome vector → LLM space
        micro_embeds = self.encoder(microbiome_vector.to(self.encoder.net[0].weight.dtype))    # (B, EMBED_DIM)
        micro_tokens = self.projection(micro_embeds)       # (B, LLM_HIDDEN)
        micro_tokens = micro_tokens.unsqueeze(1)           # (B, 1, LLM_HIDDEN)

        # 2. Get text embeddings from LLM
        text_embeds = self.llm.base_model.model.model.embed_tokens(input_ids)

        # 3. Fuse: prepend micro token to text
        # Cast micro_tokens to match LLM dtype (LLM may be bfloat16 while encoder is float32)
        micro_tokens = micro_tokens.to(text_embeds.dtype)
        combined_embeds = torch.cat([micro_tokens, text_embeds], dim=1)  # (B, S+1, H)

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

        # 6. Forward through LLM (no labels — compute_loss handles them)
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=new_mask,
            **kwargs,
        )
        return outputs

    def generate(self, **kwargs):
        """Generate with multimodal inputs. Handled separately."""
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
    """Dataset with text + microbiome abundance vector"""
    def __init__(self, data, vectors, tokenizer, class_weights, max_length=1024):
        self.data = data
        self.vectors = vectors  # numpy array
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.weights = [class_weights.get(item["label"], 1.0) for item in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item["messages"]
        vector = self.vectors[idx].astype("float32")

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
            "microbiome_vector": vector,
            "sample_weight": self.weights[idx],
        }


class MultimodalDataCollator:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id or 0

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        vectors = [item["microbiome_vector"] for item in batch]
        sample_weights = [item.get("sample_weight", 1.0) for item in batch]

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

        # Stack vectors (no padding needed)
        vec_tensor = torch.tensor(np.stack(vectors), dtype=torch.float32)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "microbiome_vector": vec_tensor,
            "sample_weight": torch.tensor(sample_weights, dtype=torch.float),
        }


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_weights = inputs.pop("sample_weight", None)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (B, S+1, V) — micro token prepended

        # Align labels: prepend -100 to match the extra micro token
        new_labels = torch.full(
            (labels.size(0), labels.size(1) + 1), -100,
            device=labels.device, dtype=labels.dtype,
        )
        new_labels[:, 1:] = labels  # (B, S+1)

        shift_logits = logits[..., :-1, :].contiguous()  # (B, S, V)
        shift_labels = new_labels[..., 1:].contiguous()   # (B, S) — matches!

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_logits.size(0), -1)

        mask = (shift_labels != -100).float()
        loss = loss * mask
        per_sample_loss = loss.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        if sample_weights is not None:
            per_sample_loss = per_sample_loss * sample_weights.to(per_sample_loss.device)

        return (per_sample_loss.mean(), outputs) if return_outputs else per_sample_loss.mean()


# ═════════════════════════════════════════════════════════════════════
#  Evaluation
# ═════════════════════════════════════════════════════════════════════

def evaluate_multimodal(model, tokenizer, test_data, test_vectors, device,
                        name="model", max_new_tokens=64):
    """Evaluate multimodal model"""
    model.eval()
    predictions = []
    true_labels = []
    pred_labels = []

    for idx, item in enumerate(test_data):
        messages = item["messages"]
        true_label = item["label"]
        vec = test_vectors[idx]

        # 1. Encode microbiome
        vec_t = torch.from_numpy(vec).float().unsqueeze(0).to(device)
        if next(model.encoder.parameters()).dtype == torch.bfloat16:
            vec_t = vec_t.bfloat16()
        with torch.no_grad():
            micro_embed = model.encoder(vec_t)
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

        # 3. Get text embeddings and combine
        with torch.no_grad():
            text_embeds = model.llm.base_model.model.model.embed_tokens(
                prompt_inputs["input_ids"]
            )
            combined = torch.cat([micro_token, text_embeds], dim=1)  # (1, text_len+1, H)

            # 4. Manual generation loop (generate with inputs_embeds is unreliable via PeftModel)
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

def compute_class_weights(train_data):
    counts = Counter(d["label"] for d in train_data)
    total = sum(counts.values())
    n_classes = len(counts)
    return {label: total / (n_classes * count) for label, count in counts.items()}


def main():
    import numpy as np
    print("=" * 60)
    print("  ProCyon-style 微生物组多模态训练")
    print("  MicrobiomeEncoder + Projection + Qwen2.5-7B (LoRA)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    torch.cuda.empty_cache()

    # ── 1. Load data ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  [1/6] 加载多模态数据")
    print(f"{'='*60}")
    train_data = load_jsonl(TRAIN_DATA)
    test_data = load_jsonl(TEST_DATA)
    train_vectors = np.load(TRAIN_VECTORS).astype(np.float32)
    test_vectors = np.load(TEST_VECTORS).astype(np.float32)

    train_dist = Counter(d["label"] for d in train_data)
    test_dist = Counter(d["label"] for d in test_data)
    class_weights = compute_class_weights(train_data)

    print(f"  训练集: {len(train_data)} 样本, 分布: {dict(train_dist)}")
    print(f"  测试集: {len(test_data)} 样本, 分布: {dict(test_dist)}")
    print(f"  向量维度: {train_vectors.shape}")
    print(f"  类别权重: {class_weights}")

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

    # ── 3. Build ProCyon-style model ──────────────────────────────
    print(f"\n{'='*60}")
    print(f"  [3/6] 构建 ProCyon-style 多模态模型")
    print(f"{'='*60}")

    # LoRA on LLM
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    lora_model = get_peft_model(base_model, lora_config)

    # Encoder + Projection (random init, or load pretrained encoder)
    encoder = MicrobiomeEncoder()

    # Load pretrained encoder weights if available
    pretrained_path = PRETRAINED_ENCODER
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"  加载预训练编码器权重: {pretrained_path}")
        pretrained_state = torch.load(pretrained_path, map_location=device)
        # The pretrained state may have a "net." prefix or be directly compatible
        if list(pretrained_state.keys())[0].startswith("net."):
            encoder.load_state_dict(pretrained_state)
        else:
            # Try to match keys (pretrained saved as full autoencoder encoder keys)
            filtered = {k: v for k, v in pretrained_state.items()
                        if k in encoder.state_dict()}
            if len(filtered) == len(encoder.state_dict()):
                encoder.load_state_dict(filtered)
                print(f"    匹配到 {len(filtered)} 个参数键")
            else:
                print(f"    键不匹配: encoder={len(encoder.state_dict())}, "
                      f"pretrained={len(pretrained_state)}, matched={len(filtered)}")
                # Fall back to random init, but log warning
                print(f"    ⚠ 键不匹配，使用随机初始化")

    else:
        print(f"  随机初始化编码器 (未找到预训练权重)")
        if pretrained_path:
            print(f"    (期望路径: {pretrained_path})")

    projection = ProjectionLayer()
    # Convert encoder + projection to bfloat16 to match LLM precision
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
    print(f"    Encoder: {enc_params:,}")
    print(f"    Projection: {proj_params:,}")
    print(f"    LoRA: {lora_params:,}")

    # ── 4. Zero-shot eval (base LLM, text-only baseline) ──────────
    print(f"\n{'='*60}")
    print(f"  [4/6] 零样本评估 (跳过 — 训练后统一评测)")
    print(f"{'='*60}")

    results_before = {"accuracy": 0, "macro_f1": 0, "weighted_f1": 0,
                      "predictions": []}
    print("  (释放 GPU 内存以用于训练)")

    torch.cuda.empty_cache()

    # ── 5. Train ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  [5/6] 多模态 LoRA 微调 (ProCyon-style)")
    print(f"{'='*60}")

    train_dataset = MultimodalDataset(
        train_data, train_vectors, tokenizer, class_weights, max_length=MAX_LENGTH,
    )
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
    trainer = WeightedTrainer(
        model=multimodal_model, args=training_args,
        train_dataset=train_dataset, data_collator=data_collator,
    )

    print("\n开始训练...")
    train_result = trainer.train()
    final_loss = train_result.training_loss
    print(f"  训练完成！Loss: {final_loss:.4f}")

    # Save LoRA adapter + encoder + projection separately
    multimodal_model.llm.save_pretrained(OUTPUT_DIR)
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "projection_state_dict": projection.state_dict(),
    }, os.path.join(OUTPUT_DIR, "multimodal_components.pt"))
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  模型保存至: {OUTPUT_DIR}/")

    # ── 6. Evaluate ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  [6/6] 多模态评估")
    print(f"{'='*60}")

    # Reload model for eval with use_cache=True
    del multimodal_model, trainer
    torch.cuda.empty_cache()

    eval_base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="cuda:0",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    eval_lora = PeftModel.from_pretrained(eval_base, OUTPUT_DIR)
    eval_lora.config.use_cache = True

    # Reload encoder + projection
    eval_encoder = MicrobiomeEncoder().to(device, dtype=torch.bfloat16)
    eval_projection = ProjectionLayer().to(device, dtype=torch.bfloat16)
    ckpt = torch.load(os.path.join(OUTPUT_DIR, "multimodal_components.pt"), map_location=device)
    eval_encoder.load_state_dict(ckpt["encoder_state_dict"])
    eval_projection.load_state_dict(ckpt["projection_state_dict"])

    eval_multimodal = MultimodalMicrobiomeModel(eval_lora, eval_encoder, eval_projection)
    eval_multimodal.to(device)

    results_after = evaluate_multimodal(
        eval_multimodal, tokenizer, test_data, test_vectors, device,
        name="ProCyon-style (7B + Encoder + LoRA)", max_new_tokens=64,
    )

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ProCyon-style 多模态训练完成!")
    print(f"{'='*60}")
    print(f"{'指标':<25} {'微调后':<22}")
    print(f"{'-'*25} {'-'*22}")

    for k in ["accuracy", "macro_f1", "weighted_f1"]:
        a_val = results_after.get(k, 0)
        print(f"{k:<25} {a_val:<22.4f}")

    results = {
        "model": "ProCyon-style_Microbiome_7B",
        "task": "multimodal_healthy_vs_disease",
        "architecture": "MicrobiomeEncoder(1222->512->768) + Projection(768->3584) + Qwen2.5-7B(LoRA)",
        "class_weights": class_weights,
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
