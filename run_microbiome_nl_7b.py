#!/usr/bin/env python3
"""
ProCyon-style NL microbiome training with enriched explanations.

Architecture:
  genus_token_sequence → MGMEncoder (Transformer × 6) → pooled (768)
  → Projection (768→3584) → prepend as virtual token → Qwen2.5-7B (LoRA)
"""
import os, re, json, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["TORCH_FLASH_ATTN_ENABLED"] = "0"

# Fix flash_attn + deepspeed import errors before any model imports
import fix_flash_attn  # noqa: F401

# Disable deepspeed (incompatible install)
import accelerate.utils.imports as _acc_imports
_acc_imports.is_deepspeed_available = lambda: False
# accelerate.utils.other caches the reference at import time, so patch it too
import accelerate.utils.other as _acc_other
_acc_other.is_deepspeed_available = lambda: False

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

from mgm_encoder import MGMEncoder

# ── Config ──────────────────────────────────────────────────────────
# Toggle between original and augmented data
USE_AUGMENTED = False  # set to True to use augmented (failed) data

if USE_AUGMENTED:
    DATA_DIR = "/hd/liujx/microbiome_llm_project/data/agp_ftp_processed_nl_aug"
    TRAIN_DATA = os.path.join(DATA_DIR, "train_nl_aug.jsonl")
    TEST_DATA = os.path.join(DATA_DIR, "test_nl_aug.jsonl")
    GENUS_DATA_DIR = DATA_DIR
    TRAIN_SEQUENCES = os.path.join(GENUS_DATA_DIR, "train_genus_sequences_aug.npy")
    TRAIN_MASKS = os.path.join(GENUS_DATA_DIR, "train_genus_masks_aug.npy")
    TEST_SEQUENCES = os.path.join("/hd/liujx/microbiome_llm_project/data/agp_ftp_processed", "test_genus_sequences.npy")
    TEST_MASKS = os.path.join("/hd/liujx/microbiome_llm_project/data/agp_ftp_processed", "test_genus_masks.npy")
    OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/saved_models/procyon_nl_7b_aug"
    EVAL_DIR = "/hd/liujx/microbiome_llm_project/eval_results_procyon_nl_7b_aug"
else:
    DATA_DIR = "/hd/liujx/microbiome_llm_project/data/agp_ftp_processed_nl"
    TRAIN_DATA = os.path.join(DATA_DIR, "train_nl.jsonl")
    TEST_DATA = os.path.join(DATA_DIR, "test_nl.jsonl")
    GENUS_DATA_DIR = "/hd/liujx/microbiome_llm_project/data/agp_ftp_processed"
    TRAIN_SEQUENCES = os.path.join(GENUS_DATA_DIR, "train_genus_sequences.npy")
    TRAIN_MASKS = os.path.join(GENUS_DATA_DIR, "train_genus_masks.npy")
    TEST_SEQUENCES = os.path.join(GENUS_DATA_DIR, "test_genus_sequences.npy")
    TEST_MASKS = os.path.join(GENUS_DATA_DIR, "test_genus_masks.npy")
    OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/saved_models/procyon_nl_7b"
    EVAL_DIR = "/hd/liujx/microbiome_llm_project/eval_results_procyon_nl_7b"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

MODEL_PATH = "/hd/gcr/hf_models/Qwen2.5-7B-Instruct"

# MGM Encoder dims
VOCAB_SIZE = 1226
EMBED_DIM = 768
LLM_HIDDEN = 3584
MAX_SEQ_LEN = 86

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

PRETRAINED_ENCODER = "/hd/liujx/microbiome_llm_project/saved_models/mgm_encoder_pretrained/mgm_encoder.pt"

ALL_LABELS = ["Healthy", "Disease"]


def load_tokenizer(model_path):
    """Load Qwen2.5 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


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


# ═════════════════════════════════════════════════════════════════════
#  Model Components
# ═════════════════════════════════════════════════════════════════════

class ProjectionLayer(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, llm_hidden=LLM_HIDDEN):
        super().__init__()
        self.proj = nn.Linear(embed_dim, llm_hidden)

    def forward(self, x):
        return self.proj(x)


class MultimodalNLModel(nn.Module):
    def __init__(self, llm_peft: PeftModel,
                 encoder: MGMEncoder,
                 projection: ProjectionLayer):
        super().__init__()
        self.llm = llm_peft
        self.encoder = encoder
        self.projection = projection
        self.config = llm_peft.config

    def gradient_checkpointing_enable(self, **kwargs):
        self.llm.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        self.llm.enable_input_require_grads()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        genus_ids: Optional[torch.Tensor] = None,
        genus_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        device = next(self.encoder.parameters()).device
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
        if genus_ids is not None:
            genus_ids = genus_ids.to(device)
        if genus_mask is not None:
            genus_mask = genus_mask.to(device)

        batch_size, seq_len = input_ids.shape

        micro_embeds = self.encoder(genus_ids, genus_mask)
        micro_embeds = micro_embeds.to(self.projection.proj.weight.dtype)
        micro_tokens = self.projection(micro_embeds).unsqueeze(1)

        text_embeds = self.llm.base_model.model.model.embed_tokens(input_ids)
        micro_tokens = micro_tokens.to(text_embeds.dtype)
        combined_embeds = torch.cat([micro_tokens, text_embeds], dim=1)

        if labels is not None:
            new_labels = torch.full(
                (batch_size, seq_len + 1), -100,
                device=labels.device, dtype=labels.dtype,
            )
            new_labels[:, 1:] = labels
        else:
            new_labels = None

        if attention_mask is not None:
            new_mask = torch.ones(
                batch_size, seq_len + 1,
                device=attention_mask.device, dtype=attention_mask.dtype,
            )
            new_mask[:, 1:] = attention_mask
        else:
            new_mask = None

        # Explicitly create position_ids on the same device as the embeddings
        # to avoid device mismatch when using device_map="auto"
        seq_len_combined = combined_embeds.shape[1]
        position_ids = torch.arange(
            seq_len_combined, dtype=torch.long, device=combined_embeds.device
        ).unsqueeze(0)

        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=new_mask,
            position_ids=position_ids,
            labels=new_labels,
            **kwargs,
        )
        return outputs


# ═════════════════════════════════════════════════════════════════════
#  Dataset
# ═════════════════════════════════════════════════════════════════════

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def balance_data(data, sequences, masks, seed=42):
    """Oversample Disease entries to match Healthy count."""
    labels = [d["label"] for d in data]
    healthy_idx = [i for i, l in enumerate(labels) if l == "Healthy"]
    disease_idx = [i for i, l in enumerate(labels) if l == "Disease"]
    print(f"  Class balance: {len(healthy_idx)} Healthy, {len(disease_idx)} Disease "
          f"(ratio {len(healthy_idx)/max(len(disease_idx),1):.1f}:1)")
    if len(disease_idx) == 0 or len(healthy_idx) <= len(disease_idx):
        return data, sequences, masks
    rng = random.Random(seed)
    n_healthy = len(healthy_idx)
    n_disease = len(disease_idx)
    # Replicate disease entries to match healthy count
    replicates = n_healthy // n_disease
    remainder = n_healthy % n_disease
    extra = rng.sample(disease_idx, remainder) if remainder else []
    dup_indices = disease_idx * replicates + extra
    new_data = list(data)
    if isinstance(sequences, np.ndarray):
        new_seqs = [sequences[i] for i in range(len(data))]
        new_masks = [masks[i] for i in range(len(data))]
        for i in dup_indices:
            new_data.append(data[i])
            new_seqs.append(sequences[i])
            new_masks.append(masks[i])
        new_seqs = np.stack(new_seqs)
        new_masks = np.stack(new_masks)
    else:
        new_seqs = list(sequences)
        new_masks = list(masks)
        for i in dup_indices:
            new_data.append(data[i])
            new_seqs.append(sequences[i])
            new_masks.append(masks[i])
    print(f"  Balanced: {sum(1 for d in new_data if d['label']=='Healthy')} Healthy, "
          f"{sum(1 for d in new_data if d['label']=='Disease')} Disease")
    return new_data, new_seqs, new_masks


class NLMultimodalDataset(Dataset):
    def __init__(self, data, sequences, masks, tokenizer, max_length=1024):
        self.data = data
        self.sequences = sequences
        self.masks = masks
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Pre-encode all samples for speed
        def _tokenize(msgs, **kw):
            r = tokenizer.apply_chat_template(msgs, tokenize=True, **kw)
            # Handle BatchEncoding from newer transformers versions
            if hasattr(r, 'input_ids'):
                return r.input_ids
            return r

        self.encoded = []
        for item in data:
            messages = item["messages"]
            full_ids = _tokenize(messages, max_length=max_length, truncation=True)
            prompt_ids = _tokenize([messages[0]], max_length=max_length,
                                   truncation=True, add_generation_prompt=True)
            user_len = len(prompt_ids)

            labels = [-100] * len(full_ids)
            for i in range(user_len, len(full_ids)):
                labels[i] = full_ids[i]

            self.encoded.append({
                "input_ids": full_ids,
                "labels": labels,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        enc = self.encoded[idx]
        seq = self.sequences[idx].astype(np.int64)
        msk = self.masks[idx]
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": [1] * len(enc["input_ids"]),
            "labels": enc["labels"],
            "genus_ids": seq,
            "genus_mask": msk,
        }


class NLDataCollator:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id or 0

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        genus_ids = [item["genus_ids"] for item in batch]
        genus_mask = [item["genus_mask"] for item in batch]

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

        # Pad genus sequences
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
        }


# ═════════════════════════════════════════════════════════════════════
#  Evaluation
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_nl(model, tokenizer, test_data, test_sequences, test_masks, device,
                name="model", max_new_tokens=128):
    model.eval()

    task_groups = {}
    for i, item in enumerate(test_data):
        task_groups.setdefault(item["task_type"], []).append((i, item))

    all_predictions = []

    for task_type, items in task_groups.items():
        print(f"\n  Evaluating {task_type} ({len(items)} samples)...")

        for batch_idx, (data_idx, item) in enumerate(items):
            true_label = item["label"]
            messages = item["messages"]

            # Get genus sequence
            seq = test_sequences[data_idx if data_idx < len(test_sequences) else 0]
            if isinstance(seq, np.ndarray) and seq.ndim == 2:
                seq = seq[0]
            msk = test_masks[data_idx if data_idx < len(test_masks) else 0]
            if isinstance(msk, np.ndarray) and msk.ndim == 2:
                msk = msk[0]

            seq = seq[:MAX_SEQ_LEN]
            msk = msk[:MAX_SEQ_LEN]

            genus_ids = torch.from_numpy(np.asarray(seq).astype(np.int64)).long().unsqueeze(0).to(device)
            genus_mask = torch.from_numpy(np.asarray(msk)).bool().unsqueeze(0).to(device)

            # Encode microbiome
            micro_embed = model.encoder(genus_ids, genus_mask)
            micro_embed = micro_embed.to(model.projection.proj.weight.dtype)
            micro_token = model.projection(micro_embed).unsqueeze(1)

            # Build prompt
            prompt = tokenizer.apply_chat_template(
                [messages[0]], tokenize=False, add_generation_prompt=True,
            )
            prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                       max_length=MAX_LENGTH).to(device)

            # Generate
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

            predicted_label = extract_label(generated) if task_type == "diagnosis" else None

            all_predictions.append({
                "sample_idx": data_idx,
                "task_type": task_type,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "generated": generated.strip()[:500],
            })

            if (batch_idx + 1) % 100 == 0:
                print(f"    {batch_idx+1}/{len(items)}", flush=True)

    return all_predictions


def print_eval_results(predictions, name="model"):
    from sklearn.metrics import classification_report, accuracy_score, f1_score

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    diag_preds = [p for p in predictions if p["task_type"] == "diagnosis" and p["predicted_label"] is not None]
    if diag_preds:
        true = [p["true_label"] for p in diag_preds]
        pred = [p["predicted_label"] for p in diag_preds]
        acc = accuracy_score(true, pred)
        macro_f1 = f1_score(true, pred, labels=ALL_LABELS, average="macro", zero_division=0)
        print(f"\n  Diagnosis Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Diagnosis Macro F1:  {macro_f1:.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(true, pred, labels=ALL_LABELS, zero_division=0))

    for task_type in ["diagnosis", "marker_analysis", "comparison"]:
        examples = [p for p in predictions if p["task_type"] == task_type]
        if examples:
            print(f"\n  --- {task_type} examples ---")
            for label_type in ["Healthy", "Disease"]:
                ex = [e for e in examples if e["true_label"] == label_type]
                if ex:
                    print(f"\n  [{label_type}] Generated:\n    {ex[0]['generated'][:300]}\n")


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  ProCyon NL Microbiome Training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    torch.cuda.empty_cache()

    # ── 1. Data ──
    print(f"\n{'='*60}")
    print(f"  [1/6] Loading NL enriched data")
    print(f"{'='*60}")

    train_data = load_jsonl(TRAIN_DATA)
    test_data = load_jsonl(TEST_DATA)

    train_sequences = np.load(TRAIN_SEQUENCES).astype(np.int64)
    train_masks = np.load(TRAIN_MASKS)
    test_sequences = np.load(TEST_SEQUENCES).astype(np.int64)
    test_masks = np.load(TEST_MASKS)

    n_tasks = 3  # diagnosis, marker_analysis, comparison
    expanded_train_seqs = np.repeat(train_sequences, n_tasks, axis=0)[:len(train_data)]
    expanded_train_masks = np.repeat(train_masks, n_tasks, axis=0)[:len(train_data)]
    expanded_test_seqs = np.repeat(test_sequences, n_tasks, axis=0)[:len(test_data)]
    expanded_test_masks = np.repeat(test_masks, n_tasks, axis=0)[:len(test_data)]

    train_tasks = Counter(d["task_type"] for d in train_data)
    print(f"  Enriched train: {len(train_data)} ({dict(train_tasks)})")
    print(f"  Enriched test:  {len(test_data)}")

    # ── 2. Tokenizer + LLM ──
    print(f"\n{'='*60}")
    print(f"  [2/6] Loading Qwen2.5-7B")
    print(f"{'='*60}")
    tokenizer = load_tokenizer(MODEL_PATH)
    print(f"  Tokenizer: vocab={tokenizer.vocab_size}")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    base_model.config.use_cache = False
    print(f"  Base model: {sum(p.numel() for p in base_model.parameters())/1e9:.2f}B")

    # ── 3. Build multimodal model ──
    print(f"\n{'='*60}")
    print(f"  [3/6] Building multimodal NL model")
    print(f"{'='*60}")

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.gradient_checkpointing_enable()

    encoder = MGMEncoder(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
        n_layers=MGM_LAYERS, n_heads=MGM_HEADS, ffn_dim=MGM_FFN_DIM,
        max_seq_len=MAX_SEQ_LEN, dropout=MGM_DROPOUT,
    )

    if PRETRAINED_ENCODER and os.path.exists(PRETRAINED_ENCODER):
        print(f"  Loading pretrained MGM encoder: {PRETRAINED_ENCODER}")
        pretrained_state = torch.load(PRETRAINED_ENCODER, map_location=device)
        matched = {k: v for k, v in pretrained_state.items() if k in encoder.state_dict()}
        if len(matched) > 0:
            encoder.load_state_dict(matched, strict=False)
            print(f"    ✓ {len(matched)} params loaded")

    projection = ProjectionLayer()

    multimodal_model = MultimodalNLModel(lora_model, encoder, projection)
    # Tell Trainer this is already model-parallel (from device_map="auto")
    # so it won't wrap in DataParallel which breaks cross-device routing
    multimodal_model.is_parallelizable = True
    multimodal_model.model_parallel = True
    if hasattr(lora_model, "hf_device_map"):
        multimodal_model.hf_device_map = lora_model.hf_device_map

    # Move encoder + projection to device 0 (where LLM inputs need to be)
    encoder.to("cuda:0")
    projection.to("cuda:0")

    trainable = sum(p.numel() for p in multimodal_model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,}")

    # ── 4. Train ──
    print(f"\n{'='*60}")
    print(f"  [4/6] NL generation training")
    print(f"{'='*60}")

    # Balance classes: oversample Disease to match Healthy
    BALANCE_CLASSES = True
    if BALANCE_CLASSES:
        balanced_data, balanced_seqs, balanced_masks = balance_data(
            train_data, expanded_train_seqs, expanded_train_masks
        )
    else:
        balanced_data, balanced_seqs, balanced_masks = \
            train_data, expanded_train_seqs, expanded_train_masks

    train_dataset = NLMultimodalDataset(
        balanced_data, balanced_seqs, balanced_masks,
        tokenizer, max_length=MAX_LENGTH,
    )

    # Enable gradient checkpointing for memory efficiency
    multimodal_model.llm.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        report_to="none",
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    data_collator = NLDataCollator(tokenizer, max_length=MAX_LENGTH)

    trainer = Trainer(
        model=multimodal_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("\nStarting NL generation training...")
    train_result = trainer.train()
    final_loss = train_result.training_loss
    print(f"  Done! Loss: {final_loss:.4f}")

    # Save
    multimodal_model.llm.save_pretrained(OUTPUT_DIR)
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "projection_state_dict": projection.state_dict(),
    }, os.path.join(OUTPUT_DIR, "multimodal_components.pt"))
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  Saved to {OUTPUT_DIR}/")

    # ── 5. Evaluate ──
    print(f"\n{'='*60}")
    print(f"  [5/6] Evaluating NL generation")
    print(f"{'='*60}")

    del multimodal_model, trainer
    torch.cuda.empty_cache()

    eval_base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    eval_lora = PeftModel.from_pretrained(eval_base, OUTPUT_DIR)
    eval_lora.config.use_cache = True

    eval_encoder = MGMEncoder(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
        n_layers=MGM_LAYERS, n_heads=MGM_HEADS, ffn_dim=MGM_FFN_DIM,
        max_seq_len=MAX_SEQ_LEN, dropout=MGM_DROPOUT,
    )
    eval_projection = ProjectionLayer()
    ckpt = torch.load(os.path.join(OUTPUT_DIR, "multimodal_components.pt"), map_location=device)
    eval_encoder.load_state_dict(ckpt["encoder_state_dict"])
    eval_projection.load_state_dict(ckpt["projection_state_dict"])
    eval_encoder.to(eval_lora.device, dtype=torch.bfloat16)
    eval_projection.to(eval_lora.device, dtype=torch.bfloat16)

    eval_model = MultimodalNLModel(eval_lora, eval_encoder, eval_projection)

    predictions = evaluate_nl(
        eval_model, tokenizer, test_data,
        expanded_test_seqs, expanded_test_masks, device,
        name="ProCyon NL (7B + MGMEncoder)", max_new_tokens=128,
    )
    print_eval_results(predictions, name="ProCyon NL (7B + MGMEncoder)")

    with open(os.path.join(EVAL_DIR, "predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    results = {
        "model": "ProCyon_NL_7B",
        "training_loss": float(final_loss),
    }
    with open(os.path.join(EVAL_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Done! Results in {EVAL_DIR}/")


if __name__ == "__main__":
    main()
