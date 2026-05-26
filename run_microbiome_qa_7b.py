#!/usr/bin/env python3
"""
Free-form QA training for microbiome samples (像豆包).

Architecture: Same as NL training (MGMEncoder + Qwen2.5-7B LoRA)
Data format: Diverse QA pairs per sample
Init: Loads from existing NL checkpoint for transfer learning
"""
import os, json, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["TORCH_FLASH_ATTN_ENABLED"] = "0"

import fix_flash_attn  # noqa: F401
import accelerate.utils.imports as _acc_imports
_acc_imports.is_deepspeed_available = lambda: False
import accelerate.utils.other as _acc_other
_acc_other.is_deepspeed_available = lambda: False

import re
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
DATA_DIR = "/hd/liujx/microbiome_llm_project/data/agp_ftp_processed_qa"
TRAIN_DATA = os.path.join(DATA_DIR, "train_qa.jsonl")
TEST_DATA = os.path.join(DATA_DIR, "test_qa.jsonl")
TRAIN_SEQUENCES = os.path.join(DATA_DIR, "train_genus_sequences.npy")
TRAIN_MASKS = os.path.join(DATA_DIR, "train_genus_masks.npy")
TEST_SEQUENCES = os.path.join(DATA_DIR, "test_genus_sequences.npy")
TEST_MASKS = os.path.join(DATA_DIR, "test_genus_masks.npy")

OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/saved_models/procyon_qa_balanced_7b"
EVAL_DIR = "/hd/liujx/microbiome_llm_project/eval_results_procyon_qa_balanced_7b"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

MODEL_PATH = "/hd/gcr/hf_models/Qwen2.5-7B-Instruct"
NL_CHECKPOINT = "/hd/liujx/microbiome_llm_project/saved_models/procyon_nl_7b"

VOCAB_SIZE = 1226
EMBED_DIM = 768
LLM_HIDDEN = 3584
MAX_SEQ_LEN = 86
MGM_LAYERS = 6
MGM_HEADS = 8
MGM_FFN_DIM = 2048
MGM_DROPOUT = 0.1

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS = 3
LR = 1e-4
MAX_LENGTH = 1024

ALL_LABELS = ["Healthy", "Disease"]


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def extract_label(text: str) -> str:
    """Extract diagnosis label from generated text."""
    m = re.search(r'诊断结果[：:]\s*(\S+)', text)
    if m:
        label = m.group(1).strip('。，, \n')
        if label in ALL_LABELS:
            return label
    cn_map = {'健康': 'Healthy', '疾病': 'Disease', '正常': 'Healthy'}
    for cn, en in cn_map.items():
        if cn in text:
            return en
    return None


# ═════════════════════════════════════════════════════════════════════
#  Model Components (same as NL training)
# ═════════════════════════════════════════════════════════════════════

class ProjectionLayer(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, llm_hidden=LLM_HIDDEN):
        super().__init__()
        self.proj = nn.Linear(embed_dim, llm_hidden)

    def forward(self, x):
        return self.proj(x)


class MultimodalQAModel(nn.Module):
    """Same architecture as NL model - just renamed for QA."""
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

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                genus_ids=None, genus_mask=None, **kwargs):
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

        micro_embeds = self.encoder(genus_ids, genus_mask)
        micro_embeds = micro_embeds.to(self.projection.proj.weight.dtype)
        micro_tokens = self.projection(micro_embeds).unsqueeze(1)

        text_embeds = self.llm.base_model.model.model.embed_tokens(input_ids)
        micro_tokens = micro_tokens.to(text_embeds.dtype)
        combined_embeds = torch.cat([micro_tokens, text_embeds], dim=1)

        if labels is not None:
            new_labels = torch.full(
                (input_ids.size(0), input_ids.size(1) + 1), -100,
                device=labels.device, dtype=labels.dtype,
            )
            new_labels[:, 1:] = labels
        else:
            new_labels = None

        if attention_mask is not None:
            new_mask = torch.ones(
                input_ids.size(0), input_ids.size(1) + 1,
                device=attention_mask.device, dtype=attention_mask.dtype,
            )
            new_mask[:, 1:] = attention_mask
        else:
            new_mask = None

        seq_len = combined_embeds.shape[1]
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=combined_embeds.device
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
    with open(path) as f:
        return [json.loads(line) for line in f]


def balance_qa(data, seqs, masks, seed=42):
    """Oversample Disease entries to roughly match Healthy count."""
    healthy_idx = [i for i, d in enumerate(data) if d.get("label") == "Healthy"]
    disease_idx = [i for i, d in enumerate(data) if d.get("label") == "Disease"]
    print(f"  Class balance: Healthy={len(healthy_idx)}, Disease={len(disease_idx)} "
          f"(ratio {len(healthy_idx)/max(len(disease_idx),1):.2f}:1)")
    if not disease_idx or len(healthy_idx) <= len(disease_idx):
        return data, seqs, masks
    rng = random.Random(seed)
    n_h, n_d = len(healthy_idx), len(disease_idx)
    reps = n_h // n_d
    rem = n_h % n_d
    extra = rng.sample(disease_idx, rem) if rem else []
    dup = disease_idx * reps + extra
    new_data = list(data)
    new_seqs = [seqs[i] for i in range(len(data))]
    new_masks = [masks[i] for i in range(len(data))]
    for i in dup:
        new_data.append(data[i])
        new_seqs.append(seqs[i])
        new_masks.append(masks[i])
    print(f"  Balanced: Healthy={sum(1 for d in new_data if d.get('label')=='Healthy')}, "
          f"Disease={sum(1 for d in new_data if d.get('label')=='Disease')}")
    return new_data, np.stack(new_seqs), np.stack(new_masks)


class QADataset(Dataset):
    def __init__(self, data, sequences, masks, tokenizer, max_length=1024):
        self.data = data
        self.sequences = sequences
        self.masks = masks
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded = []
        for item in data:
            messages = item["messages"]
            full_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, max_length=max_length, truncation=True,
            )
            prompt_ids = tokenizer.apply_chat_template(
                [messages[0]], tokenize=True, max_length=max_length,
                truncation=True, add_generation_prompt=True,
            )
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


class QADataCollator:
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
#  Main
# ═════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  ProCyon QA Microbiome Training")
    print("  Free-form Q&A like 豆包")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    torch.cuda.empty_cache()

    # ── 1. Data ──
    print(f"\n{'='*60}")
    print(f"  [1/6] Loading QA data")
    print(f"{'='*60}")
    train_data = load_jsonl(TRAIN_DATA)
    test_data = load_jsonl(TEST_DATA)
    train_sequences = np.load(TRAIN_SEQUENCES).astype(np.int64)
    train_masks = np.load(TRAIN_MASKS)
    test_sequences = np.load(TEST_SEQUENCES).astype(np.int64)
    test_masks = np.load(TEST_MASKS)

    # Expand sequences to match QA count
    # Each sample has multiple QA pairs, sequences are per-sample
    sid_to_seq = {}
    for i, item in enumerate(load_jsonl(os.path.join(
            "/hd/liujx/microbiome_llm_project/data/agp_ftp_processed", "train_set.jsonl"))):
        sid_to_seq[item["sample_id"]] = i

    # Build expanded arrays for QA
    expanded_train_seqs = np.zeros((len(train_data), train_sequences.shape[1]), dtype=np.int64)
    expanded_train_masks = np.zeros((len(train_data), train_masks.shape[1]), dtype=bool)
    for i, item in enumerate(train_data):
        sid = item["sample_id"]
        if sid in sid_to_seq:
            idx = sid_to_seq[sid]
            expanded_train_seqs[i] = train_sequences[idx]
            expanded_train_masks[i] = train_masks[idx]
        else:
            expanded_train_seqs[i] = train_sequences[i % len(train_sequences)]
            expanded_train_masks[i] = train_masks[i % len(train_masks)]

    test_sid_to_seq = {}
    for i, item in enumerate(load_jsonl(os.path.join(
            "/hd/liujx/microbiome_llm_project/data/agp_ftp_processed", "test_set.jsonl"))):
        test_sid_to_seq[item["sample_id"]] = i

    expanded_test_seqs = np.zeros((len(test_data), test_sequences.shape[1]), dtype=np.int64)
    expanded_test_masks = np.zeros((len(test_data), test_masks.shape[1]), dtype=bool)
    for i, item in enumerate(test_data):
        sid = item["sample_id"]
        if sid in test_sid_to_seq:
            idx = test_sid_to_seq[sid]
            expanded_test_seqs[i] = test_sequences[idx]
            expanded_test_masks[i] = test_masks[idx]
        else:
            expanded_test_seqs[i] = test_sequences[i % len(test_sequences)]
            expanded_test_masks[i] = test_masks[i % len(test_masks)]

    print(f"  QA train: {len(train_data)} entries")
    print(f"  QA test:  {len(test_data)} entries")

    # Balance QA classes (oversample Disease to match Healthy)
    print("\n  Balancing QA training set...")
    balanced_train_data, balanced_train_seqs, balanced_train_masks = balance_qa(
        train_data, expanded_train_seqs, expanded_train_masks
    )

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

    # ── 3. Build model ──
    print(f"\n{'='*60}")
    print(f"  [3/6] Building QA model")
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
    projection = ProjectionLayer()

    # Load pretrained MGM encoder
    PRETRAINED_ENCODER = "/hd/liujx/microbiome_llm_project/saved_models/mgm_encoder_pretrained/mgm_encoder.pt"
    if os.path.exists(PRETRAINED_ENCODER):
        pretrained_state = torch.load(PRETRAINED_ENCODER, map_location=device)
        matched = {k: v for k, v in pretrained_state.items() if k in encoder.state_dict()}
        if matched:
            encoder.load_state_dict(matched, strict=False)
            print(f"  ✓ MGM encoder: {len(matched)} params loaded from pretrain")

    # Transfer learning from NL checkpoint (if exists)
    NL_COMPONENTS = os.path.join(NL_CHECKPOINT, "multimodal_components.pt")
    if os.path.exists(NL_COMPONENTS):
        print(f"  Loading NL checkpoint for transfer learning: {NL_COMPONENTS}")
        nl_ckpt = torch.load(NL_COMPONENTS, map_location=device)
        encoder.load_state_dict(nl_ckpt["encoder_state_dict"], strict=False)
        projection.load_state_dict(nl_ckpt["projection_state_dict"])
        print(f"  ✓ Transferred encoder + projection from NL model")
    else:
        print(f"  No NL checkpoint found, starting from scratch")

    model = MultimodalQAModel(lora_model, encoder, projection)
    model.is_parallelizable = True
    model.model_parallel = True
    if hasattr(lora_model, "hf_device_map"):
        model.hf_device_map = lora_model.hf_device_map

    encoder.to("cuda:0")
    projection.to("cuda:0")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,}")

    # ── 4. Train ──
    print(f"\n{'='*60}")
    print(f"  [4/6] QA training")
    print(f"{'='*60}")

    train_dataset = QADataset(
        balanced_train_data, balanced_train_seqs, balanced_train_masks,
        tokenizer, max_length=MAX_LENGTH,
    )
    model.llm.gradient_checkpointing_enable()

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
    data_collator = QADataCollator(tokenizer, max_length=MAX_LENGTH)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("\nStarting QA training...")
    train_result = trainer.train()
    final_loss = train_result.training_loss
    print(f"  Done! Loss: {final_loss:.4f}")

    # Save
    model.llm.save_pretrained(OUTPUT_DIR)
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "projection_state_dict": projection.state_dict(),
    }, os.path.join(OUTPUT_DIR, "multimodal_components.pt"))
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  Saved to {OUTPUT_DIR}/")

    # ── 5. Quick eval ──
    print(f"\n{'='*60}")
    print(f"  [5/6] Quick QA evaluation")
    print(f"{'='*60}")

    # Test a few random samples
    model.eval()
    test_indices = list(range(min(20, len(test_data))))
    with torch.no_grad():
        for idx in test_indices:
            item = test_data[idx]
            seq = expanded_test_seqs[idx % len(expanded_test_seqs)]
            msk = expanded_test_masks[idx % len(expanded_test_masks)]
            if seq.ndim == 2:
                seq = seq[0]
            if msk.ndim == 2:
                msk = msk[0]

            genus_ids = torch.from_numpy(
                np.asarray(seq[:MAX_SEQ_LEN]).astype(np.int64)
            ).long().unsqueeze(0).to(device)
            genus_mask = torch.from_numpy(
                np.asarray(msk[:MAX_SEQ_LEN])
            ).bool().unsqueeze(0).to(device)

            micro_embed = model.encoder(genus_ids, genus_mask)
            micro_embed = micro_embed.to(model.projection.proj.weight.dtype)
            micro_token = model.projection(micro_embed).unsqueeze(1)

            prompt_text = item["messages"][0]["content"]
            prompt = f"问：{prompt_text.replace('问：', '')}"
            prompt_inputs = tokenizer(prompt, return_tensors="pt",
                                       truncation=True, max_length=MAX_LENGTH).to(device)

            text_embeds = model.llm.base_model.model.model.embed_tokens(
                prompt_inputs["input_ids"]
            )
            micro_token = micro_token.to(text_embeds.dtype)
            combined = torch.cat([micro_token, text_embeds], dim=1)
            seq_len = combined.shape[1]
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)

            outputs = model.llm(
                inputs_embeds=combined, position_ids=position_ids, use_cache=True
            )
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids = [next_token]
            current_len = seq_len
            for _ in range(96):
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
            print(f"\n  [{idx}] Q: {item['messages'][0]['content']}")
            print(f"  True: {item['messages'][1]['content'][:150]}")
            print(f"  Pred: {generated[:150]}")

    print(f"\n{'='*60}")
    print(f"  Done! Model saved to {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
