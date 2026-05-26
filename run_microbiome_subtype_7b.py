#!/usr/bin/env python3
"""
Stage-2 subtyping: classify Disease samples as CD or UC.

Filters AGP-FTP train/test for label_detail in {"CD", "UC"}, builds new
diagnosis prompts ("CD or UC?"), then fine-tunes the same MGMEncoder +
Qwen2.5-7B LoRA architecture used by run_microbiome_nl_7b.py.

Init: copy encoder + projection from procyon_nl_7b for transfer.
"""
import os, json, re, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["TORCH_FLASH_ATTN_ENABLED"] = "0"

import fix_flash_attn  # noqa: F401
import accelerate.utils.imports as _acc_imports
_acc_imports.is_deepspeed_available = lambda: False
import accelerate.utils.other as _acc_other
_acc_other.is_deepspeed_available = lambda: False

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
BASE = "/hd/liujx/microbiome_llm_project"
SRC_TRAIN = os.path.join(BASE, "data/agp_ftp_processed/train_set.jsonl")
SRC_TEST = os.path.join(BASE, "data/agp_ftp_processed/test_set.jsonl")
SRC_TRAIN_SEQ = os.path.join(BASE, "data/agp_ftp_processed/train_genus_sequences.npy")
SRC_TRAIN_MSK = os.path.join(BASE, "data/agp_ftp_processed/train_genus_masks.npy")
SRC_TEST_SEQ = os.path.join(BASE, "data/agp_ftp_processed/test_genus_sequences.npy")
SRC_TEST_MSK = os.path.join(BASE, "data/agp_ftp_processed/test_genus_masks.npy")
SRC_TRAIN_VEC = os.path.join(BASE, "data/agp_ftp_processed/train_set_vectors.npy")
SRC_TEST_VEC = os.path.join(BASE, "data/agp_ftp_processed/test_set_vectors.npy")
SRC_GENUS = os.path.join(BASE, "data/agp_ftp_processed/genus_names.npy")

OUTPUT_DIR = os.path.join(BASE, "saved_models/procyon_subtype_v2_7b")
EVAL_DIR = os.path.join(BASE, "eval_results_procyon_subtype_v2_7b")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

MODEL_PATH = "/hd/gcr/hf_models/Qwen2.5-7B-Instruct"
NL_CHECKPOINT = os.path.join(BASE, "saved_models/procyon_nl_7b")

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
EPOCHS = 5  # smaller dataset → more epochs
LR = 1e-4
MAX_LENGTH = 1024

ALL_LABELS = ["CD", "UC"]


def extract_label(text: str) -> str:
    m = re.search(r'(?:诊断结果|分型|亚型)[：:]\s*([A-Za-z]+)', text)
    if m:
        lab = m.group(1).strip().upper()
        if lab in ALL_LABELS:
            return lab
    # Fallback: first occurrence
    cd_pos = text.find("CD")
    uc_pos = text.find("UC")
    if cd_pos == -1 and uc_pos == -1:
        return None
    if cd_pos == -1:
        return "UC"
    if uc_pos == -1:
        return "CD"
    return "CD" if cd_pos < uc_pos else "UC"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


# ─── Prompt templates ────────────────────────────────────────────────
SUBTYPE_PROMPT = (
    "你是一位专业的肠道微生物分析师。该样本已知患有炎症性肠病（IBD），"
    "需要进一步判定其亚型。\n\n"
    "【主要菌属构成】: {genus_str}\n\n"
    "请判断该样本属于克罗恩病（CD）还是溃疡性结肠炎（UC），"
    "格式为：\"诊断结果：CD\" 或 \"诊断结果：UC\"，并简要说明理由。"
)

def extract_genus_str(item):
    """Pull '【主要菌属构成】: ...' segment from existing user message."""
    user_msg = item["messages"][0]["content"]
    m = re.search(r'【主要菌属构成】[:：]\s*(.+?)(?:\n|【|$)', user_msg, re.DOTALL)
    if m:
        return m.group(1).strip()
    return user_msg[:300]


def deviation_summary(sample_vec, baseline, genus_names, top_k=5):
    """Top-k genera by |sample - baseline|. Returns list of formatted strings."""
    diff = sample_vec - baseline
    order = np.argsort(-np.abs(diff))[:top_k]
    parts = []
    for i in order:
        s, b = float(sample_vec[i]), float(baseline[i])
        d = s - b
        sign = "升高" if d > 0 else "降低"
        parts.append(
            f"{genus_names[i]} {s:.2f}%（基线 {b:.2f}%，{sign} {abs(d):.2f}个百分点）"
        )
    return parts


def build_subtype_answer(label_detail, sample_vec, baseline, genus_names, top_k=5):
    """Sample-specific answer derived from per-sample deviation from healthy baseline.
    Each sample gets its own top-K deviating genera with actual percentages —
    forcing the LM to truly attend to the encoded sample, not memorize a template."""
    devs = deviation_summary(sample_vec, baseline, genus_names, top_k=top_k)
    bullets = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(devs))
    if label_detail == "CD":
        tail = (
            "上述偏离模式与回肠/结肠克罗恩病(CD)相符："
            "CD 患者常见菌群多样性显著降低、Faecalibacterium 等抗炎菌属减少，"
            "并伴随菌群结构整体重塑。"
        )
    else:  # UC
        tail = (
            "上述偏离模式与远端结肠溃疡性结肠炎(UC)相符："
            "UC 患者常见肠道屏障相关菌属减少、Bacteroides 类与促炎相关菌属相对升高，"
            "提示远端结肠菌群失衡。"
        )
    return (
        f"诊断结果：{label_detail}。\n\n"
        f"关键菌属偏离（相对健康基线）：\n{bullets}\n\n"
        f"{tail}"
    )


def build_subtype_records(src_jsonl, src_seq, src_msk, src_vec, baseline, genus_names):
    """Return (records, seqs, masks) filtered to CD/UC samples."""
    items = load_jsonl(src_jsonl)
    seqs = np.load(src_seq).astype(np.int64)
    msks = np.load(src_msk)
    vecs = np.load(src_vec)
    out_records, out_seqs, out_masks = [], [], []
    for i, item in enumerate(items):
        det = item.get("label_detail", "")
        if det not in ("CD", "UC"):
            continue
        genus_str = extract_genus_str(item)
        rec = {
            "task_type": "subtype_diagnosis",
            "sample_id": item.get("sample_id"),
            "label": det,
            "messages": [
                {"role": "user", "content": SUBTYPE_PROMPT.format(genus_str=genus_str)},
                {"role": "assistant", "content": build_subtype_answer(det, vecs[i], baseline, genus_names)},
            ],
        }
        out_records.append(rec)
        out_seqs.append(seqs[i])
        out_masks.append(msks[i])
    return out_records, np.stack(out_seqs), np.stack(out_masks)


def balance_subtype(records, seqs, masks, seed=42):
    """Oversample minority class (UC < CD)."""
    cd_idx = [i for i, r in enumerate(records) if r["label"] == "CD"]
    uc_idx = [i for i, r in enumerate(records) if r["label"] == "UC"]
    print(f"  Subtype distribution: CD={len(cd_idx)}, UC={len(uc_idx)}")
    if len(cd_idx) == len(uc_idx) or min(len(cd_idx), len(uc_idx)) == 0:
        return records, seqs, masks
    if len(cd_idx) > len(uc_idx):
        minority, majority = uc_idx, cd_idx
    else:
        minority, majority = cd_idx, uc_idx
    rng = random.Random(seed)
    n_extra = len(majority) - len(minority)
    dup = [minority[rng.randrange(len(minority))] for _ in range(n_extra)]
    new_records = list(records) + [records[i] for i in dup]
    new_seqs = np.concatenate([seqs] + [seqs[dup]], axis=0)
    new_masks = np.concatenate([masks] + [masks[dup]], axis=0)
    print(f"  Balanced: CD={sum(1 for r in new_records if r['label']=='CD')}, "
          f"UC={sum(1 for r in new_records if r['label']=='UC')}")
    return new_records, new_seqs, new_masks


# ═════════════════════════════════════════════════════════════════════
#  Model components (same as NL training)
# ═════════════════════════════════════════════════════════════════════

class ProjectionLayer(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, llm_hidden=LLM_HIDDEN):
        super().__init__()
        self.proj = nn.Linear(embed_dim, llm_hidden)

    def forward(self, x):
        return self.proj(x)


class MultimodalSubtypeModel(nn.Module):
    def __init__(self, llm_peft, encoder, projection):
        super().__init__()
        self.llm = llm_peft
        self.encoder = encoder
        self.projection = projection
        self.config = llm_peft.config

    def gradient_checkpointing_enable(self, **kw):
        self.llm.gradient_checkpointing_enable(**kw)

    def enable_input_require_grads(self):
        self.llm.enable_input_require_grads()

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                genus_ids=None, genus_mask=None, **kw):
        device = next(self.encoder.parameters()).device
        if input_ids is not None: input_ids = input_ids.to(device)
        if attention_mask is not None: attention_mask = attention_mask.to(device)
        if labels is not None: labels = labels.to(device)
        if genus_ids is not None: genus_ids = genus_ids.to(device)
        if genus_mask is not None: genus_mask = genus_mask.to(device)

        micro_embeds = self.encoder(genus_ids, genus_mask)
        micro_embeds = micro_embeds.to(self.projection.proj.weight.dtype)
        micro_tokens = self.projection(micro_embeds).unsqueeze(1)

        text_embeds = self.llm.base_model.model.model.embed_tokens(input_ids)
        micro_tokens = micro_tokens.to(text_embeds.dtype)
        combined = torch.cat([micro_tokens, text_embeds], dim=1)

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

        seq_len = combined.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=combined.device).unsqueeze(0)
        return self.llm(
            inputs_embeds=combined, attention_mask=new_mask,
            position_ids=position_ids, labels=new_labels, **kw,
        )


class SubtypeDataset(Dataset):
    def __init__(self, data, sequences, masks, tokenizer, max_length=1024):
        self.data = data
        self.sequences = sequences
        self.masks = masks
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded = []
        for item in data:
            full_ids = tokenizer.apply_chat_template(
                item["messages"], tokenize=True,
                max_length=max_length, truncation=True,
            )
            prompt_ids = tokenizer.apply_chat_template(
                [item["messages"][0]], tokenize=True,
                max_length=max_length, truncation=True,
                add_generation_prompt=True,
            )
            user_len = len(prompt_ids)
            labels = [-100] * len(full_ids)
            for i in range(user_len, len(full_ids)):
                labels[i] = full_ids[i]
            self.encoded.append({"input_ids": full_ids, "labels": labels})

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


class SubtypeCollator:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad = tokenizer.pad_token_id or 0

    def __call__(self, batch):
        ids = [b["input_ids"] for b in batch]
        mask = [b["attention_mask"] for b in batch]
        lbls = [b["labels"] for b in batch]
        gids = [b["genus_ids"] for b in batch]
        gmsk = [b["genus_mask"] for b in batch]

        max_len = min(max(len(i) for i in ids), self.max_length)
        pads, masks, lbs = [], [], []
        for i in range(len(ids)):
            x, m, l = ids[i], mask[i], lbls[i]
            pad_len = max_len - len(x)
            if pad_len > 0:
                pads.append(x + [self.pad] * pad_len)
                masks.append(m + [0] * pad_len)
                lbs.append(l + [-100] * pad_len)
            else:
                pads.append(x[:max_len])
                masks.append(m[:max_len])
                lbs.append(l[:max_len])

        gids = [g[:MAX_SEQ_LEN] for g in gids]
        gmsk = [g[:MAX_SEQ_LEN] for g in gmsk]
        max_g = max(len(g) for g in gids)
        pgids, pgmsk = [], []
        for i in range(len(gids)):
            pl = max_g - len(gids[i])
            if pl > 0:
                pgids.append(np.concatenate([gids[i], np.zeros(pl, dtype=np.int64)]))
                pgmsk.append(np.concatenate([gmsk[i], np.zeros(pl, dtype=gmsk[i].dtype)]))
            else:
                pgids.append(gids[i])
                pgmsk.append(gmsk[i])

        return {
            "input_ids": torch.tensor(pads, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(lbs, dtype=torch.long),
            "genus_ids": torch.tensor(np.stack(pgids), dtype=torch.long),
            "genus_mask": torch.tensor(np.stack(pgmsk), dtype=torch.bool),
        }


# ═════════════════════════════════════════════════════════════════════
#  Eval
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_subtype(model, tokenizer, records, seqs, masks, device,
                     max_new_tokens=128):
    model.eval()
    preds = []
    for idx, item in enumerate(records):
        seq = seqs[idx][:MAX_SEQ_LEN]
        msk = masks[idx][:MAX_SEQ_LEN]
        gid = torch.from_numpy(np.asarray(seq).astype(np.int64)).long().unsqueeze(0).to(device)
        gmk = torch.from_numpy(np.asarray(msk)).bool().unsqueeze(0).to(device)

        micro = model.encoder(gid, gmk)
        micro = micro.to(model.projection.proj.weight.dtype)
        mt = model.projection(micro).unsqueeze(1)

        prompt = tokenizer.apply_chat_template(
            [item["messages"][0]], tokenize=False, add_generation_prompt=True,
        )
        pin = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=MAX_LENGTH).to(device)
        te = model.llm.base_model.model.model.embed_tokens(pin["input_ids"])
        mt = mt.to(te.dtype)
        comb = torch.cat([mt, te], dim=1)
        L = comb.shape[1]
        pos = torch.arange(0, L, dtype=torch.long, device=device).unsqueeze(0)
        out = model.llm(inputs_embeds=comb, position_ids=pos, use_cache=True)
        nt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        gen_ids = [nt]
        cur = L
        for _ in range(max_new_tokens):
            pid = torch.full((1, 1), cur, dtype=torch.long, device=device)
            o = model.llm(input_ids=nt, position_ids=pid,
                          past_key_values=out.past_key_values, use_cache=True)
            nt = torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)
            if nt.item() == tokenizer.eos_token_id:
                break
            gen_ids.append(nt)
            cur += 1
            out.past_key_values = o.past_key_values
        gen_ids = torch.cat(gen_ids, dim=1)
        gen = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        pred = extract_label(gen)
        preds.append({
            "idx": idx,
            "sample_id": item["sample_id"],
            "true_label": item["label"],
            "predicted_label": pred,
            "generated": gen.strip()[:300],
        })
    return preds


# ═════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  ProCyon CD/UC Subtype Training (Stage-2)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Build subtype data ──
    print("\n[1/5] Building subtype dataset (filtering CD/UC)...")
    # Compute Healthy baseline from training set for deviation-based answers
    train_items_full = load_jsonl(SRC_TRAIN)
    train_vec_full = np.load(SRC_TRAIN_VEC)
    genus_names = np.load(SRC_GENUS, allow_pickle=True)
    if len(genus_names) > train_vec_full.shape[1]:
        genus_names = genus_names[:train_vec_full.shape[1]]
    h_idx = [i for i, d in enumerate(train_items_full) if d.get("label") == "Healthy"]
    baseline = np.median(train_vec_full[h_idx], axis=0)
    print(f"  Healthy baseline (median) from {len(h_idx)} samples (genera: {len(genus_names)})")

    train_records, train_seqs, train_masks = build_subtype_records(
        SRC_TRAIN, SRC_TRAIN_SEQ, SRC_TRAIN_MSK, SRC_TRAIN_VEC, baseline, genus_names
    )
    test_records, test_seqs, test_masks = build_subtype_records(
        SRC_TEST, SRC_TEST_SEQ, SRC_TEST_MSK, SRC_TEST_VEC, baseline, genus_names
    )
    print(f"  Train: {len(train_records)} records (seqs {train_seqs.shape})")
    print(f"  Test:  {len(test_records)} records")
    bal_records, bal_seqs, bal_masks = balance_subtype(
        train_records, train_seqs, train_masks
    )

    # ── 2. Tokenizer + LLM ──
    print("\n[2/5] Loading Qwen2.5-7B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", attn_implementation="eager",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    base.config.use_cache = False

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    lora = get_peft_model(base, lora_cfg)
    lora.gradient_checkpointing_enable()

    # ── 3. Build encoder + projection (transfer from NL) ──
    encoder = MGMEncoder(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
        n_layers=MGM_LAYERS, n_heads=MGM_HEADS, ffn_dim=MGM_FFN_DIM,
        max_seq_len=MAX_SEQ_LEN, dropout=MGM_DROPOUT,
    )
    projection = ProjectionLayer()

    nl_ckpt_path = os.path.join(NL_CHECKPOINT, "multimodal_components.pt")
    if os.path.exists(nl_ckpt_path):
        ck = torch.load(nl_ckpt_path, map_location=device)
        encoder.load_state_dict(ck["encoder_state_dict"])
        projection.load_state_dict(ck["projection_state_dict"])
        print(f"  ✓ Transferred encoder + projection from {nl_ckpt_path}")

    model = MultimodalSubtypeModel(lora, encoder, projection)
    model.is_parallelizable = True
    model.model_parallel = True
    if hasattr(lora, "hf_device_map"):
        model.hf_device_map = lora.hf_device_map
    encoder.to("cuda:0")
    projection.to("cuda:0")

    # ── 4. Train ──
    print("\n[3/5] Training (oversampled CD/UC balanced)...")
    train_dataset = SubtypeDataset(
        bal_records, bal_seqs, bal_masks, tokenizer, max_length=MAX_LENGTH,
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
    collator = SubtypeCollator(tokenizer, max_length=MAX_LENGTH)
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, data_collator=collator,
    )
    print("\nStarting subtype training...")
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

    # ── 5. Evaluate ──
    print("\n[4/5] Evaluating on test set (CD/UC only)...")
    del trainer
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
    eval_proj = ProjectionLayer()
    ck = torch.load(os.path.join(OUTPUT_DIR, "multimodal_components.pt"),
                    map_location=device)
    eval_encoder.load_state_dict(ck["encoder_state_dict"])
    eval_proj.load_state_dict(ck["projection_state_dict"])
    eval_encoder.to(eval_lora.device, dtype=torch.bfloat16)
    eval_proj.to(eval_lora.device, dtype=torch.bfloat16)
    eval_model = MultimodalSubtypeModel(eval_lora, eval_encoder, eval_proj)

    preds = evaluate_subtype(
        eval_model, tokenizer, test_records, test_seqs, test_masks, device,
    )

    from sklearn.metrics import accuracy_score, f1_score, classification_report
    valid = [p for p in preds if p["predicted_label"] is not None]
    if valid:
        true = [p["true_label"] for p in valid]
        pr = [p["predicted_label"] for p in valid]
        acc = accuracy_score(true, pr)
        macro_f1 = f1_score(true, pr, labels=ALL_LABELS, average="macro", zero_division=0)
        print("\n" + "=" * 60)
        print(f"  SUBTYPE EVALUATION RESULTS")
        print("=" * 60)
        print(f"  Total: {len(preds)}, Parseable: {len(valid)}")
        print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Macro F1: {macro_f1:.4f}")
        print("\n  Classification report:")
        print(classification_report(true, pr, labels=ALL_LABELS, zero_division=0))
    else:
        acc = macro_f1 = None
        print("  No parseable predictions!")

    with open(os.path.join(EVAL_DIR, "predictions.json"), "w") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
    with open(os.path.join(EVAL_DIR, "results.json"), "w") as f:
        json.dump({
            "model": "ProCyon_Subtype_7B",
            "n_test": len(preds),
            "n_parseable": len(valid),
            "accuracy": float(acc) if acc is not None else None,
            "macro_f1": float(macro_f1) if macro_f1 is not None else None,
            "training_loss": float(final_loss),
        }, f, indent=2)

    print(f"\n[5/5] ✅ Done! Results in {EVAL_DIR}/")


if __name__ == "__main__":
    main()
