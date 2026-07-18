#!/usr/bin/env python3
"""Stage 2: Cascade Inference — V6 predicts first, then text LLM makes final decision.

Architecture:
  V6 (MGM encoder + Proj + Qwen2.5-7B LoRA) → prediction "Healthy"/"Disease"
      ↓
  Stage 1 (pure text LLM) ← original NL text + V6 prediction as extra context
      ↓
  Final diagnosis

Compares: V6-only | Text-only (Stage 1) | Cascade
"""
import json, os, sys, time, re
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, '/hd/liujx/microbiome_llm_project')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_FLASH_ATTN_ENABLED"] = "0"
import fix_flash_attn
import accelerate.utils.imports as _ai; _ai.is_deepspeed_available = lambda: False
import accelerate.utils.other as _ao; _ao.is_deepspeed_available = lambda: False
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from run_v6_merged import MGMEnc, Proj, MM, LABELS

# Paths
DATA_DIR = '/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
V6_DIR = '/hd/liujx/microbiome_llm_project/saved_models/v6_curriculum'
STAGE1_DIR = '/hd/liujx/microbiome_llm_project/saved_models/stage1_text_only'
ENCODER_PATH = '/hd/liujx/microbiome_llm_project/saved_models/mgm_encoder_pretrained/mgm_encoder.pt'
LLM_PATH = '/hd/gcr/hf_models/Qwen2.5-7B-Instruct'
RESULT_DIR = '/hd/liujx/microbiome_llm_project/experiments/results'
os.makedirs(RESULT_DIR, exist_ok=True)

V = 1226; E = 768; LH = 3584; SL = 86; NMT = 4; PS = 0.1
LR_R = 16; LR_A = 32; LR_D = 0.03; ML = 1024

def extract_label(text):
    import re
    m = re.search(r'诊断结果[：:]\s*(\S+)', text)
    if m:
        lb = m.group(1).strip()
        if lb in LABELS: return lb
    for k in LABELS:
        if k in text: return k
    for cn, en in [('健康', 'Healthy'), ('疾病', 'Disease')]:
        if cn in text: return en
    return None

def load_data():
    data = []
    for split in ['train_nl.jsonl', 'test_nl.jsonl']:
        with open(os.path.join(DATA_DIR, split)) as f:
            for line in f: data.append(json.loads(line))
    train_seqs = np.load(os.path.join(DATA_DIR, 'train_genus_sequences.npy'))
    test_seqs = np.load(os.path.join(DATA_DIR, 'test_genus_sequences.npy'))
    train_masks = np.load(os.path.join(DATA_DIR, 'train_genus_masks.npy'))
    test_masks = np.load(os.path.join(DATA_DIR, 'test_genus_masks.npy'))
    all_seqs = np.concatenate([train_seqs, test_seqs], axis=0)
    all_masks = np.concatenate([train_masks, test_masks], axis=0)
    all_labels = [d['label'] for d in data]
    return data, all_seqs, all_masks, all_labels

def load_v6_model(device='cuda:0'):
    """Load V6: MGM encoder + Projection + Qwen2.5-7B LoRA."""
    tok = AutoTokenizer.from_pretrained(V6_DIR, trust_remote_code=True)
    # Load encoder
    enc = MGMEnc()
    ck = torch.load(ENCODER_PATH, map_location='cpu')
    st = ck.get('model_state_dict', ck)
    enc.load_state_dict(st, strict=False)
    enc.to(device).eval()
    # Load projection
    proj = Proj()
    mm_ck = torch.load(os.path.join(V6_DIR, 'multimodal_components.pt'), map_location='cpu')
    proj.load_state_dict(mm_ck['proj_state_dict'])
    proj.to(device).eval()
    # Load LLM + LoRA
    llm = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.bfloat16, trust_remote_code=True)
    lm = PeftModel.from_pretrained(llm, V6_DIR)
    lm.to(device).eval()
    # Build MM model (no dropout for eval)
    mm = MM(lm, enc, proj, dropout_prob=0.0)
    return mm, tok

def load_stage1_model(device='cuda:0'):
    """Load Stage 1: pure text LLM + LoRA."""
    tok = AutoTokenizer.from_pretrained(STAGE1_DIR, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.bfloat16, trust_remote_code=True)
    lm = PeftModel.from_pretrained(llm, STAGE1_DIR)
    lm.to(device).eval()
    return lm, tok

def v6_predict_one(mm, tok, genus_ids, genus_mask, label=None):
    """Run V6 autoregressive inference. Returns predicted label string."""
    device = next(mm.parameters()).device
    gi = torch.from_numpy(genus_ids.astype(np.int64)).long().unsqueeze(0).to(device)
    gm = torch.from_numpy(genus_mask.astype(bool)).unsqueeze(0).to(device)

    # Extract projection tokens
    with torch.no_grad():
        me = mm.enc(gi, gm).to(mm.proj.p[0].weight.dtype)
        mt = mm.proj(me)  # [1, 4, 3584]

    # Build prompt text embedding for autoregressive generation
    # Use a simple generation prompt
    prompt = '<|im_start|>assistant\n'
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    import torch.nn.functional as F
    te = lm.base_model.model.model.embed_tokens(torch.tensor([prompt_ids], device=device))
    mt = mt.to(te.dtype)
    ce = torch.cat([mt, te], dim=1)  # [1, 4+prompt_len, 3584]

    # Autoregressive generation
    generated_ids = list(prompt_ids)
    past_kv = None
    for _ in range(64):
        if past_kv is None:
            pid = torch.arange(ce.shape[1], dtype=torch.long, device=device).unsqueeze(0)
            out = mm.llm(inputs_embeds=ce, position_ids=pid, use_cache=True)
        else:
            pid = torch.tensor([[ce.shape[1] + len(generated_ids) - len(prompt_ids) - 1]], dtype=torch.long, device=device)
            cur_emb = te[:, -1:, :]
            out = mm.llm(inputs_embeds=cur_emb, position_ids=pid, past_key_values=past_kv, use_cache=True)

        past_kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1).item()
        if next_token == tok.eos_token_id or next_token == tok.pad_token_id:
            break
        generated_ids.append(next_token)
        # Update te for next iteration
        te = lm.base_model.model.model.embed_tokens(torch.tensor([[next_token]], device=device))

    text = tok.decode(generated_ids, skip_special_tokens=True)
    pred = extract_label(text)
    return pred

def text_llm_predict_one(lm, tok, prompt_text, device='cuda:0'):
    """Run Stage 1 text-only LLM inference on a prompt. Returns predicted label."""
    ids = tok.apply_chat_template(prompt_text, tokenize=True, max_length=ML, truncation=True, add_generation_prompt=True).input_ids
    input_ids = torch.tensor([ids], dtype=torch.long).to(device)
    am = torch.ones_like(input_ids)
    with torch.no_grad():
        out = lm.generate(input_ids=input_ids, attention_mask=am, max_new_tokens=64,
                         do_sample=False, pad_token_id=tok.pad_token_id or 0)
    text = tok.decode(out[0], skip_special_tokens=True)
    return extract_label(text)

def build_cascade_prompt(sample, v6_pred):
    """Inject V6 prediction into the NL text as additional context for text LLM."""
    msgs = sample['messages']
    user_msg = msgs[0]['content']
    # Append V6 prediction as expert opinion
    augmented_user = f"{user_msg}\n\n【微生物组模型辅助判断】：该样本经菌群特征分析，初步判断为 {v6_pred}。请结合以上信息，给出你的最终诊断。"

    return [{'role': 'user', 'content': augmented_user}]

def main():
    print("=" * 60)
    print("Stage 2: Cascade Inference (V6 → Text LLM)")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    data, all_seqs, all_masks, all_labels = load_data()
    print(f"  {len(data)} samples, Disease={sum(1 for l in all_labels if l=='Disease')}")

    # Only evaluate on test set
    train_data = [d for d in data if d.get('dataset_type', '') == 'qiita_2538']
    # Use the full dataset (train+test split from data dir)
    # train_nl.jsonl = 659, test_nl.jsonl = 167
    # But load_data loads both, so use all
    test_indices = list(range(len(data)))
    # Actually use the proper test split
    train_n = 659
    test_indices = list(range(train_n, len(data)))
    print(f"  Using test indices: {train_n}..{len(data)-1} ({len(test_indices)} samples)")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # === V6 Inference ===
    print("\n[2/4] V6 Inference...")
    v6_mm, v6_tok = load_v6_model(device)
    v6_preds = []
    t0 = time.time()
    for idx in test_indices:
        gi = all_seqs[idx, :SL]
        gm = all_masks[idx, :SL]
        pred = v6_predict_one(v6_mm, v6_tok, gi, gm)
        v6_preds.append(pred)
    v6_time = time.time() - t0
    print(f"  V6 done in {v6_time:.0f}s")

    # Free V6 GPU memory
    del v6_mm
    torch.cuda.empty_cache()

    # === Stage 1 Text-only Inference ===
    print("\n[3/4] Stage 1 Text-only & Cascade Inference...")
    s1_lm, s1_tok = load_stage1_model(device)
    s1_preds = []
    cascade_preds = []
    t0 = time.time()
    for i, idx in enumerate(test_indices):
        sample = data[idx]
        # Text-only: use original prompt
        pred_text = text_llm_predict_one(s1_lm, s1_tok, sample['messages'], device)
        s1_preds.append(pred_text)

        # Cascade: inject V6 prediction into prompt
        v6_pred = v6_preds[i]
        cascade_msgs = build_cascade_prompt(sample, v6_pred)
        pred_cascade = text_llm_predict_one(s1_lm, s1_tok, cascade_msgs, device)
        cascade_preds.append(pred_cascade)

        if (i+1) % 20 == 0:
            print(f'  {i+1}/{len(test_indices)} done ({time.time()-t0:.0f}s)')

    cascade_time = time.time() - t0
    print(f"  Text-only + Cascade done in {cascade_time:.0f}s")

    # === Evaluate ===
    print("\n[4/4] Results...")
    true_labels = [all_labels[i] for i in test_indices]

    def calc_acc(preds, trues):
        valid = [(p, t) for p, t in zip(preds, trues) if p is not None]
        if not valid: return 0, 0, 0
        correct = sum(1 for p, t in valid if p == t)
        return correct / len(valid), correct, len(valid)

    v6_acc, v6_cor, v6_tot = calc_acc(v6_preds, true_labels)
    s1_acc, s1_cor, s1_tot = calc_acc(s1_preds, true_labels)
    cas_acc, cas_cor, cas_tot = calc_acc(cascade_preds, true_labels)

    print(f"\n{'='*70}")
    print(f"  CASCADE RESULTS (test set, {len(test_indices)} samples)")
    print(f"  {'-'*50}")
    print(f"  {'Method':<30} {'ACC':>10} {'Valid':>10}")
    print(f"  {'-'*50}")
    print(f"  {'V6-only (MGM+LLM)':<30} {v6_acc:>10.4f} {v6_tot:>10}")
    print(f"  {'Text-only (Stage 1 LLM)':<30} {s1_acc:>10.4f} {s1_tot:>10}")
    print(f"  {'Cascade (V6 → Text LLM)':<30} {cas_acc:>10.4f} {cas_tot:>10}")
    print(f"  {'='*70}")

    # Show a few examples
    print(f"\n  Example predictions:")
    for i in range(min(5, len(test_indices))):
        print(f"  [{i}] True={true_labels[i]} | V6={v6_preds[i]} | Text={s1_preds[i]} | Cascade={cascade_preds[i]}")

    result = {
        'experiment': 'stage2_cascade',
        'v6_only': {'accuracy': v6_acc, 'correct': v6_cor, 'total': v6_tot},
        'text_only': {'accuracy': s1_acc, 'correct': s1_cor, 'total': s1_tot},
        'cascade': {'accuracy': cas_acc, 'correct': cas_cor, 'total': cas_tot},
        'n_test': len(test_indices),
        'v6_time_s': v6_time,
        'cascade_time_s': cascade_time,
        'timestamp': str(__import__('datetime').datetime.now()),
    }
    with open(os.path.join(RESULT_DIR, 'stage2_cascade.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to {RESULT_DIR}/stage2_cascade.json")

if __name__ == '__main__':
    main()
