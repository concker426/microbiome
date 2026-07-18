#!/usr/bin/env python3
"""Stage 1: Train Qwen2.5-7B + LoRA on PURE NL TEXT only (no encoder, no microbiome).
This gives the LLM domain knowledge about IBD diagnosis from text alone,
before we cascade it with the V6 microbiome model.
"""
import json, os, sys, time, re, torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

MODEL_DIR = '/hd/gcr/hf_models/Qwen2.5-7B-Instruct'
DATA_DIR = '/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
RESULT_DIR = '/hd/liujx/microbiome_llm_project/experiments/results'
SAVE_DIR = '/hd/liujx/microbiome_llm_project/saved_models/stage1_text_only'
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

LR = 3e-5; NE = 4; BS = 1; GA = 8; ML = 1024; DW = 1.5
LR_R = 16; LR_A = 32; LR_D = 0.03
LABELS = ["Healthy", "Disease"]

def extract_label(text):
    m = re.search(r'诊断结果[：:]\s*(\S+)', text)
    if m:
        lb = m.group(1).strip()
        if lb in LABELS: return lb
    for k in LABELS:
        if k in text: return k
    for cn, en in [('健康', 'Healthy'), ('疾病', 'Disease')]:
        if cn in text: return en
    return None

class TextDS(Dataset):
    def __init__(self, data, tok, ml=ML, dw=DW):
        self.data = data; self.num = len(data)
        self.tok = tok; self.ml = ml; self.dw = dw
    def __len__(self): return self.num
    def __getitem__(self, i):
        item = self.data[i]; msgs = item['messages']
        full = self.tok.apply_chat_template(msgs, tokenize=True, max_length=self.ml, truncation=True, add_generation_prompt=False).input_ids
        prompt = self.tok.apply_chat_template([msgs[0]], tokenize=True, max_length=self.ml, truncation=True, add_generation_prompt=True).input_ids
        plen = len(prompt)
        labels = [-100] * len(full)
        for j in range(plen, len(full)):
            labels[j] = full[j]
        sw = self.dw if item.get('label', 'Healthy') == 'Disease' else 1.0
        return {'input_ids': full, 'labels': labels, 'sample_weight': sw, 'true_label': item.get('label', '')}

class Coll:
    def __init__(self, tok, ml=ML):
        self.tok = tok; self.ml = ml; self.pid = tok.pad_token_id or 0
    def __call__(self, batch):
        ml2 = min(max(len(x['input_ids']) for x in batch), self.ml)
        pi, am, lb, sw, tl = [], [], [], [], []
        for x in batch:
            ids = x['input_ids'][:ml2]; msk = [1]*len(ids); l = x['labels'][:ml2]
            pad = ml2 - len(ids)
            if pad > 0:
                pi.append(ids + [self.pid]*pad)
                am.append(msk + [0]*pad)
                lb.append(l + [-100]*pad)
            else:
                pi.append(ids); am.append(msk); lb.append(l)
            sw.append(x['sample_weight']); tl.append(x['true_label'])
        return {'input_ids': torch.tensor(pi, dtype=torch.long), 'attention_mask': torch.tensor(am, dtype=torch.long),
                'labels': torch.tensor(lb, dtype=torch.long), 'sample_weights': torch.tensor(sw, dtype=torch.float32),
                'true_labels': tl}  # list of strings, not tensor

def main():
    print("=" * 60)
    print("Stage 1: Pure Text LLM Training (No Microbiome)")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    data = []
    for split in ['train_nl.jsonl', 'test_nl.jsonl']:
        with open(os.path.join(DATA_DIR, split)) as f:
            for line in f: data.append(json.loads(line))
    np.random.seed(42); np.random.shuffle(data)
    split = int(len(data)*0.8)
    train_data, test_data = data[:split], data[split:]
    print(f"  train={len(train_data)} test={len(test_data)}")

    train_ds = TextDS(train_data, tok)
    test_ds = TextDS(test_data, tok)
    coll = Coll(tok)

    print("[2/4] Loading Qwen2.5-7B + LoRA...")
    llm = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, trust_remote_code=True)
    lc = LoraConfig(r=LR_R, lora_alpha=LR_A, target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'], lora_dropout=LR_D, bias='none', task_type=TaskType.CAUSAL_LM)
    lm = get_peft_model(llm, lc)
    lm.print_trainable_parameters()
    lm.cuda()

    print("[3/4] Training...")
    opt = torch.optim.AdamW(lm.parameters(), lr=LR)
    for ep in range(NE):
        dl = DataLoader(train_ds, batch_size=BS, shuffle=True, collate_fn=coll)
        lm.train(); total_loss = 0; n = 0; t0 = time.time()
        print(f'  [Epoch {ep+1}/{NE}] {len(dl)} batches')
        for si, batch in enumerate(dl):
            ids = batch['input_ids'].cuda(); am = batch['attention_mask'].cuda()
            lb = batch['labels'].cuda(); sw = batch['sample_weights'].cuda()
            out = lm(input_ids=ids, attention_mask=am, labels=lb)
            ce_loss = out.loss
            # Weighted loss
            logits = out.logits[:, :-1, :].contiguous()
            shift_labels = lb[:, 1:].contiguous()
            per_token = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), shift_labels.view(-1), reduction='none').view(len(ids), -1)
            valid_mask = (shift_labels != -100).float()
            per_sample = (per_token * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
            loss = (per_sample * sw).sum() / sw.sum()
            loss = loss / GA; loss.backward(); total_loss += loss.item()*GA; n += 1
            if (si+1) % GA == 0:
                torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            if (si+1) % 50 == 0:
                print(f'    step {si+1}/{len(dl)} loss={total_loss/max(n,1):.4f} elapsed={time.time()-t0:.0f}s')
        print(f'  Epoch {ep+1} done, loss={total_loss/max(n,1):.4f}')

    # Save BEFORE evaluation (OOM-safe)
    lm.save_pretrained(SAVE_DIR); tok.save_pretrained(SAVE_DIR)
    print(f"Saved to {SAVE_DIR}")

    # Free training memory before eval
    torch.cuda.empty_cache()

    # Evaluate
    t0 = time.time()
    print("[4/4] Evaluating...")
    dl = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=coll)
    lm.eval()
    correct = 0; total = 0; preds = []
    with torch.no_grad():
        for batch in dl:
            ids = batch['input_ids'].cuda(); am = batch['attention_mask'].cuda()
            true_label = batch['true_labels'][0]
            out = lm.generate(input_ids=ids, attention_mask=am, max_new_tokens=64, do_sample=False, pad_token_id=coll.pid)
            text_out = tok.decode(out[0], skip_special_tokens=True)
            pred_label = extract_label(text_out)
            if pred_label and true_label:
                total += 1
                correct += 1 if pred_label == true_label else 0
            preds.append({'pred': pred_label, 'true': true_label, 'text': text_out[:200]})

    acc = correct/max(total,1)
    print(f"\nResults: ACC={acc:.4f} ({correct}/{total}) time={time.time()-t0:.0f}s")

    result = {'experiment':'stage1_text_only','accuracy':acc,'n_correct':correct,'n_total':total,'n_train':len(train_data),'n_test':len(test_data),'epochs':NE,'lr':LR,'bs':BS,'ga':GA}
    with open(os.path.join(RESULT_DIR,'stage1_text_only.json'),'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved results to {RESULT_DIR}/stage1_text_only.json")

if __name__ == '__main__':
    main()
