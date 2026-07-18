#!/usr/bin/env python3
"""H4.2: Curriculum dropout vs fixed dropout.

Compares:
  - Fixed 0.5 dropout (best from H4.1)
  - Curriculum [0.3, 0.5, 0.7, 0.8] across 4 epochs
Both use identical architecture (V5: 4 tokens, PS=0.1, pretrained MGM).
"""
import json, os, re, sys, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, '/hd/liujx/microbiome_llm_project')
from run_v6_merged import MGMEnc, Proj, MM, DS, Coll, WT, el, LABELS

DATA_DIR = '/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
TD = os.path.join(DATA_DIR, 'train_nl.jsonl')
TED = os.path.join(DATA_DIR, 'test_nl.jsonl')
TS = os.path.join(DATA_DIR, 'train_genus_sequences.npy')
TM = os.path.join(DATA_DIR, 'train_genus_masks.npy')
XS = os.path.join(DATA_DIR, 'test_genus_sequences.npy')
XM = os.path.join(DATA_DIR, 'test_genus_masks.npy')
MP = '/hd/gcr/hf_models/Qwen2.5-7B-Instruct'
ENCODER_PATH = '/hd/liujx/microbiome_llm_project/saved_models/mgm_encoder_pretrained/mgm_encoder.pt'
RESULT_DIR = '/hd/liujx/microbiome_llm_project/experiments/results'
SAVE_DIR = '/hd/liujx/microbiome_llm_project/saved_models/h4_2_curriculum'
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

NMT, PS, LR_R, LR_A, LR_D = 4, 0.1, 16, 32, 0.03
BS, GA, NE, LR, ML, DW = 1, 8, 4, 3e-5, 1024, 1.5
SL = 86
CURRICULUM_SCHEDULE = [0.3, 0.5, 0.7, 0.8]

def lj(p):
    data = []
    with open(p) as f:
        for line in f: data.append(json.loads(line))
    return data

@torch.no_grad()
def evaluate(model, tok, test_data, test_seqs, test_masks, device, max_tok=128):
    model.eval()
    results = {}
    for mode in ['normal', 'dropout']:
        predictions = []
        for i, item in enumerate(test_data):
            sq = test_seqs[i]; mk = test_masks[i]
            gi = torch.from_numpy(np.asarray(sq).astype(np.int64)).long().unsqueeze(0).to(device)
            gm = torch.from_numpy(np.asarray(mk)).bool().unsqueeze(0).to(device)
            me = model.enc(gi, gm).to(model.proj.p[0].weight.dtype)
            mt = model.proj(me)
            if mode == 'dropout':
                mt = mt * 0.0
            prompt = tok.apply_chat_template([item['messages'][0]], tokenize=False, add_generation_prompt=True)
            pi = tok(prompt, return_tensors='pt', truncation=True, max_length=ML).to(device)
            te = model.llm.base_model.model.model.embed_tokens(pi['input_ids'])
            mt = mt.to(te.dtype)
            ce = torch.cat([mt, te], dim=1)
            sl = ce.shape[1]
            pid = torch.arange(0, sl, dtype=torch.long, device=device).unsqueeze(0)
            o = model.llm(inputs_embeds=ce, position_ids=pid, use_cache=True)
            next_tok = torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)
            generated = [next_tok]; cur_len = sl
            for _ in range(max_tok):
                pos = torch.full((1, 1), cur_len, dtype=torch.long, device=device)
                out = model.llm(input_ids=next_tok, position_ids=pos, past_key_values=o.past_key_values, use_cache=True)
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                if next_tok.item() == tok.eos_token_id: break
                generated.append(next_tok); cur_len += 1
                o.past_key_values = out.past_key_values
            gen_text = tok.decode(torch.cat(generated, dim=1)[0], skip_special_tokens=True)
            pred = el(gen_text)
            predictions.append({'true_label': item['label'], 'predicted_label': pred})
        from sklearn.metrics import accuracy_score, f1_score
        valid = [p for p in predictions if p['predicted_label']]
        trues = [p['true_label'] for p in valid]; preds = [p['predicted_label'] for p in valid]
        acc = accuracy_score(trues, preds) if valid else 0.0
        f1 = f1_score(trues, preds, labels=LABELS, average='macro', zero_division=0) if valid else 0.0
        results[mode] = {'accuracy': float(acc), 'f1': float(f1), 'n_valid': len(valid), 'n_total': len(predictions)}
    return results

def train_one(name, dropout_rate, curriculum=None):
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")
    save_path = os.path.join(SAVE_DIR, name)
    os.makedirs(save_path, exist_ok=True)

    train_data = lj(TD); test_data = lj(TED)
    train_seqs = np.load(TS); train_masks = np.load(TM)
    test_seqs = np.load(XS); test_masks = np.load(XM)

    tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = 'right'

    llm = AutoModelForCausalLM.from_pretrained(MP, device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16)
    llm.config.use_cache = False
    lora_config = LoraConfig(r=LR_R, lora_alpha=LR_A, target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'], lora_dropout=LR_D, bias='none', task_type=TaskType.CAUSAL_LM)
    lm = get_peft_model(llm, lora_config)

    enc = MGMEnc()
    ck = torch.load(ENCODER_PATH, map_location='cpu')
    st = ck.get('model_state_dict', ck)
    enc.load_state_dict(st, strict=False)
    proj = Proj()
    mm = MM(lm, enc, proj, dropout_prob=dropout_rate)
    enc.to('cuda:0'); proj.to('cuda:0')
    for p in enc.parameters(): p.requires_grad = False

    ds = DS(train_data, train_seqs, train_masks, tok, ml=ML, dw=DW)
    mm.llm.gradient_checkpointing_enable()
    ta = TrainingArguments(output_dir=save_path, per_device_train_batch_size=BS,
        gradient_accumulation_steps=GA, num_train_epochs=NE, learning_rate=LR, bf16=True,
        gradient_checkpointing=True, logging_steps=10, save_strategy='no',
        remove_unused_columns=False, dataloader_num_workers=0,
        ddp_find_unused_parameters=False, optim='adamw_torch', lr_scheduler_type='cosine',
        warmup_ratio=0.2, report_to='none',
        gradient_checkpointing_kwargs={'use_reentrant':False}, max_grad_norm=1.0)
    dc = Coll(tok, ml=ML)
    tr = WT(model=mm, args=ta, train_dataset=ds, data_collator=dc)

    # Curriculum callback
    if curriculum:
        class CurriculumCB(TrainerCallback):
            def __init__(self, model, schedule):
                self.model = model; self.schedule = schedule; self.ep = 0
            def on_epoch_begin(self, args, state, control, **kwargs):
                p = self.schedule[min(self.ep, len(self.schedule)-1)]
                self.model.set_dropout_prob(p)
                print(f'\n  [Epoch {self.ep+1}] curriculum dropout_prob={p}')
                self.ep += 1
        tr.add_callback(CurriculumCB(mm, curriculum))

    t0 = time.time(); res = tr.train(); train_time = time.time() - t0

    device = torch.device('cuda:0')
    mm.llm.config.use_cache = True
    eval_results = evaluate(mm, tok, test_data, test_seqs, test_masks, device)

    del mm, tr, lm, llm; torch.cuda.empty_cache()
    return {'name': name, 'dropout_rate': dropout_rate, 'curriculum': curriculum,
            'train_loss': float(res.training_loss), 'train_time_min': round(train_time/60, 1),
            'normal': eval_results['normal'], 'dropout': eval_results['dropout'],
            'gap': eval_results['normal']['accuracy'] - eval_results['dropout']['accuracy']}

def main():
    print("=" * 60)
    print("H4.2: Curriculum vs Fixed Dropout")
    print("=" * 60)

    all_results = []

    # Fixed 0.5 dropout
    print("\n--- Fixed 0.5 dropout ---")
    all_results.append(train_one('fixed_0.5', 0.5))

    # Curriculum [0.3, 0.5, 0.7, 0.8]
    print("\n--- Curriculum [0.3, 0.5, 0.7, 0.8] ---")
    all_results.append(train_one('curriculum', 0.5, curriculum=CURRICULUM_SCHEDULE))

    # Summary
    print("\n" + "=" * 70)
    print("H4.2 SUMMARY: Fixed vs Curriculum Dropout")
    print("=" * 70)
    print(f"{'Method':<25} {'Enc+NL ACC':>12} {'NL-only ACC':>12} {'Gap':>10}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['name']:<25} {r['normal']['accuracy']:>12.4f} "
              f"{r['dropout']['accuracy']:>12.4f} {r['gap']:>10.4f}")

    fixed = all_results[0]; curr = all_results[1]
    print(f"\nCurriculum advantage (NL-only): {curr['dropout']['accuracy'] - fixed['dropout']['accuracy']:+.4f}")
    print(f"Curriculum cost (Enc+NL): {curr['normal']['accuracy'] - fixed['normal']['accuracy']:+.4f}")

    output = {
        'experiment': 'H4.2',
        'hypothesis': 'Curriculum dropout > fixed dropout for NL-only',
        'results': all_results,
        'curriculum_advantage_nl': curr['dropout']['accuracy'] - fixed['dropout']['accuracy'],
        'timestamp': str(__import__('datetime').datetime.now()),
        'metrics': {
            'fixed_nl_only': fixed['dropout']['accuracy'],
            'curriculum_nl_only': curr['dropout']['accuracy'],
            'fixed_enc_nl': fixed['normal']['accuracy'],
            'curriculum_enc_nl': curr['normal']['accuracy'],
        }
    }

    with open(os.path.join(RESULT_DIR, 'H4.2.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {RESULT_DIR}/H4.2.json")

if __name__ == '__main__':
    main()
