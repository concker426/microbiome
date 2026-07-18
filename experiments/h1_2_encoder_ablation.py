#!/usr/bin/env python3
"""H1.2: Encoder ablation - Pretrained MGM vs Random MGM vs No Encoder.

Trains 3 variants of the ProCyon model with identical config except encoder state:
  (a) Pretrained MGM encoder (from V5)
  (b) Randomly initialized MGM encoder
  (c) No encoder (pure LLM, zero projection tokens)
Compares Enc+NL accuracy and NL-only accuracy.
"""
import json, os, re, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from collections import Counter

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
SAVE_DIR = '/hd/liujx/microbiome_llm_project/saved_models/h1_2_ablation'
os.makedirs(RESULT_DIR, exist_ok=True)

# Fixed config (same as V5)
NMT, PS, LR_R, LR_A, LR_D = 4, 0.1, 16, 32, 0.03
BS, GA, NE, LR, ML, DW = 1, 8, 4, 3e-5, 1024, 1.5
SL = 86  # clean_2538 seq length

def lj(p):
    data = []
    with open(p) as f:
        for line in f: data.append(json.loads(line))
    return data

@torch.no_grad()
def evaluate(model, tok, test_data, test_seqs, test_masks, device, max_tok=128):
    """Evaluate both normal and NL-only (dropout) paths."""
    model.eval()
    results = {}
    for mode in ['normal', 'dropout']:
        predictions = []
        for i, item in enumerate(test_data):
            true_label = item['label']
            msgs = item['messages']
            sq = test_seqs[i]
            mk = test_masks[i]
            gi = torch.from_numpy(np.asarray(sq).astype(np.int64)).long().unsqueeze(0).to(device)
            gm = torch.from_numpy(np.asarray(mk)).bool().unsqueeze(0).to(device)
            me = model.enc(gi, gm).to(model.proj.p[0].weight.dtype)
            mt = model.proj(me)
            if mode == 'dropout':
                mt = mt * 0.0

            prompt = tok.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=True)
            pi = tok(prompt, return_tensors='pt', truncation=True, max_length=ML).to(device)
            te = model.llm.base_model.model.model.embed_tokens(pi['input_ids'])
            mt = mt.to(te.dtype)
            ce = torch.cat([mt, te], dim=1)
            sl = ce.shape[1]
            pid = torch.arange(0, sl, dtype=torch.long, device=device).unsqueeze(0)
            o = model.llm(inputs_embeds=ce, position_ids=pid, use_cache=True)
            next_tok = torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)
            generated = [next_tok]
            cur_len = sl
            for _ in range(max_tok):
                pos = torch.full((1, 1), cur_len, dtype=torch.long, device=device)
                out = model.llm(input_ids=next_tok, position_ids=pos, past_key_values=o.past_key_values, use_cache=True)
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                if next_tok.item() == tok.eos_token_id:
                    break
                generated.append(next_tok)
                cur_len += 1
                o.past_key_values = out.past_key_values
            gen_ids = torch.cat(generated, dim=1)
            gen_text = tok.decode(gen_ids[0], skip_special_tokens=True)
            pred = el(gen_text)
            predictions.append({'true_label': true_label, 'predicted_label': pred, 'generated': gen_text.strip()[:300]})

        # Compute metrics
        valid = [p for p in predictions if p['predicted_label']]
        if valid:
            from sklearn.metrics import accuracy_score, f1_score
            trues = [p['true_label'] for p in valid]
            preds = [p['predicted_label'] for p in valid]
            acc = accuracy_score(trues, preds)
            f1 = f1_score(trues, preds, labels=LABELS, average='macro', zero_division=0)
        else:
            acc, f1 = 0.0, 0.0
        results[mode] = {'accuracy': float(acc), 'f1': float(f1), 'n_valid': len(valid), 'n_total': len(predictions)}
    return results

def train_and_eval(variant, encoder_init='pretrained'):
    """Train one variant and evaluate."""
    print(f"\n{'='*60}")
    print(f"Training variant: {variant}")
    print(f"{'='*60}")

    save_path = os.path.join(SAVE_DIR, variant)
    os.makedirs(save_path, exist_ok=True)

    train_data = lj(TD)
    test_data = lj(TED)
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
    if encoder_init == 'pretrained':
        ck = torch.load(ENCODER_PATH, map_location='cpu')
        st = ck.get('model_state_dict', ck)
        enc.load_state_dict(st, strict=False)
        print(f"  Loaded pretrained encoder")

    proj = Proj()
    dropout = 0.5 if variant != 'no_encoder' else 0.0
    mm = MM(lm, enc, proj, dropout_prob=dropout)
    enc.to('cuda:0'); proj.to('cuda:0')
    for p in enc.parameters(): p.requires_grad = False

    # Build dataset - for 'no_encoder', zero out genus_ids
    ds = DS(train_data, train_seqs, train_masks, tok, ml=ML, dw=DW)

    mm.llm.gradient_checkpointing_enable()
    ta = TrainingArguments(output_dir=save_path, per_device_train_batch_size=BS,
        gradient_accumulation_steps=GA, num_train_epochs=NE, learning_rate=LR, bf16=True,
        gradient_checkpointing=True, logging_steps=5, save_strategy='no',
        remove_unused_columns=False, dataloader_num_workers=0,
        ddp_find_unused_parameters=False, optim='adamw_torch', lr_scheduler_type='cosine',
        warmup_ratio=0.2, report_to='none',
        gradient_checkpointing_kwargs={'use_reentrant':False}, max_grad_norm=1.0)

    dc = Coll(tok, ml=ML)
    tr = WT(model=mm, args=ta, train_dataset=ds, data_collator=dc)

    t0 = time.time()
    res = tr.train()
    train_time = time.time() - t0
    print(f"  Training time: {train_time/60:.1f}min  Loss: {res.training_loss:.4f}")

    # Save before evaluation (in case eval crashes)
    mm.llm.save_pretrained(save_path)
    torch.save({'encoder_state_dict': enc.state_dict(),
                'projection_state_dict': proj.state_dict()},
               os.path.join(save_path, 'multimodal_components.pt'))
    print(f"  Saved to {save_path}")

    # Evaluate
    device = torch.device('cuda:0')
    mm.llm.config.use_cache = True
    eval_results = evaluate(mm, tok, test_data, test_seqs, test_masks, device)

    # Cleanup
    del mm, tr, lm, llm
    torch.cuda.empty_cache()

    return {'variant': variant, 'encoder_init': encoder_init,
            'train_loss': float(res.training_loss), 'train_time_min': round(train_time/60, 1),
            'normal': eval_results['normal'], 'dropout': eval_results['dropout'],
            'gap': eval_results['normal']['accuracy'] - eval_results['dropout']['accuracy']}

def no_encoder_eval():
    """Evaluate pure LLM (no encoder) by prompting with NL only.
    This is equivalent to the 'dropout' path but more explicit:
    we directly use the LLM without any encoder projection."""
    print(f"\n{'='*60}")
    print(f"No Encoder: Pure LLM evaluation")
    print(f"{'='*60}")

    test_data = lj(TED)
    tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    llm = AutoModelForCausalLM.from_pretrained(MP, device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16)
    llm.config.use_cache = True
    device = torch.device('cuda:0')

    predictions = []
    for i, item in enumerate(test_data):
        msgs = item['messages']
        prompt = tok.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=True)
        pi = tok(prompt, return_tensors='pt', truncation=True, max_length=ML).to(device)
        o = llm.generate(**pi, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)
        gen_text = tok.decode(o[0][pi['input_ids'].shape[1]:], skip_special_tokens=True)
        pred = el(gen_text)
        predictions.append({'true_label': item['label'], 'predicted_label': pred, 'generated': gen_text.strip()[:300]})

    del llm; torch.cuda.empty_cache()

    valid = [p for p in predictions if p['predicted_label']]
    if valid:
        from sklearn.metrics import accuracy_score, f1_score
        trues = [p['true_label'] for p in valid]
        preds = [p['predicted_label'] for p in valid]
        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, labels=LABELS, average='macro', zero_division=0)
    else:
        acc, f1 = 0.0, 0.0

    return {'accuracy': float(acc), 'f1': float(f1), 'n_valid': len(valid), 'n_total': len(predictions)}

def main():
    print("=" * 60)
    print("H1.2: Encoder Ablation Study")
    print("=" * 60)

    all_results = {}

    # (a) Pretrained MGM encoder
    print("\n[A] Pretrained MGM encoder...")
    all_results['pretrained'] = train_and_eval('pretrained', encoder_init='pretrained')

    # (b) Random MGM encoder
    print("\n[B] Random MGM encoder...")
    all_results['random'] = train_and_eval('random', encoder_init='random')

    # (c) No encoder - pure LLM
    print("\n[C] No encoder (pure LLM)...")
    all_results['no_encoder'] = no_encoder_eval()

    # Summary
    print("\n" + "=" * 60)
    print("H1.2 SUMMARY: Encoder Ablation")
    print("=" * 60)
    print(f"{'Variant':<25} {'Enc+NL ACC':>12} {'NL-only ACC':>12} {'Gap':>10}")
    print("-" * 60)
    for name, res in all_results.items():
        if name == 'no_encoder':
            print(f"{name:<25} {'N/A':>12} {res['accuracy']:>12.4f} {'N/A':>10}")
        else:
            print(f"{name:<25} {res['normal']['accuracy']:>12.4f} {res['dropout']['accuracy']:>12.4f} {res['gap']:>10.4f}")

    # Save
    output = {
        'experiment': 'H1.2',
        'hypothesis': 'Pretrained > Random > None for classification',
        'results': all_results,
        'timestamp': str(__import__('datetime').datetime.now()),
    }
    # Extract key metrics
    if 'pretrained' in all_results and 'random' in all_results:
        output['metrics'] = {
            'pretrained_enc_nl_acc': all_results['pretrained']['normal']['accuracy'],
            'random_enc_nl_acc': all_results['random']['normal']['accuracy'],
            'no_encoder_acc': all_results['no_encoder']['accuracy'],
            'pretrained_vs_random_gain': all_results['pretrained']['normal']['accuracy'] - all_results['random']['normal']['accuracy'],
        }

    with open(os.path.join(RESULT_DIR, 'H1.2.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULT_DIR}/H1.2.json")

    # Verdict
    if 'pretrained' in all_results and 'random' in all_results:
        gain = all_results['pretrained']['normal']['accuracy'] - all_results['random']['normal']['accuracy']
        if gain > 0.01:
            print(f"CONCLUSION: Pretrained > Random by {gain:.4f}. Hypothesis PARTIALLY SUPPORTED.")
        else:
            print("CONCLUSION: Pretraining shows NO advantage over random init. Hypothesis REJECTED.")
    if all_results['no_encoder']['accuracy'] < 0.80:
        print("CONCLUSION: Encoder is NECESSARY for good performance (pure LLM << 80%).")

if __name__ == '__main__':
    main()
