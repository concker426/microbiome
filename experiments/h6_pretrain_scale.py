#!/usr/bin/env python3
"""H6: Pretraining Scale vs Downstream Performance.

Fine-tunes ProCyon V5 (no dropout) on clean_2538 using MGM encoders
pretrained at different scales: random, 10k, 50k, 250k.
Measures downstream IBD classification accuracy.
"""
import json, os, sys, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, '/hd/liujx/microbiome_llm_project')
from run_v6_merged import MGMEnc, Proj, MM, DS, Coll, WT, el, LABELS

DATA_DIR = '/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
TD = os.path.join(DATA_DIR, 'train_nl.jsonl'); TED = os.path.join(DATA_DIR, 'test_nl.jsonl')
TS = os.path.join(DATA_DIR, 'train_genus_sequences.npy'); TM = os.path.join(DATA_DIR, 'train_genus_masks.npy')
XS = os.path.join(DATA_DIR, 'test_genus_sequences.npy'); XM = os.path.join(DATA_DIR, 'test_genus_masks.npy')
MP = '/hd/gcr/hf_models/Qwen2.5-7B-Instruct'
RESULT_DIR = '/hd/liujx/microbiome_llm_project/experiments/results'
SAVE_BASE = '/hd/liujx/microbiome_llm_project/saved_models/h6_scale'
os.makedirs(RESULT_DIR, exist_ok=True); os.makedirs(SAVE_BASE, exist_ok=True)

NMT, PS, LR_R, LR_A, LR_D = 4, 0.1, 16, 32, 0.03
BS, GA, NE, LR, ML, DW = 1, 8, 4, 3e-5, 1024, 1.5
SL = 86

ENCODERS = {
    'random': None,  # Random init
    '10k': '/hd/liujx/microbiome_llm_project/saved_models/mgm_encoder_pretrained/mgm_encoder.pt',
    '50k': '/hd/liujx/microbiome_llm_project/saved_models/mgm_encoder_qiita_50k/mgm_encoder.pt',
    '250k': '/hd/liujx/microbiome_llm_project/saved_models/mgm_encoder_qiita_250k/mgm_encoder.pt',
}

def lj(p):
    data = []
    with open(p) as f:
        for line in f: data.append(json.loads(line))
    return data

@torch.no_grad()
def evaluate(model, tok, test_data, test_seqs, test_masks, device, max_tok=128):
    """Evaluate Enc+NL accuracy using proven autoregressive method."""
    model.eval()
    predictions = []
    for i, item in enumerate(test_data):
        sq = test_seqs[i]; mk = test_masks[i]
        gi = torch.from_numpy(np.asarray(sq).astype(np.int64)).long().unsqueeze(0).to(device)
        gm = torch.from_numpy(np.asarray(mk)).bool().unsqueeze(0).to(device)
        me = model.enc(gi, gm).to(model.proj.p[0].weight.dtype)
        mt = model.proj(me)
        prompt = tok.apply_chat_template([item['messages'][0]], tokenize=False, add_generation_prompt=True)
        pi = tok(prompt, return_tensors='pt', truncation=True, max_length=ML).to(device)
        te = model.llm.base_model.model.model.embed_tokens(pi['input_ids'])
        mt = mt.to(te.dtype)
        ce = torch.cat([mt, te], dim=1)
        sl2 = ce.shape[1]
        pid = torch.arange(0, sl2, dtype=torch.long, device=device).unsqueeze(0)
        o = model.llm(inputs_embeds=ce, position_ids=pid, use_cache=True)
        next_tok = torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)
        generated = [next_tok]; cur_len = sl2
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
    return {'accuracy': float(acc), 'f1': float(f1), 'n_valid': len(valid), 'n_total': len(predictions)}

def train_one(scale_name, encoder_path):
    print(f"\n{'='*60}")
    print(f"Training with encoder: {scale_name}")
    print(f"{'='*60}")
    save_path = os.path.join(SAVE_BASE, scale_name)
    os.makedirs(save_path, exist_ok=True)

    train_data = lj(TD); test_data = lj(TED)
    train_seqs = np.load(TS); train_masks = np.load(TM)
    test_seqs = np.load(XS); test_masks = np.load(XM)

    tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = 'right'

    llm = AutoModelForCausalLM.from_pretrained(MP, trust_remote_code=True, torch_dtype=torch.bfloat16).to('cuda:0')
    llm.config.use_cache = False
    lora_config = LoraConfig(r=LR_R, lora_alpha=LR_A, target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'], lora_dropout=LR_D, bias='none', task_type=TaskType.CAUSAL_LM)
    lm = get_peft_model(llm, lora_config)

    enc = MGMEnc()
    if encoder_path:
        ck = torch.load(encoder_path, map_location='cpu')
        st = ck.get('model_state_dict', ck)
        missing, unexpected = enc.load_state_dict(st, strict=False)
        print(f"  Loaded pretrained: {len([k for k in st if k in enc.state_dict()])} params matched")
    proj = Proj()
    mm = MM(lm, enc, proj, dropout_prob=0.0)  # No dropout for clean comparison
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

    t0 = time.time(); res = tr.train(); train_time = time.time() - t0
    print(f"  Loss: {res.training_loss:.4f}  Time: {train_time/60:.1f}min")

    device = torch.device('cuda:0')
    mm.llm.config.use_cache = True
    eval_r = evaluate(mm, tok, test_data, test_seqs, test_masks, device)

    del mm, tr, lm, llm; torch.cuda.empty_cache()
    return {'scale': scale_name, 'n_pretrain': int(scale_name.replace('k','000')) if scale_name != 'random' else 0,
            'train_loss': float(res.training_loss), 'train_time_min': round(train_time/60, 1),
            'accuracy': eval_r['accuracy'], 'f1': eval_r['f1'],
            'n_valid': eval_r['n_valid'], 'n_total': eval_r['n_total']}

def main():
    print("=" * 60)
    print("H6: Pretraining Scale vs Downstream Performance")
    print("=" * 60)
    print("Config: V5 baseline (no dropout), clean_2538, 4 epochs")
    print("Varying: MGM encoder pretraining data scale")

    results = []
    for scale_name, path in ENCODERS.items():
        r = train_one(scale_name, path)
        results.append(r)
        print(f"  {scale_name:>6s}: ACC={r['accuracy']:.4f} F1={r['f1']:.4f} ({r['n_valid']}/{r['n_total']} valid)")

    # Summary
    print("\n" + "=" * 70)
    print("H6 RESULTS: Pretraining Scale vs Downstream ACC")
    print("=" * 70)
    print(f"{'Scale':<12} {'N Pretrain':>10} {'ACC':>10} {'F1':>10} {'Train Loss':>12} {'Valid':>10}")
    print("-" * 66)
    for r in results:
        label = f"{r['n_pretrain']/1000:.0f}k" if r['n_pretrain'] > 0 else 'random'
        print(f"{label:<12} {r['n_pretrain']:>10,} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['train_loss']:>12.4f} {r['n_valid']:>6}/{r['n_total']:<4}")

    # Analysis
    nonzero = [r for r in results if r['n_pretrain'] > 0]
    if nonzero:
        best = max(nonzero, key=lambda x: x['accuracy'])
        random_r = [r for r in results if r['n_pretrain'] == 0][0]
        print(f"\nBest pretrained: {best['scale']} ACC={best['accuracy']:.4f}")
        print(f"Random init: ACC={random_r['accuracy']:.4f}")
        print(f"Pretraining gain: {best['accuracy'] - random_r['accuracy']:+.4f}")

    output = {
        'experiment': 'H6',
        'hypothesis': 'More pretraining data -> better downstream performance',
        'results': results,
        'timestamp': str(__import__('datetime').datetime.now()),
    }
    if nonzero and results:
        output['metrics'] = {
            'random_acc': random_r['accuracy'],
            'best_pretrained_acc': best['accuracy'],
            'best_pretrained_scale': best['scale'],
            'pretraining_gain': best['accuracy'] - random_r['accuracy'],
        }

    with open(os.path.join(RESULT_DIR, 'H6.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {RESULT_DIR}/H6.json")

    if nonzero and output['metrics']['pretraining_gain'] > 0.01:
        print("CONCLUSION: Pretraining improves downstream performance. Hypothesis SUPPORTED.")
    else:
        print("CONCLUSION: Pretraining shows no clear benefit at tested scales.")

if __name__ == '__main__':
    main()
