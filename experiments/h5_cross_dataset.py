#!/usr/bin/env python3
"""H5: Cross-dataset generalization using proven V6 evaluation method.

H5.1: V6b model (clean_2538) -> merged_all test
H5.2: V6c model (merged_all) -> clean_2538 test

Uses the exact same autoregressive eval as run_v6_merged.py (known good).
"""
import json, os, sys, time, re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, '/hd/liujx/microbiome_llm_project')
from run_v6_merged import MGMEnc, Proj, MM, el, LABELS

MP = '/hd/gcr/hf_models/Qwen2.5-7B-Instruct'
RESULT_DIR = '/hd/liujx/microbiome_llm_project/experiments/results'
os.makedirs(RESULT_DIR, exist_ok=True)

CLEAN_MODEL = '/hd/liujx/microbiome_llm_project/saved_models/v6_curriculum'
MERGED_MODEL = '/hd/liujx/microbiome_llm_project/saved_models/v6_merged'
CLEAN_DIR = '/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
MERGED_DIR = '/hd/liujx/microbiome_llm_project/data/qiita_ibd/merged_all'
ML = 1024
SL_CLEAN = 86
SL_MERGED = 175

def load_test_data(data_dir):
    data = []
    nl_file = os.path.join(data_dir, 'test_nl.jsonl')
    if not os.path.exists(nl_file): nl_file = os.path.join(data_dir, 'test.jsonl')
    with open(nl_file) as f:
        for line in f: data.append(json.loads(line))
    seqs = np.load(os.path.join(data_dir, 'test_genus_sequences.npy'))
    masks = np.load(os.path.join(data_dir, 'test_genus_masks.npy'))
    return data, seqs, masks

def load_model(model_dir, device):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MP, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    llm = PeftModel.from_pretrained(base, model_dir)
    llm.config.use_cache = True
    enc = MGMEnc(); proj = Proj()
    ck = torch.load(os.path.join(model_dir, 'multimodal_components.pt'), map_location=device)
    enc.load_state_dict(ck['encoder_state_dict']); proj.load_state_dict(ck['projection_state_dict'])
    enc.to(device, dtype=torch.bfloat16); proj.to(device, dtype=torch.bfloat16)
    for p in enc.parameters(): p.requires_grad = False
    return MM(llm, enc, proj, dropout_prob=0.0), tok

@torch.no_grad()
def eval_both(model, tok, test_data, test_seqs, test_masks, device, max_tok=128, sl=86):
    """Same proven eval as run_v6_merged.py."""
    model.eval()
    results = {}
    for mode in ['normal', 'dropout']:
        predictions = []
        for i, item in enumerate(test_data):
            sq = test_seqs[i]; mk = test_masks[i]
            if sq.ndim == 2: sq = sq[0]
            if mk.ndim == 2: mk = mk[0]
            sq = sq[:sl]; mk = mk[:sl]
            gi = torch.from_numpy(np.asarray(sq).astype(np.int64)).long().unsqueeze(0).to(device)
            gm = torch.from_numpy(np.asarray(mk)).bool().unsqueeze(0).to(device)
            me = model.enc(gi, gm).to(model.proj.p[0].weight.dtype)
            mt = model.proj(me)
            if mode == 'dropout': mt = mt * 0.0

            msgs = item['messages']
            prompt = tok.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=True)
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
        results[mode] = {'accuracy': float(acc), 'f1': float(f1), 'n_valid': len(valid), 'n_total': len(predictions)}
        print(f"  [{mode}] ACC={acc:.4f} F1={f1:.4f} ({len(valid)}/{len(predictions)} valid)")

    return results

def main():
    print("=" * 60)
    print("H5: Cross-Dataset Generalization (V6 proven eval)")
    print("=" * 60)
    device = torch.device('cuda:0')
    all_results = {}

    # H5.1: clean_2538 model -> merged_all test
    print("\n[H5.1] V6b (clean_2538, 86-seq) -> merged_all test (175-seq)")
    if os.path.exists(CLEAN_MODEL):
        t0 = time.time()
        model, tok = load_model(CLEAN_MODEL, device)
        td, ts, tm = load_test_data(MERGED_DIR)
        print(f"  Test: {len(td)} samples, seq={ts.shape}")
        r = eval_both(model, tok, td, ts, tm, device, sl=SL_MERGED)
        print(f"  Enc+NL={r['normal']['accuracy']:.4f}  NL-only={r['dropout']['accuracy']:.4f}  Gap={r['normal']['accuracy']-r['dropout']['accuracy']:.4f}")
        print(f"  Time: {(time.time()-t0)/60:.1f}min")
        all_results['H5.1_clean_to_merged'] = r
        del model; torch.cuda.empty_cache()
    else:
        print(f"  SKIP: {CLEAN_MODEL} not found")

    # H5.2: merged_all model -> clean_2538 test
    print("\n[H5.2] V6c (merged_all, 175-seq) -> clean_2538 test (86-seq)")
    if os.path.exists(MERGED_MODEL):
        t0 = time.time()
        model, tok = load_model(MERGED_MODEL, device)
        td, ts, tm = load_test_data(CLEAN_DIR)
        print(f"  Test: {len(td)} samples, seq={ts.shape}")
        r = eval_both(model, tok, td, ts, tm, device, sl=SL_CLEAN)
        print(f"  Enc+NL={r['normal']['accuracy']:.4f}  NL-only={r['dropout']['accuracy']:.4f}  Gap={r['normal']['accuracy']-r['dropout']['accuracy']:.4f}")
        print(f"  Time: {(time.time()-t0)/60:.1f}min")
        all_results['H5.2_merged_to_clean'] = r
        del model; torch.cuda.empty_cache()
    else:
        print(f"  SKIP: {MERGED_MODEL} not found")

    # Summary
    print("\n" + "=" * 70)
    print("H5 RESULTS: Cross-Dataset Generalization")
    print("=" * 70)
    print(f"{'Direction':<35} {'Enc+NL':>10} {'NL-only':>10} {'Gap':>10} {'Valid':>8}")
    print("-" * 70)
    print(f"{'clean->clean (V6b, in-domain)':<35} {'0.8623':>10} {'0.7605':>10} {'0.1018':>10} {'167/167':>8}")
    print(f"{'merged->merged (V6c, in-domain)':<35} {'0.8317':>10} {'0.8210':>10} {'0.0107':>10} {'838/838':>8}")
    for name, r in all_results.items():
        label = 'clean->merged (CROSS)' if 'clean_to_merged' in name else 'merged->clean (CROSS)'
        n = r['normal']; d = r['dropout']
        print(f"{label:<35} {n['accuracy']:>10.4f} {d['accuracy']:>10.4f} {n['accuracy']-d['accuracy']:>10.4f} {n['n_valid']:>4}/{n['n_total']:<4}")

    output = {
        'experiment': 'H5',
        'hypothesis': 'Model generalizes across datasets',
        'in_domain': {'clean': {'enc_nl': 0.8623, 'nl_only': 0.7605}, 'merged': {'enc_nl': 0.8317, 'nl_only': 0.8210}},
        'cross_dataset': all_results,
        'timestamp': str(__import__('datetime').datetime.now()),
    }
    if 'H5.1_clean_to_merged' in all_results:
        drop = 0.8623 - all_results['H5.1_clean_to_merged']['normal']['accuracy']
        output['metrics'] = {'clean_to_merged_drop': drop}
        print(f"\nH5.1 cross-dataset drop: {drop:.4f} ({drop*100:.1f}%)")
        if drop < 0.05: print("GOOD generalization (<5% drop)")
        elif drop < 0.10: print("MODERATE generalization (5-10% drop)")
        else: print("SIGNIFICANT domain shift (>10% drop)")

    with open(os.path.join(RESULT_DIR, 'H5.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved to {RESULT_DIR}/H5.json")

if __name__ == '__main__':
    main()
