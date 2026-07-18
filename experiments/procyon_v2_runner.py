#!/usr/bin/env python3
"""
ProCyon v2 Automated Experiment Runner
=======================================
Phase A1: Pooling ablation (Mean vs Attention vs CLS)
Phase A2: Transformer usefulness (with/without 6-layer Transformer)
Phase B1: SimpleEmb + Projection + Qwen LoRA full training

All on clean_2538. Results saved to experiments/results/procyon_v2_*.json
"""
import json, os, sys, time, copy, gc
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_FLASH_ATTN_ENABLED"] = "0"
import transformers.utils.import_utils as _iu; _iu.is_flash_attn_2_available = lambda: False
import transformers.utils as _utils; _utils.is_flash_attn_2_available = lambda: False
import accelerate.utils.imports as _ai; _ai.is_deepspeed_available = lambda: False
import accelerate.utils.other as _ao; _ao.is_deepspeed_available = lambda: False

sys.path.insert(0, '/hd/liujx/microbiome_llm_project')
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

DATA_DIR = '/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
RESULT_DIR = '/hd/liujx/microbiome_llm_project/experiments/results'
os.makedirs(RESULT_DIR, exist_ok=True)
V=1226; E=768; SL=86; DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════
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
    all_labels = np.array([1 if d['label'] == 'Disease' else 0 for d in data])
    return all_seqs, all_masks, all_labels

def evaluate_sklearn(X, y, n_folds=5, n_repeats=3):
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
    all_scores = []
    for seed in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42+seed)
        scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
        all_scores.extend(scores)
    return {'accuracy_mean': float(np.mean(all_scores)), 'accuracy_std': float(np.std(all_scores)), 'n_splits': len(all_scores)}

# ═══════════════════════════════════════════════════════════════════
# Encoder components
# ═══════════════════════════════════════════════════════════════════
class SimpleEmb(nn.Module):
    def __init__(self, vocab=V, dim=E):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
    def forward(self, ids, mask=None):
        return self.emb(ids)

def mean_pool(x, mask):
    mf = mask.float().unsqueeze(-1)
    return (x * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1)

def attn_pool(x, mask=None):
    lin = nn.Linear(E, 1, device=x.device)
    sc = torch.tanh(lin(x)).squeeze(-1)
    if mask is not None: sc = sc.masked_fill(~mask, float('-inf'))
    w = F.softmax(sc, dim=-1)
    return (x * w.unsqueeze(-1)).sum(dim=1)

def cls_pool(x, mask=None):
    return x[:, 0, :]

# ═══════════════════════════════════════════════════════════════════
# Phase A1: Pooling ablation (Mean vs Attention vs CLS)
# Uses SimpleEmb with DIFFERENT pooling → MLP
# ═══════════════════════════════════════════════════════════════════
def phase_a1(seqs, masks, labels):
    print("\n" + "="*60)
    print("Phase A1: Pooling Ablation (SimpleEmb + Pool → MLP)")
    print("="*60)
    results = {}

    emb = SimpleEmb().to(DEVICE).eval()
    # Extract base embeddings once
    print("  Extracting embeddings...")
    embs = []
    with torch.no_grad():
        for i in range(0, len(seqs), 128):
            gi = torch.from_numpy(seqs[i:i+128].astype(np.int64)).long().to(DEVICE)
            embs.append(emb(gi).cpu())
    embs = torch.cat(embs, dim=0)  # [N, 86, 768]
    masks_t = torch.from_numpy(masks.astype(bool))

    for pool_name, pool_fn in [('Mean', mean_pool), ('Attention', attn_pool), ('CLS', cls_pool)]:
        print(f"\n  [{pool_name} Pool]")
        t0 = time.time()
        feats = []
        with torch.no_grad():
            for i in range(0, len(embs), 128):
                x = embs[i:i+128]; m = masks_t[i:i+128]
                p = pool_fn(x.to(DEVICE), m.to(DEVICE) if pool_name != 'CLS' else None).cpu().numpy()
                feats.append(p)
        feats = np.concatenate(feats, axis=0)
        print(f"    Extraction: {time.time()-t0:.1f}s, shape={feats.shape}")
        r = evaluate_sklearn(feats, labels)
        results[pool_name] = r
        print(f"    {pool_name} Pool: ACC={r['accuracy_mean']:.4f} ±{r['accuracy_std']:.4f}")

    # Report
    print(f"\n  {'-'*50}")
    print(f"  A1 POOLING ABLATION RESULTS")
    print(f"  {'-'*50}")
    for name, r in results.items():
        print(f"  {name:>10}: {r['accuracy_mean']:.4f} ±{r['accuracy_std']:.4f}")
    best = max(results.items(), key=lambda x: x[1]['accuracy_mean'])
    print(f"  Best: {best[0]} ({best[1]['accuracy_mean']:.4f})")

    output = {'experiment':'procyon_v2_A1_pooling','results':results,'best':best[0],
              'best_acc':best[1]['accuracy_mean']}
    with open(os.path.join(RESULT_DIR,'procyon_v2_A1_pooling.json'),'w') as f:
        json.dump(output, f, indent=2)
    return results

# ═══════════════════════════════════════════════════════════════════
# Phase A2: Transformer usefulness
# SimpleEmb+Mean+MLP vs SimpleEmb+6L-Transformer+Mean+MLP
# ═══════════════════════════════════════════════════════════════════
def phase_a2(seqs, masks, labels):
    print("\n" + "="*60)
    print("Phase A2: Transformer Usefulness Ablation")
    print("="*60)

    # Reuse mean_pool features from A1 (SimpleEmb + Mean)
    print("\n  [A] SimpleEmb + Mean Pool (no Transformer) ...")
    emb = SimpleEmb().to(DEVICE).eval()
    masks_t = torch.from_numpy(masks.astype(bool))
    feats = []
    with torch.no_grad():
        for i in range(0, len(seqs), 128):
            gi = torch.from_numpy(seqs[i:i+128].astype(np.int64)).long().to(DEVICE)
            gm = masks_t[i:i+128]
            x = emb(gi); p = mean_pool(x, gm.to(DEVICE)).cpu().numpy()
            feats.append(p)
    feats_a = np.concatenate(feats, axis=0)
    r_a = evaluate_sklearn(feats_a, labels)
    print(f"  No Transformer: ACC={r_a['accuracy_mean']:.4f} ±{r_a['accuracy_std']:.4f}")

    # B: SimpleEmb + 6-layer Transformer + Mean Pool
    print("\n  [B] SimpleEmb + 6L Transformer + Mean Pool ...")
    from run_v6_merged import MGMEnc
    enc = MGMEnc()
    enc.to(DEVICE).eval()
    feats_b = []
    with torch.no_grad():
        for i in range(0, len(seqs), 64):
            gi = torch.from_numpy(seqs[i:i+64].astype(np.int64)).long().to(DEVICE)
            gm = torch.from_numpy(masks[i:i+64]).bool().to(DEVICE)
            # Get MGM encoder output (before attention pooling)
            x = enc.emb(gi)
            for blk in enc.blks: x = blk(x, kpm=None)
            # Apply MEAN pool instead of attention pool
            p = mean_pool(x, gm).cpu().numpy()
            feats_b.append(p)
    feats_b = np.concatenate(feats_b, axis=0)
    r_b = evaluate_sklearn(feats_b, labels)
    print(f"  +Transformer: ACC={r_b['accuracy_mean']:.4f} ±{r_b['accuracy_std']:.4f}")

    delta = r_b['accuracy_mean'] - r_a['accuracy_mean']
    print(f"\n  Delta (Transformer - No Transformer): {delta:+.4f}")

    output = {'experiment':'procyon_v2_A2_transformer',
              'no_transformer':r_a, 'with_transformer':r_b,
              'delta':delta}
    with open(os.path.join(RESULT_DIR,'procyon_v2_A2_transformer.json'),'w') as f:
        json.dump(output, f, indent=2)
    return output

# ═══════════════════════════════════════════════════════════════════
# Phase B1: SimpleEmb + Projection + Qwen LoRA
# Full ProCyon pipeline, replaces MGM encoder with SimpleEmb
# ═══════════════════════════════════════════════════════════════════
def phase_b1(seqs, masks):
    print("\n" + "="*60)
    print("Phase B1: SimpleEmb + Projection + Qwen LoRA")
    print("="*60)

    import json as _json, time as _time
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.utils.data import Dataset as _DS, DataLoader as _DL

    from run_v6_merged import Proj, MM, el, LABELS

    # Paths
    LLM_PATH = '/hd/gcr/hf_models/Qwen2.5-7B-Instruct'
    SAVE_DIR = '/hd/liujx/microbiome_llm_project/saved_models/procyon_v2_b1'
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Config (identical to V6)
    LH=3584; SL2=86; NMT=4; PS=0.1
    LR_R=16; LR_A=32; LR_D=0.03; BS=1; GA=8; NE=4; LR=3e-5; ML=1024; DW=1.5
    DROPOUT_PROB=0.5

    # Data
    print("\n[1/4] Loading data...")
    train_data = []; test_data = []
    with open(os.path.join(DATA_DIR,'train_nl.jsonl')) as f:
        for l in f: train_data.append(_json.loads(l))
    with open(os.path.join(DATA_DIR,'test_nl.jsonl')) as f:
        for l in f: test_data.append(_json.loads(l))
    train_seqs = np.load(os.path.join(DATA_DIR,'train_genus_sequences.npy'))
    test_seqs = np.load(os.path.join(DATA_DIR,'test_genus_sequences.npy'))
    train_masks = np.load(os.path.join(DATA_DIR,'train_genus_masks.npy'))
    test_masks = np.load(os.path.join(DATA_DIR,'test_genus_masks.npy'))
    print(f"  train={len(train_data)} test={len(test_data)}")

    tok = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)

    # SimpleEmb encoder (no Transformer)
    import torch.nn as nn
    class SimpleEmbEnc(nn.Module):
        def __init__(self): super().__init__(); self.emb=nn.Embedding(V,E,padding_idx=0)
        def forward(self, ids, mask=None):
            x=self.emb(ids); mf=mask.float().unsqueeze(-1) if mask is not None else torch.ones_like(x[:,:,0:1])
            return (x*mf).sum(dim=1)/mf.sum(dim=1).clamp(min=1)

    # Dataset (same as V6)
    class V2DS(_DS):
        def __init__(s, data, seqs, mks, tok, ml=ML, dw=DW):
            s.data=data; s.seqs=seqs; s.mks=mks; s.enc=[]; s.sw=[]
            for it in data:
                msgs=it['messages']
                fi=tok.apply_chat_template(msgs,tokenize=True,max_length=ml,truncation=True,add_generation_prompt=False).input_ids
                pi=tok.apply_chat_template([msgs[0]],tokenize=True,max_length=ml,truncation=True,add_generation_prompt=True).input_ids
                ul=len(pi); lb=[-100]*len(fi)
                for j in range(ul,len(fi)): lb[j]=fi[j]
                s.enc.append({'ids':fi,'lb':lb})
                s.sw.append(dw if it.get('label','Healthy')=='Disease' else 1.0)
        def __len__(s): return len(s.data)
        def __getitem__(s,i):
            e=s.enc[i]; sq=s.seqs[i].astype(np.int64); mk=s.mks[i]
            return {'input_ids':e['ids'],'attention_mask':[1]*len(e['ids']),'labels':e['lb'],
                    'genus_ids':sq,'genus_mask':mk,'sample_weights':s.sw[i]}

    class V2Coll:
        def __init__(s,tok,ml=ML): s.tok=tok; s.ml=ml; s.pid=tok.pad_token_id or 0
        def __call__(s,b):
            ids=[x['input_ids'] for x in b]; am=[x['attention_mask'] for x in b]
            lb=[x['labels'] for x in b]; gi=[x['genus_ids'] for x in b]
            gm=[x['genus_mask'] for x in b]; sw=[x['sample_weights'] for x in b]
            ml2=min(max(len(i) for i in ids),s.ml); pi2,pm,pl=[],[],[]
            for i in range(len(ids)):
                d=ids[i][:ml2]; m=am[i][:ml2]; l=lb[i][:ml2]; p=ml2-len(d)
                pi2.append(d+[s.pid]*p if p>0 else d)
                pm.append(m+[0]*p if p>0 else m)
                pl.append(l+[-100]*p if p>0 else l)
            tg=[g[:SL2] for g in gi]; tmm=[m[:SL2] for m in gm]
            mgl=max(len(g) for g in tg); pg,pm2=[],[]
            for i in range(len(tg)):
                g=tg[i]; m=tmm[i]; p=mgl-len(g)
                pg.append(np.pad(g,(0,p),constant_values=0) if p>0 else g)
                pm2.append(np.pad(m,(0,p),constant_values=False) if p>0 else m)
            return {'input_ids':torch.tensor(pi2,dtype=torch.long),
                    'attention_mask':torch.tensor(pm,dtype=torch.long),
                    'labels':torch.tensor(pl,dtype=torch.long),
                    'genus_ids':torch.tensor(np.array(pg),dtype=torch.long),
                    'genus_mask':torch.tensor(np.array(pm2),dtype=torch.bool),
                    'sample_weights':torch.tensor(sw,dtype=torch.float32)}

    coll = V2Coll(tok)
    train_ds = V2DS(train_data, train_seqs, train_masks, tok)
    test_ds = V2DS(test_data, test_seqs, test_masks, tok)

    print("[2/4] Building model: SimpleEmb + Proj + Qwen LoRA ...")
    import torch.nn as nn
    llm = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.bfloat16, trust_remote_code=True)
    lc = LoraConfig(r=LR_R,lora_alpha=LR_A,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],lora_dropout=LR_D,bias='none',task_type=TaskType.CAUSAL_LM)
    lm = get_peft_model(llm, lc)
    enc = SimpleEmbEnc()
    for p in enc.parameters(): p.requires_grad = False  # freeze encoder
    proj = Proj()
    mm = MM(lm, enc, proj, dropout_prob=DROPOUT_PROB)
    mm.cuda(); device = next(mm.parameters()).device
    trainable = sum(p.numel() for p in mm.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in mm.parameters())
    print(f"  Trainable: {trainable:,}/{total_p:,} ({100*trainable/total_p:.2f}%)")

    print("[3/4] Training...")
    opt = torch.optim.AdamW([p for p in mm.parameters() if p.requires_grad], lr=LR)
    for ep in range(NE):
        dl = _DL(train_ds, batch_size=BS, shuffle=True, collate_fn=coll)
        mm.train(); total_loss=0; n=0; t0=_time.time()
        print(f'  [Epoch {ep+1}/{NE}] {len(dl)} batches')
        for si, batch in enumerate(dl):
            out = mm(**batch); loss = out.loss/GA; loss.backward()
            total_loss += loss.item()*GA; n += 1
            if (si+1)%GA==0:
                torch.nn.utils.clip_grad_norm_(mm.parameters(),1.0)
                opt.step(); opt.zero_grad()
            if (si+1)%100==0:
                print(f'    step {si+1}/{len(dl)} loss={total_loss/max(n,1):.4f} elapsed={_time.time()-t0:.0f}s')
        print(f'  Epoch {ep+1} done, loss={total_loss/max(n,1):.4f}')

    # Save
    mm.llm.save_pretrained(os.path.join(SAVE_DIR,'llm_lora'))
    tok.save_pretrained(os.path.join(SAVE_DIR,'llm_lora'))
    torch.save({'proj_state_dict':mm.proj.state_dict()}, os.path.join(SAVE_DIR,'multimodal_components.pt'))

    # Evaluate (H4.1-style autoregressive)
    print("[4/4] Evaluating...")
    def eval_model(mm, test_data, test_seqs, test_masks, mode, max_tok=128):
        mm.eval(); correct=0; total=0
        with torch.no_grad():
            for i in range(len(test_data)):
                sq=test_seqs[i]; mk=test_masks[i]
                gi=torch.from_numpy(np.asarray(sq).astype(np.int64)).long().unsqueeze(0).to(device)
                gm=torch.from_numpy(np.asarray(mk)).bool().unsqueeze(0).to(device)
                me=mm.enc(gi,gm).to(mm.proj.p[0].weight.dtype)
                mt=mm.proj(me)
                if mode=='dropout': mt=mt*0.0
                msgs=test_data[i]['messages']
                prompt=tok.apply_chat_template([msgs[0]],tokenize=False,add_generation_prompt=True)
                pi=tok(prompt,return_tensors='pt',truncation=True,max_length=ML).to(device)
                te=mm.llm.base_model.model.model.embed_tokens(pi['input_ids'])
                mt=mt.to(te.dtype); ce=torch.cat([mt,te],dim=1)
                sl=ce.shape[1]; pid=torch.arange(0,sl,dtype=torch.long,device=device).unsqueeze(0)
                o=mm.llm(inputs_embeds=ce,position_ids=pid,use_cache=True)
                next_tok=torch.argmax(o.logits[:,-1,:],dim=-1,keepdim=True)
                generated=[next_tok]; cur_len=sl
                for _ in range(max_tok):
                    pos=torch.full((1,1),cur_len,dtype=torch.long,device=device)
                    out=mm.llm(input_ids=next_tok,position_ids=pos,past_key_values=o.past_key_values,use_cache=True)
                    next_tok=torch.argmax(out.logits[:,-1,:],dim=-1,keepdim=True)
                    if next_tok.item()==tok.eos_token_id: break
                    generated.append(next_tok); cur_len+=1
                    o.past_key_values=out.past_key_values
                gen_ids=torch.cat(generated,dim=1)
                gen_text=tok.decode(gen_ids[0],skip_special_tokens=True)
                pred=el(gen_text); true=test_data[i]['label']
                if pred: total+=1; correct+=1 if pred==true else 0
        return {'accuracy':correct/max(total,1),'correct':correct,'total':total}

    mm.set_dropout_prob(0.0)
    t0=_time.time()
    enc_nl = eval_model(mm, test_data, test_seqs, test_masks, 'normal')
    print(f"  Enc+NL: {enc_nl['accuracy']:.4f} ({enc_nl['correct']}/{enc_nl['total']}) time={_time.time()-t0:.0f}s")

    t0=_time.time()
    mm.set_dropout_prob(1.0)
    nl_only = eval_model(mm, test_data, test_seqs, test_masks, 'dropout')
    print(f"  NL-only: {nl_only['accuracy']:.4f} ({nl_only['correct']}/{nl_only['total']}) time={_time.time()-t0:.0f}s")

    gap = enc_nl['accuracy'] - nl_only['accuracy']
    print(f"  Gap: {gap:.4f}")

    output = {'experiment':'procyon_v2_B1_simpleemb_llm',
              'enc_nl':enc_nl,'nl_only':nl_only,'gap':gap,
              'dropout_prob':DROPOUT_PROB}
    with open(os.path.join(RESULT_DIR,'procyon_v2_B1.json'),'w') as f:
        json.dump(output, f, indent=2)
    return output

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("="*60)
    print("ProCyon v2 Automated Experiment Runner")
    print("="*60)
    print(f"Device: {DEVICE}")

    all_seqs, all_masks, all_labels = load_data()
    print(f"Data: {len(all_labels)} samples, Disease={all_labels.sum()}")

    all_results = {}

    # Phase A1: Pooling ablation
    all_results['A1'] = phase_a1(all_seqs, all_masks, all_labels)

    # Phase A2: Transformer usefulness
    all_results['A2'] = phase_a2(all_seqs, all_masks, all_labels)

    # Cleanup before B1
    gc.collect(); torch.cuda.empty_cache()

    # Phase B1: SimpleEmb + LLM
    all_results['B1'] = phase_b1(all_seqs, all_masks)

    # Final summary
    print("\n" + "="*70)
    print("PROCYON V2 — ALL EXPERIMENTS COMPLETE")
    print("="*70)

    print("\nPhase A1 — Pooling Ablation:")
    for k,v in all_results['A1'].items():
        print(f"  {k:>10}: {v['accuracy_mean']:.4f} ±{v['accuracy_std']:.4f}")

    print("\nPhase A2 — Transformer Usefulness:")
    a2 = all_results['A2']
    print(f"  No Transformer:  {a2['no_transformer']['accuracy_mean']:.4f} ±{a2['no_transformer']['accuracy_std']:.4f}")
    print(f"  +Transformer:    {a2['with_transformer']['accuracy_mean']:.4f} ±{a2['with_transformer']['accuracy_std']:.4f}")
    print(f"  Delta: {a2['delta']:+.4f}")

    print("\nPhase B1 — SimpleEmb + LLM:")
    b1 = all_results['B1']
    print(f"  Enc+NL:  {b1['enc_nl']['accuracy']:.4f} ({b1['enc_nl']['correct']}/{b1['enc_nl']['total']})")
    print(f"  NL-only: {b1['nl_only']['accuracy']:.4f} ({b1['nl_only']['correct']}/{b1['nl_only']['total']})")
    print(f"  Gap:     {b1['gap']:.4f}")

    # Comparison with MGM baseline
    print("\n--- vs MGM Baselines ---")
    print(f"  MGM + MLP:      0.8860")
    print(f"  MGM + LLM (V5): 0.8860")
    print(f"  MGM + LLM (V6): 0.8800")
    print(f"  SimpleEmb + MLP: {all_results['A1']['Mean']['accuracy_mean']:.4f}")

    with open(os.path.join(RESULT_DIR,'procyon_v2_summary.json'),'w') as f:
        json.dump({k: str(v) for k,v in all_results.items()}, f, indent=2, default=str)
    print(f"\nAll results saved to {RESULT_DIR}/procyon_v2_*.json")
    print("DONE")
