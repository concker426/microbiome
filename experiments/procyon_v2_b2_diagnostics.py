#!/usr/bin/env python3
"""
ProCyon v2 B2 — Projection Diagnostics
=======================================
Q: Why does SimpleEmb+LLM crash to 55.7% when SimpleEmb+MLP is 91.5%?

B2-zero: SimpleEmb + MGM's original Projection (sanity check)
B2a:     SimpleEmb + LayerNorm + Linear → LLM
B2b:     SimpleEmb + Deep Adapter (LN→2048→GELU→14336) → LLM
B2c:     SimpleEmb + Deep Adapter 8 tokens → LLM

All reuse: frozen SimpleEmb, same LoRA config, same data, same hyperparams.
"""
import os, sys, json, time, gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_FLASH_ATTN_ENABLED"] = "0"
import transformers.utils.import_utils as _iu; _iu.is_flash_attn_2_available = lambda: False
import transformers.utils as _utils; _utils.is_flash_attn_2_available = lambda: False
import accelerate.utils.imports as _ai; _ai.is_deepspeed_available = lambda: False
import accelerate.utils.other as _ao; _ao.is_deepspeed_available = lambda: False
sys.path.insert(0, '/hd/liujx/microbiome_llm_project')

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset as _DS, DataLoader as _DL
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

DATA_DIR = '/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
LLM_PATH = '/hd/gcr/hf_models/Qwen2.5-7B-Instruct'
RESULT_DIR = '/hd/liujx/microbiome_llm_project/experiments/results'
SAVE_BASE = '/hd/liujx/microbiome_llm_project/saved_models/procyon_v2_b2'
os.makedirs(RESULT_DIR, exist_ok=True)

V=1226; E=768; LH=3584; SL_V=86
NMT=4; PS=0.1; LR_R=16; LR_A=32; LR_D=0.03
BS=1; GA=8; NE=4; LR=3e-5; ML=1024; DW=1.5; DROPOUT_PROB=0.5
LABELS=["Healthy","Disease"]
from run_v6_merged import el

# ═══════════════════════════════════════
# Generic MM (works with any proj)
# ═══════════════════════════════════════
class MM(nn.Module):
    def __init__(self, llm, enc, proj, dropout_prob=DROPOUT_PROB):
        super().__init__(); self.llm=llm; self.enc=enc; self.proj=proj
        self.config=llm.config; self.nmt=proj.nt; self.dropout_prob=dropout_prob
    def gradient_checkpointing_enable(self,**kw): self.llm.gradient_checkpointing_enable(**kw)
    def set_dropout_prob(self,p): self.dropout_prob=p
    def forward(self, input_ids=None, attention_mask=None, labels=None,
                genus_ids=None, genus_mask=None, sample_weights=None, **kw):
        B=genus_ids.shape[0]; d=next(self.parameters()).device; nt=self.nmt
        if input_ids is not None: input_ids=input_ids.to(d)
        if attention_mask is not None: attention_mask=attention_mask.to(d)
        if labels is not None: labels=labels.to(d)
        genus_ids=genus_ids.to(d); genus_mask=genus_mask.to(d)
        S=input_ids.shape[1]
        me=self.enc(genus_ids, genus_mask)
        p_dtype = next(self.proj.parameters()).dtype
        me = me.to(p_dtype)
        mt=self.proj(me)
        if self.training and self.dropout_prob>0:
            mask=torch.rand(B,1,1,device=mt.device)>self.dropout_prob
            mt=mt*mask.float()
        te=self.llm.base_model.model.model.embed_tokens(input_ids)
        mt=mt.to(te.dtype); ce=torch.cat([mt,te],dim=1)
        if labels is not None:
            nl=torch.full((B,S+nt),-100,device=labels.device,dtype=labels.dtype)
            nl[:,nt:]=labels
        else: nl=None
        if attention_mask is not None:
            nm=torch.ones(B,S+nt,device=attention_mask.device,dtype=attention_mask.dtype)
            nm[:,nt:]=attention_mask
        else: nm=None
        pid=torch.arange(ce.shape[1],dtype=torch.long,device=ce.device).unsqueeze(0)
        o=self.llm(inputs_embeds=ce,attention_mask=nm,position_ids=pid,labels=nl,**kw)
        if sample_weights is not None and o.loss is not None:
            sw=sample_weights.to(d)
            lo=o.logits[:,:-1,:].contiguous()
            sl2=nl[:,1:].contiguous() if nl is not None else None
            tl=F.cross_entropy(lo.view(-1,lo.size(-1)),sl2.view(-1),reduction='none').view(B,-1)
            vm=(sl2!=-100).float(); tl=tl*vm
            sloss=tl.sum(dim=1)/vm.sum(dim=1).clamp(min=1)
            o.loss=(sloss*sw).sum()/sw.sum()
        return o

# ═══════════════════════════════════════
# SimpleEmb Encoder (shared, frozen)
# ═══════════════════════════════════════
class SimpleEmbEnc(nn.Module):
    def __init__(self):
        super().__init__(); self.emb = nn.Embedding(V, E, padding_idx=0)
    def forward(self, ids, mask=None):
        x = self.emb(ids)
        mf = mask.float().unsqueeze(-1) if mask is not None else torch.ones_like(x[..., :1])
        return (x * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1)

# ═══════════════════════════════════════
# Projection variants
# ═══════════════════════════════════════
class ProjZero(nn.Module):
    """B2-zero: MGM's original Projection (LN→768→7168→GELU→7168→14336→LN→×0.1)"""
    def __init__(self):
        super().__init__()
        from run_v6_merged import Proj as _Proj
        self.p = _Proj(); self.nt = self.p.nt
    def forward(self, x): return self.p(x)

class ProjLN(nn.Module):
    """B2a: LayerNorm → Linear(768→3584*N) → reshape → ×scale"""
    def __init__(self, nt=NMT):
        super().__init__()
        self.ln = nn.LayerNorm(E)
        self.linear = nn.Linear(E, LH*nt)
        self.sc = nn.Parameter(torch.ones(1)*PS); self.nt = nt
    def forward(self, x):
        x = self.ln(x); x = self.linear(x)
        x = x.view(-1, self.nt, LH)
        return x * self.sc

class ProjAdapter(nn.Module):
    """B2b: LN → 768→2048 → GELU → 2048→3584*N → reshape → ×scale"""
    def __init__(self, nt=NMT):
        super().__init__()
        self.ln = nn.LayerNorm(E)
        self.fc1 = nn.Linear(E, 2048)
        self.fc2 = nn.Linear(2048, LH*nt)
        self.sc = nn.Parameter(torch.ones(1)*PS); self.nt = nt
    def forward(self, x):
        x = self.ln(x); x = F.gelu(self.fc1(x))
        x = self.fc2(x); x = x.view(-1, self.nt, LH)
        return x * self.sc

# ═══════════════════════════════════════
# MM wrapper (reuse MM base with custom enc+proj)
# ═══════════════════════════════════════
def build_mm(proj, device='cuda:0'):
    enc = SimpleEmbEnc()
    for p in enc.parameters(): p.requires_grad = False
    enc.to(device)
    proj.to(device)
    llm_base = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.bfloat16, trust_remote_code=True)
    lc = LoraConfig(r=LR_R, lora_alpha=LR_A, target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'], lora_dropout=LR_D, bias='none', task_type=TaskType.CAUSAL_LM)
    lm = get_peft_model(llm_base, lc)
    mm = MM(lm, enc, proj, dropout_prob=DROPOUT_PROB)
    mm.to(device)
    return mm

# ═══════════════════════════════════════
# Data (same as V6)
# ═══════════════════════════════════════
def load_data():
    train_data, test_data = [], []
    with open(os.path.join(DATA_DIR,'train_nl.jsonl')) as f:
        for l in f: train_data.append(json.loads(l))
    with open(os.path.join(DATA_DIR,'test_nl.jsonl')) as f:
        for l in f: test_data.append(json.loads(l))
    ts=np.load(os.path.join(DATA_DIR,'train_genus_sequences.npy'))
    xs=np.load(os.path.join(DATA_DIR,'test_genus_sequences.npy'))
    tm=np.load(os.path.join(DATA_DIR,'train_genus_masks.npy'))
    xm=np.load(os.path.join(DATA_DIR,'test_genus_masks.npy'))
    return train_data, test_data, ts, xs, tm, xm

def build_datasets(train_data, test_data, ts, xs, tm, xm, tok):
    class DS(_DS):
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
    class Coll:
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
            tg=[g[:SL_V] for g in gi]; tmm=[m[:SL_V] for m in gm]
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
    return DS(train_data,ts,tm,tok), DS(test_data,xs,xm,tok), Coll(tok)

# ═══════════════════════════════════════
# Training
# ═══════════════════════════════════════
def train_one(mm, train_ds, name, device):
    opt = torch.optim.AdamW([p for p in mm.parameters() if p.requires_grad], lr=LR)
    for ep in range(NE):
        dl = _DL(train_ds, batch_size=BS, shuffle=True, collate_fn=coll)
        mm.train(); total_loss=0; n=0; t0=time.time()
        for si, batch in enumerate(dl):
            out = mm(**batch); loss = out.loss / GA; loss.backward()
            total_loss += loss.item()*GA; n += 1
            if (si+1)%GA==0:
                torch.nn.utils.clip_grad_norm_(mm.parameters(),1.0)
                opt.step(); opt.zero_grad()
        print(f'  [{name}] Epoch {ep+1}/{NE} loss={total_loss/max(n,1):.4f} time={time.time()-t0:.0f}s')
    return mm

# ═══════════════════════════════════════
# Evaluation (H4.1-style autoregressive)
# ═══════════════════════════════════════
def evaluate_mm(mm, test_data, test_seqs, test_masks, mode, device, max_tok=128):
    mm.eval(); correct=0; total=0
    with torch.no_grad():
        for i in range(len(test_data)):
            sq=test_seqs[i]; mk=test_masks[i]
            gi=torch.from_numpy(np.asarray(sq).astype(np.int64)).long().unsqueeze(0).to(device)
            gm=torch.from_numpy(np.asarray(mk)).bool().unsqueeze(0).to(device)
            me=mm.enc(gi,gm).to(next(mm.proj.parameters()).dtype)
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
            gen_text=tok.decode(torch.cat(generated,dim=1)[0],skip_special_tokens=True)
            pred=el(gen_text); true=test_data[i]['label']
            if pred: total+=1; correct+=1 if pred==true else 0
    return {'accuracy':correct/max(total,1),'correct':correct,'total':total}

# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════
if __name__ == '__main__':
    print("="*60)
    print("ProCyon v2 — B2 Projection Diagnostics")
    print("="*60)

    tok = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
    train_data, test_data, ts, xs, tm, xm = load_data()
    train_ds, test_ds, coll = build_datasets(train_data, test_data, ts, xs, tm, xm, tok)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"train={len(train_data)} test={len(test_data)} device={device}")

    variants = {
        'B2-zero (MGM Proj)':  ProjZero,
        'B2a (LN+Linear)':     ProjLN,
        'B2b (Adapter)':       ProjAdapter,
        'B2c (Adapter 8t)':    lambda: ProjAdapter(nt=8),
    }

    all_results = {}

    for name, proj_cls in variants.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        proj = proj_cls()
        mm = build_mm(proj, device)
        trainable = sum(p.numel() for p in mm.parameters() if p.requires_grad)
        print(f"  Trainable: {trainable:,}")

        t0 = time.time()
        mm = train_one(mm, train_ds, name, device)

        # Eval Enc+NL
        mm.set_dropout_prob(0.0)
        enc_nl = evaluate_mm(mm, test_data, xs, xm, 'normal', device)
        # Eval NL-only
        mm.set_dropout_prob(1.0)
        nl_only = evaluate_mm(mm, test_data, xs, xm, 'dropout', device)
        gap = enc_nl['accuracy'] - nl_only['accuracy']

        print(f"  Enc+NL:  {enc_nl['accuracy']:.4f} ({enc_nl['correct']}/{enc_nl['total']})")
        print(f"  NL-only: {nl_only['accuracy']:.4f} ({nl_only['correct']}/{nl_only['total']})")
        print(f"  Gap:     {gap:.4f}")
        print(f"  Time:    {time.time()-t0:.0f}s")

        all_results[name] = {'enc_nl': enc_nl, 'nl_only': nl_only, 'gap': gap}

        # Cleanup
        del mm; gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("  B2 DIAGNOSTICS — FINAL RESULTS")
    print(f"  {'-'*50}")
    print(f"  {'Variant':<25} {'Enc+NL':>10} {'NL-only':>10} {'Gap':>10}")
    print(f"  {'-'*50}")
    baselines = {
        'MGM+LLM (V5)':       (0.8862, 0.5330),
        'MGM+LLM (V6 d=0.5)': (0.8800, 0.7960),
        'SimpleEmb+MLP':       (0.9145, None),
        'B1 (SimpleEmb+LLM)':  (0.5569, 0.5808),
    }
    for name, (enc, nl) in baselines.items():
        g = enc-nl if nl else None
        print(f"  {name:<25} {enc:>10.4f} {nl or '':>10} {g or '':>10}")
    print(f"  {'-'*50}")
    for name, r in all_results.items():
        print(f"  {name:<25} {r['enc_nl']['accuracy']:>10.4f} {r['nl_only']['accuracy']:>10.4f} {r['gap']:>10.4f}")
    print(f"  {'='*70}")

    with open(os.path.join(RESULT_DIR,'procyon_v2_B2.json'),'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print("DONE")
