#!/usr/bin/env python3
"""Stage 2: Load Stage-1 domain-adapted LoRA, add MGM encoder + Projection,
then train full ProCyon pipeline with dropout.

Hypothesis: Domain-adapted LLM (Stage 1) absorbs microbiome signals faster,
resulting in smaller Enc+NL vs NL-only gap and potentially higher accuracy.

Comparison: V6 baseline (random LoRA, H4.1 dropout=0.5):
  Enc+NL=88.0%, NL-only=79.6%, gap=8.4%
"""
import os, sys, re, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_FLASH_ATTN_ENABLED"] = "0"
# Inline flash_attn patch (bypass sitecustomize block)
import transformers.utils.import_utils as _iu
_iu.is_flash_attn_2_available = lambda: False
import transformers.utils as _utils
_utils.is_flash_attn_2_available = lambda: False
_iu.is_flash_attn_greater_or_equal_2_10 = lambda: False
_utils.is_flash_attn_greater_or_equal_2_10 = lambda: False
import accelerate.utils.imports as _ai; _ai.is_deepspeed_available = lambda: False
import accelerate.utils.other as _ao; _ao.is_deepspeed_available = lambda: False
from collections import Counter
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ── Paths ──────────────────────────────────────────
DATA_DIR = "/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538"
LLM_PATH = "/hd/gcr/hf_models/Qwen2.5-7B-Instruct"
ENCODER_PATH = "/hd/liujx/microbiome_llm_project/saved_models/mgm_encoder_pretrained/mgm_encoder.pt"
STAGE1_DIR = "/hd/liujx/microbiome_llm_project/saved_models/stage1_text_only"
OUT_DIR = "/hd/liujx/microbiome_llm_project/saved_models/stage2_full"
RESULT_DIR = "/hd/liujx/microbiome_llm_project/experiments/results"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ── Tokenizer ───────────────────────────────────────
tok = AutoTokenizer.from_pretrained(STAGE1_DIR, trust_remote_code=True)

# ── Config ──────────────────────────────────────────
V=1226; E=768; LH=3584; SL=86; NL=6; NH=8; FF=2048; DP=0.1
NMT=4; PS=0.1; LR_R=16; LR_A=32; LR_D=0.03
BS=1; GA=8; NE=4; LR=3e-5; ML=1024; DW=1.5
DROPOUT_PROB=0.5  # Fixed dropout, = H4.1's best
LABELS=["Healthy","Disease"]

# ── Encoder (copied from run_v6_merged) ─────────────
class GenusEmb(nn.Module):
    def __init__(s,v,e,ms,d=0.1):
        super().__init__()
        s.te=nn.Embedding(v,e,padding_idx=0); s.pe=nn.Embedding(ms,e); s.drop=nn.Dropout(d)
    def forward(s,ids):
        B,L=ids.shape
        return s.drop(s.te(ids)+s.pe(torch.arange(L,device=ids.device).unsqueeze(0).expand(B,-1)))

class AttnPool(nn.Module):
    def __init__(s,e):
        super().__init__(); s.lin=nn.Linear(e,1)
    def forward(s,x,m=None):
        sc=torch.tanh(s.lin(x)).squeeze(-1)
        if m is not None: sc=sc.masked_fill(~m,float('-inf'))
        w=F.softmax(sc,dim=-1)
        return (x*w.unsqueeze(-1)).sum(dim=1)

class TBlock(nn.Module):
    def __init__(s,e,nh,ff,d=0.1):
        super().__init__()
        s.ln1=nn.LayerNorm(e); s.at=nn.MultiheadAttention(e,nh,dropout=d,batch_first=True)
        s.ln2=nn.LayerNorm(e)
        s.ff=nn.Sequential(nn.Linear(e,ff),nn.GELU(),nn.Dropout(d),nn.Linear(ff,e),nn.Dropout(d))
    def forward(s,x,kpm=None):
        xn=s.ln1(x); x=x+s.at(xn,xn,xn,attn_mask=kpm,need_weights=False)[0]
        return x+s.ff(s.ln2(x))

class MGMEnc(nn.Module):
    def __init__(s,v=V,e=E,nl=NL,nh=NH,ff=FF,dp=DP,ms=SL):
        super().__init__()
        s.ge=GenusEmb(v,e,ms,dp)
        s.blocks=nn.ModuleList([TBlock(e,nh,ff,dp) for _ in range(nl)])
        s.pool=AttnPool(e)
        s.register_buffer('cm',torch.triu(torch.ones(ms,ms),1).bool(),persistent=False)
    def forward(s,ids,m=None):
        x=s.ge(ids)
        for b in s.blocks: x=b(x,kpm=s.cm)
        return s.pool(x,m)

class Proj(nn.Module):
    def __init__(s,e=E,lh=LH,nt=NMT,sc=PS):
        super().__init__()
        s.ni=nn.LayerNorm(e)
        s.p=nn.Sequential(nn.Linear(e,lh*2),nn.GELU(),nn.Linear(lh*2,lh*nt))
        s.no=nn.LayerNorm(lh)
        s.sc=nn.Parameter(torch.ones(1)*sc); s.nt=nt; s.lh=lh
    def forward(s,x):
        x=s.ni(x); x=s.p(x); x=x.view(-1,s.nt,s.lh); x=s.no(x); return x*s.sc

class MM(nn.Module):
    def __init__(s,llm,enc,proj,dropout_prob=DROPOUT_PROB):
        super().__init__(); s.llm=llm; s.enc=enc; s.proj=proj; s.config=llm.config
        s.nmt=proj.nt; s.dropout_prob=dropout_prob
    def gradient_checkpointing_enable(s,**kw): s.llm.gradient_checkpointing_enable(**kw)
    def set_dropout_prob(s,p): s.dropout_prob=p
    def forward(s,input_ids=None,attention_mask=None,labels=None,
                genus_ids=None,genus_mask=None,sample_weights=None,**kw):
        B=genus_ids.shape[0]; d=next(s.parameters()).device; nt=s.nmt
        input_ids=input_ids.to(d); attention_mask=attention_mask.to(d)
        S=input_ids.shape[1]; ls_d=labels
        genus_ids=genus_ids.to(d); genus_mask=genus_mask.to(d)
        me=s.enc(genus_ids,genus_mask).to(s.proj.p[0].weight.dtype)
        mt=s.proj(me)
        if s.training and s.dropout_prob>0:
            mask=torch.rand(B,1,1,device=mt.device)>s.dropout_prob
            mt=mt*mask.float()
        te=s.llm.base_model.model.model.embed_tokens(input_ids)
        mt=mt.to(te.dtype); ce=torch.cat([mt,te],dim=1)
        if labels is not None:
            labels=labels.to(d)
            nl=torch.full((B,S+nt),-100,device=labels.device,dtype=labels.dtype)
            nl[:,nt:]=labels
        else: nl=None
        if attention_mask is not None:
            nm=torch.ones(B,S+nt,device=attention_mask.device,dtype=attention_mask.dtype)
            nm[:,nt:]=attention_mask
        else: nm=None
        pid=torch.arange(ce.shape[1],dtype=torch.long,device=ce.device).unsqueeze(0)
        o=s.llm(inputs_embeds=ce,attention_mask=nm,position_ids=pid,labels=nl,**kw)
        if sample_weights is not None and o.loss is not None:
            lo=o.logits[:,:-1,:].contiguous()
            sl2=nl[:,1:].contiguous() if nl is not None else None
            sw=sample_weights.to(d)
            tl=F.cross_entropy(lo.view(-1,lo.size(-1)),sl2.view(-1),reduction='none').view(B,-1)
            vm=(sl2!=-100).float(); tl=tl*vm
            sloss=tl.sum(dim=1)/vm.sum(dim=1).clamp(min=1)
            o.loss=(sloss*sw).sum()/sw.sum()
        return o

# ── Data ────────────────────────────────────────────
def lj(p):
    d=[]
    with open(p) as f:
        for l in f: d.append(json.loads(l))
    return d

def tm(tok,msgs,ml,agp=False):
    r=tok.apply_chat_template(msgs,tokenize=True,max_length=ml,truncation=True,add_generation_prompt=agp)
    return r.input_ids

class DS(Dataset):
    def __init__(s,data,seqs,masks,tok,ml=ML,dw=DW):
        s.data=data; s.seqs=seqs; s.masks=masks; s.enc=[]; s.sw=[]
        for it in data:
            msgs=it['messages']; fi=tm(tok,msgs,ml); pi=tm(tok,[msgs[0]],ml,True)
            ul=len(pi); lb=[-100]*len(fi)
            for i in range(ul,len(fi)): lb[i]=fi[i]
            s.enc.append({'ids':fi,'lb':lb})
            s.sw.append(dw if it.get('label','Healthy')=='Disease' else 1.0)
    def __len__(s): return len(s.data)
    def __getitem__(s,i):
        e=s.enc[i]; sq=s.seqs[i].astype(np.int64); mk=s.masks[i]
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
            if p>0: pi2.append(d+[s.pid]*p); pm.append(m+[0]*p); pl.append(l+[-100]*p)
            else: pi2.append(d); pm.append(m); pl.append(l)
        tg=[g[:SL] for g in gi]; tmm=[m[:SL] for m in gm]
        mgl=max(len(g) for g in tg); pg,pm2=[],[]
        for i in range(len(tg)):
            g=tg[i]; m=tmm[i]; p=mgl-len(g)
            if p>0: pg.append(np.pad(g,(0,p),constant_values=0)); pm2.append(np.pad(m,(0,p),constant_values=False))
            else: pg.append(g); pm2.append(m)
        return {'input_ids':torch.tensor(pi2,dtype=torch.long),
                'attention_mask':torch.tensor(pm,dtype=torch.long),
                'labels':torch.tensor(pl,dtype=torch.long),
                'genus_ids':torch.tensor(np.array(pg),dtype=torch.long),
                'genus_mask':torch.tensor(np.array(pm2),dtype=torch.bool),
                'sample_weights':torch.tensor(sw,dtype=torch.float32)}

# ── Eval ────────────────────────────────────────────
def extract_label(text):
    m=re.search(r'\xe8\xaf\x8a\xe6\x96\xad\xe7\xbb\x93\xe6\x9e\x9c[\xef\xbc\x9a:]\s*(\S+)',text)
    if m:
        lb=m.group(1).strip()
        if lb in LABELS: return lb
    for k in LABELS:
        if k in text: return k
    for cn,en in [('\xe5\x81\xa5\xe5\xba\xb7','Healthy'),('\xe7\x96\xbe\xe7\x97\x85','Disease')]:
        if cn in text: return en
    return None

def evaluate_model(mm, test_data, test_seqs, test_masks, device, max_tok=128):
    """H4.1-style autoregressive eval — proven correct."""
    mm.eval()
    correct=0; total=0; preds=[]
    with torch.no_grad():
        for i in range(len(test_data)):
            sq=test_seqs[i]; mk=test_masks[i]
            gi=torch.from_numpy(np.asarray(sq).astype(np.int64)).long().unsqueeze(0).to(device)
            gm=torch.from_numpy(np.asarray(mk)).bool().unsqueeze(0).to(device)
            me=mm.enc(gi,gm).to(mm.proj.p[0].weight.dtype)
            mt=mm.proj(me)
            msgs=test_data[i]['messages']
            prompt=tok.apply_chat_template([msgs[0]],tokenize=False,add_generation_prompt=True)
            pi=tok(prompt,return_tensors='pt',truncation=True,max_length=ML).to(device)
            te=mm.llm.base_model.model.model.embed_tokens(pi['input_ids'])
            mt=mt.to(te.dtype); ce=torch.cat([mt,te],dim=1)
            sl=ce.shape[1]
            pid=torch.arange(0,sl,dtype=torch.long,device=device).unsqueeze(0)
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
            pred=extract_label(gen_text)
            true=test_data[i]['label']
            if pred:
                total+=1
                if pred==true: correct+=1
    acc=correct/max(total,1)
    return {'accuracy':acc,'correct':correct,'total':total}

# ── Main ────────────────────────────────────────────
def main():
    print("="*60)
    print("Stage 2: Domain-adapted LLM + MGM Encoder")
    print("="*60)

    print("\n[1/4] Loading data (clean_2538)...")
    train_data = lj(os.path.join(DATA_DIR, 'train_nl.jsonl'))
    test_data = lj(os.path.join(DATA_DIR, 'test_nl.jsonl'))
    train_seqs = np.load(os.path.join(DATA_DIR, 'train_genus_sequences.npy'))
    test_seqs = np.load(os.path.join(DATA_DIR, 'test_genus_sequences.npy'))
    train_masks = np.load(os.path.join(DATA_DIR, 'train_genus_masks.npy'))
    test_masks = np.load(os.path.join(DATA_DIR, 'test_genus_masks.npy'))
    print(f"  train={len(train_data)} test={len(test_data)}")

    coll = Coll(tok)
    train_ds = DS(train_data, train_seqs, train_masks, tok)
    test_ds = DS(test_data, test_seqs, test_masks, tok)

    print("\n[2/5] Loading Stage-1 domain-adapted LLM...")
    base_llm = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.bfloat16, trust_remote_code=True)
    # Load Stage 1 LoRA (already has IBD domain knowledge)
    lm = PeftModel.from_pretrained(base_llm, STAGE1_DIR, is_trainable=True)
    print("  Stage-1 LoRA loaded (IBD domain-adapted)")

    print("[3/5] Adding MGM encoder + Projection (fresh init)...")
    enc = MGMEnc()
    ck = torch.load(ENCODER_PATH, map_location='cpu')
    st = ck.get('model_state_dict', ck)
    enc.load_state_dict(st, strict=False)
    for p in enc.parameters(): p.requires_grad = False

    proj = Proj()

    mm = MM(lm, enc, proj, dropout_prob=DROPOUT_PROB)
    mm.cuda(); device = next(mm.parameters()).device
    trainable = sum(p.numel() for p in mm.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in mm.parameters())
    print(f"  Trainable: {trainable:,} / {total_p:,} ({100*trainable/total_p:.2f}%)")

    print("[4/5] Training...")
    from torch.utils.data import DataLoader
    opt = torch.optim.AdamW([p for p in mm.parameters() if p.requires_grad], lr=LR)

    for ep in range(NE):
        dl = DataLoader(train_ds, batch_size=BS, shuffle=True, collate_fn=coll)
        mm.train()
        total_loss = 0; n = 0; t0 = time.time()
        print(f'  [Epoch {ep+1}/{NE}] {len(dl)} batches, dropout={DROPOUT_PROB}')
        for si, batch in enumerate(dl):
            out = mm(**batch)
            loss = out.loss / GA; loss.backward()
            total_loss += loss.item() * GA; n += 1
            if (si+1) % GA == 0:
                torch.nn.utils.clip_grad_norm_(mm.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            if (si+1) % 50 == 0:
                print(f'    step {si+1}/{len(dl)} loss={total_loss/max(n,1):.4f} elapsed={time.time()-t0:.0f}s')
        print(f'  Epoch {ep+1} done, loss={total_loss/max(n,1):.4f}')

    print("[5/5] Evaluating...")
    # Enc+NL eval
    mm.set_dropout_prob(0.0)
    enc_nl = evaluate_model(mm, test_data, test_seqs, test_masks, device)
    # NL-only eval (zero projection tokens)
    mm.set_dropout_prob(1.0)
    nl_only = evaluate_model(mm, test_data, test_seqs, test_masks, device)
    # Reset dropout
    mm.set_dropout_prob(0.0)

    gap = enc_nl['accuracy'] - nl_only['accuracy']

    print(f"\n{'='*70}")
    print(f"  STAGE 2 RESULTS (clean_2538, dropout={DROPOUT_PROB})")
    print(f"  {'-'*50}")
    print(f"  Enc+NL:   {enc_nl['accuracy']:.4f} ({enc_nl['correct']}/{enc_nl['total']})")
    print(f"  NL-only:  {nl_only['accuracy']:.4f} ({nl_only['correct']}/{nl_only['total']})")
    print(f"  Gap:      {gap:.4f} ({gap*100:.1f}%)")
    print(f"  {'='*70}")

    # Baseline comparison
    print(f"\n  vs V6 baseline (random LoRA, dropout=0.5):")
    print(f"    V6:  Enc+NL=0.8800  NL-only=0.7960  gap=0.0840")
    print(f"    S2:  Enc+NL={enc_nl['accuracy']:.4f}  NL-only={nl_only['accuracy']:.4f}  gap={gap:.4f}")

    # Save
    mm.llm.save_pretrained(os.path.join(OUT_DIR, 'llm_lora'))
    tok.save_pretrained(os.path.join(OUT_DIR, 'llm_lora'))
    torch.save({'proj_state_dict': mm.proj.state_dict()}, os.path.join(OUT_DIR, 'multimodal_components.pt'))
    print(f"\nSaved to {OUT_DIR}")

    result = {
        'experiment': 'stage2_full',
        'enc_nl': enc_nl,
        'nl_only': nl_only,
        'gap': gap,
        'dropout_prob': DROPOUT_PROB,
        'train_samples': len(train_data),
        'test_samples': len(test_data),
        'epochs': NE, 'lr': LR, 'bs': BS, 'ga': GA,
        'timestamp': str(__import__('datetime').datetime.now()),
    }
    with open(os.path.join(RESULT_DIR, 'stage2_full.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved results to {RESULT_DIR}/stage2_full.json")

if __name__ == '__main__':
    main()
