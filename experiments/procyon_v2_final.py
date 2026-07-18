#!/usr/bin/env python3
"""
ProCyon v2 — Final Stability Validation
=========================================
Dataset: merged_all (3350 train, 838 test)
Model:   SimpleEmb + Adapter 8t + Qwen2.5-7B LoRA
Seeds:   42, 123, 456

Hypothesis: 5x more data stabilizes SimpleEmb→LLM alignment,
eliminating the high-variance issue seen on clean_2538.
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from run_v6_merged import el, LABELS

DATA_DIR = '/hd/liujx/microbiome_llm_project/data/qiita_ibd/merged_all'
LLM_PATH = '/hd/gcr/hf_models/Qwen2.5-7B-Instruct'
RESULT_DIR = '/hd/liujx/microbiome_llm_project/experiments/results'
SAVE_DIR = '/hd/liujx/microbiome_llm_project/saved_models/procyon_v2_final'
os.makedirs(RESULT_DIR, exist_ok=True); os.makedirs(SAVE_DIR, exist_ok=True)

V=1226; E=768; LH=3584; SL=175  # merged_all has 175 positions
NT=8; PS=0.1; LR_R=16; LR_A=32; LR_D=0.03
BS=1; GA=8; NE=4; LR=3e-5; ML=1024; DW=1.5; DP=0.5
DEVICE='cuda:0' if torch.cuda.is_available() else 'cpu'
SEEDS=[42, 123, 456]

# ═══════════════════ Components ═══════════════════
class SimpleEmbEnc(nn.Module):
    def __init__(self): super().__init__(); self.emb=nn.Embedding(V,E,padding_idx=0)
    def forward(self,ids,mask=None):
        x=self.emb(ids)
        mf=mask.float().unsqueeze(-1) if mask is not None else torch.ones_like(x[...,:1])
        return (x*mf).sum(dim=1)/mf.sum(dim=1).clamp(min=1)

class AdapterProj8(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln=nn.LayerNorm(E); self.fc1=nn.Linear(E,2048)
        self.fc2=nn.Linear(2048,LH*NT); self.sc=nn.Parameter(torch.ones(1)*PS); self.nt=NT
    def forward(self,x):
        x=self.ln(x); x=F.gelu(self.fc1(x)); x=self.fc2(x)
        return x.view(-1,self.nt,LH)*self.sc

class MM(nn.Module):
    def __init__(self,llm,enc,proj,dropout_prob=DP):
        super().__init__(); self.llm=llm; self.enc=enc; self.proj=proj
        self.config=llm.config; self.nmt=proj.nt; self.dropout_prob=dropout_prob
    def set_dropout_prob(self,p): self.dropout_prob=p
    def forward(self,input_ids=None,attention_mask=None,labels=None,
                genus_ids=None,genus_mask=None,sample_weights=None,**kw):
        B=genus_ids.shape[0]; d=next(self.parameters()).device; nt=self.nmt
        input_ids=input_ids.to(d); attention_mask=attention_mask.to(d)
        if labels is not None: labels=labels.to(d)
        genus_ids=genus_ids.to(d); genus_mask=genus_mask.to(d)
        S=input_ids.shape[1]
        me=self.enc(genus_ids,genus_mask).to(next(self.proj.parameters()).dtype)
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

# ═══════════════════ Data ═══════════════════
def load_data():
    train_data,test_data=[],[]
    with open(f'{DATA_DIR}/train_nl.jsonl') as f:
        for l in f: train_data.append(json.loads(l))
    with open(f'{DATA_DIR}/test_nl.jsonl') as f:
        for l in f: test_data.append(json.loads(l))
    ts=np.load(f'{DATA_DIR}/train_genus_sequences.npy')
    xs=np.load(f'{DATA_DIR}/test_genus_sequences.npy')
    tm=np.load(f'{DATA_DIR}/train_genus_masks.npy')
    xm=np.load(f'{DATA_DIR}/test_genus_masks.npy')
    return train_data,test_data,ts,xs,tm,xm

def build_datasets(train_data,test_data,ts,xs,tm,xm,tok):
    class DS(Dataset):
        def __init__(s,data,seqs,mks):
            s.data=data; s.seqs=seqs; s.mks=mks; s.enc=[]; s.sw=[]
            for it in data:
                msgs=it['messages']
                fi=tok.apply_chat_template(msgs,tokenize=True,max_length=ML,truncation=True,add_generation_prompt=False).input_ids
                pi=tok.apply_chat_template([msgs[0]],tokenize=True,max_length=ML,truncation=True,add_generation_prompt=True).input_ids
                ul=len(pi); lb=[-100]*len(fi)
                for j in range(ul,len(fi)): lb[j]=fi[j]
                s.enc.append({'ids':fi,'lb':lb})
                s.sw.append(DW if it.get('label','Healthy')=='Disease' else 1.0)
        def __len__(s): return len(s.data)
        def __getitem__(s,i):
            e=s.enc[i]; sq=s.seqs[i].astype(np.int64); mk=s.mks[i]
            return {'input_ids':e['ids'],'attention_mask':[1]*len(e['ids']),'labels':e['lb'],
                    'genus_ids':sq,'genus_mask':mk,'sample_weights':s.sw[i]}
    class Coll:
        def __init__(s,tok): s.tok=tok; s.pid=tok.pad_token_id or 0
        def __call__(s,b):
            ids=[x['input_ids'] for x in b]; am=[x['attention_mask'] for x in b]
            lb=[x['labels'] for x in b]; gi=[x['genus_ids'] for x in b]
            gm=[x['genus_mask'] for x in b]; sw=[x['sample_weights'] for x in b]
            ml2=min(max(len(i) for i in ids),ML); pi2,pm,pl=[],[],[]
            for i in range(len(ids)):
                d=ids[i][:ml2]; m=am[i][:ml2]; l=lb[i][:ml2]; p=ml2-len(d)
                pi2.append(d+[s.pid]*p if p>0 else d)
                pm.append(m+[0]*p if p>0 else m)
                pl.append(l+[-100]*p if p>0 else l)
            tg=[g[:SL] for g in gi]; tmm=[m[:SL] for m in gm]
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
    return DS(train_data,ts,tm), DS(test_data,xs,xm), Coll(tok)

# ═══════════════════ Evaluation ═══════════════════
def evaluate_mm(mm,test_data,xs,xm,mode,tok,device,max_tok=128):
    mm.eval(); correct=0; total=0
    with torch.no_grad():
        for i in range(len(test_data)):
            sq=xs[i]; mk=xm[i]
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
            nt2=torch.argmax(o.logits[:,-1,:],dim=-1,keepdim=True)
            generated=[nt2]; cur_len=sl
            for _ in range(max_tok):
                pos=torch.full((1,1),cur_len,dtype=torch.long,device=device)
                out=mm.llm(input_ids=nt2,position_ids=pos,past_key_values=o.past_key_values,use_cache=True)
                nt2=torch.argmax(out.logits[:,-1,:],dim=-1,keepdim=True)
                if nt2.item()==tok.eos_token_id: break
                generated.append(nt2); cur_len+=1
                o.past_key_values=out.past_key_values
            gen_text=tok.decode(torch.cat(generated,dim=1)[0],skip_special_tokens=True)
            pred=el(gen_text); true=test_data[i]['label']
            if pred: total+=1; correct+=1 if pred==true else 0
    return {'accuracy':correct/max(total,1),'correct':correct,'total':total}

# ═══════════════════ MAIN ═══════════════════
if __name__=='__main__':
    print("="*60)
    print("ProCyon v2 — Final: merged_all + Adapter 8t × 3 seeds")
    print("="*60)

    tok=AutoTokenizer.from_pretrained(LLM_PATH,trust_remote_code=True)
    train_data,test_data,ts,xs,tm,xm=load_data()
    train_ds,test_ds,coll=build_datasets(train_data,test_data,ts,xs,tm,xm,tok)
    device=DEVICE
    print(f"train={len(train_data)} test={len(test_data)} device={device} SL={SL}")

    results=[]

    for seed in SEEDS:
        t0=time.time()
        print(f"\n{'='*50}")
        print(f"  Seed={seed}")
        print(f"{'='*50}")

        torch.manual_seed(seed); np.random.seed(seed)

        # Build model
        enc=SimpleEmbEnc()
        for p in enc.parameters(): p.requires_grad=False
        enc.to(device)
        proj=AdapterProj8().to(device)
        llm_base=AutoModelForCausalLM.from_pretrained(LLM_PATH,dtype=torch.bfloat16,trust_remote_code=True)
        lc=LoraConfig(r=LR_R,lora_alpha=LR_A,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],lora_dropout=LR_D,bias='none',task_type=TaskType.CAUSAL_LM)
        lm=get_peft_model(llm_base,lc)
        mm=MM(lm,enc,proj,dropout_prob=DP).to(device)
        trn=sum(p.numel() for p in mm.parameters() if p.requires_grad)
        print(f"  Trainable: {trn:,}")

        # Train
        opt=torch.optim.AdamW([p for p in mm.parameters() if p.requires_grad],lr=LR)
        losses=[]
        for ep in range(NE):
            dl=DataLoader(train_ds,batch_size=BS,shuffle=True,collate_fn=coll)
            mm.train(); ep_loss=0; n=0; t1=time.time()
            n_batches=len(dl)
            for si,batch in enumerate(dl):
                out=mm(**batch); loss=out.loss/GA; loss.backward()
                ep_loss+=loss.item()*GA; n+=1
                if (si+1)%GA==0:
                    torch.nn.utils.clip_grad_norm_(mm.parameters(),1.0)
                    opt.step(); opt.zero_grad()
                if (si+1)%200==0:
                    print(f'    ep{ep+1} step{si+1}/{n_batches} loss={ep_loss/max(n,1):.4f} t={time.time()-t1:.0f}s')
            avg_loss=ep_loss/max(n,1)
            losses.append(avg_loss)
            print(f'  Epoch {ep+1}/{NE} loss={avg_loss:.4f} time={time.time()-t1:.0f}s')

        # Eval Enc+NL
        mm.set_dropout_prob(0.0)
        enc_nl=evaluate_mm(mm,test_data,xs,xm,'normal',tok,device)
        # Eval NL-only
        mm.set_dropout_prob(1.0)
        nl_only=evaluate_mm(mm,test_data,xs,xm,'dropout',tok,device)
        gap=enc_nl['accuracy']-nl_only['accuracy']

        print(f"  Enc+NL={enc_nl['accuracy']:.4f} ({enc_nl['correct']}/{enc_nl['total']})")
        print(f"  NL-only={nl_only['accuracy']:.4f} ({nl_only['correct']}/{nl_only['total']})")
        print(f"  Gap={gap:.4f} Total time={time.time()-t0:.0f}s")

        results.append({'seed':seed,'enc_nl':enc_nl,'nl_only':nl_only,'gap':gap,'losses':losses})

        del mm; gc.collect(); torch.cuda.empty_cache()

    # ── Summary ──
    print("\n"+"="*70)
    print("  FINAL RESULTS — merged_all + Adapter 8t × 3 seeds")
    print("="*70)
    accs=[r['enc_nl']['accuracy'] for r in results]
    nls=[r['nl_only']['accuracy'] for r in results]
    gaps=[r['gap'] for r in results]
    mean_acc=np.mean(accs); std_acc=np.std(accs)

    print(f"\n  {'Seed':<10} {'Enc+NL':>10} {'NL-only':>10} {'Gap':>10}")
    print(f"  {'-'*40}")
    for r in results:
        print(f"  {r['seed']:<10} {r['enc_nl']['accuracy']:>10.4f} {r['nl_only']['accuracy']:>10.4f} {r['gap']:>10.4f}")
    print(f"  {'-'*40}")
    print(f"  {'MEAN':<10} {mean_acc:>10.4f} ±{std_acc:.4f}")

    # Compare to baselines
    print(f"\n  vs Baselines (clean_2538):")
    print(f"    SimpleEmb + MLP:       0.9145 ±0.0157")
    print(f"    MGM + LLM (V5):        0.8862")
    print(f"    MGM + LLM (V6 d=0.5):  0.8800")
    print(f"    B2c (Adapter 8t):      0.9162 (single lucky seed)")
    print(f"    B2c × 3 seeds:         0.7066 ±0.1195 (unstable)")

    output={'experiment':'procyon_v2_final','dataset':'merged_all','n_train':len(train_data),
            'n_test':len(test_data),'results':results,'mean':float(mean_acc),'std':float(std_acc)}
    with open(f'{RESULT_DIR}/procyon_v2_final.json','w') as f:
        json.dump(output,f,indent=2,default=str)
    print(f"\nSaved to {RESULT_DIR}/procyon_v2_final.json")

    if std_acc<0.03 and mean_acc>0.85:
        print("\n✓ STABLE. ProCyon v2 architecture CONFIRMED.")
    else:
        print(f"\n✗ Unstable (std={std_acc:.4f}). Consider two-stage alignment.")
    print("DONE")
