#!/usr/bin/env python3
"""B: Two-stage training on clean_2538. GPU 1.
Stage 1: freeze Qwen LoRA, train only Adapter (learn embedding alignment)
Stage 2: unfreeze LoRA, joint training with dropout
"""
import os,sys,json,time,gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
os.environ["TORCH_FLASH_ATTN_ENABLED"]="0"
import transformers.utils.import_utils as _iu;_iu.is_flash_attn_2_available=lambda:False
import transformers.utils as _utils;_utils.is_flash_attn_2_available=lambda:False
import accelerate.utils.imports as _ai;_ai.is_deepspeed_available=lambda:False
import accelerate.utils.other as _ao;_ao.is_deepspeed_available=lambda:False
sys.path.insert(0,'/hd/liujx/microbiome_llm_project')
import numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig,get_peft_model,TaskType
from run_v6_merged import el,LABELS

DATA_DIR='/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
LLM_PATH='/hd/gcr/hf_models/Qwen2.5-7B-Instruct'
RESULT_DIR='/hd/liujx/microbiome_llm_project/experiments/results'
os.makedirs(RESULT_DIR,exist_ok=True)
V=1226;E=768;LH=3584;SL=86;NT=8;PS=0.1;LR_R=16;LR_A=32;LR_D=0.03
BS=1;GA=8;LR=3e-5;ML=1024;DW=1.5;DP=0.5
DEVICE='cuda:2'
SEEDS=[42,123,456]
STAGE1_EP=3;STAGE2_EP=4

class SimpleEmbEnc(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(V, E, padding_idx=0)
    def forward(s, ids, mask=None):
        x = s.emb(ids)
        mf = mask.float().unsqueeze(-1) if mask is not None else torch.ones_like(x[..., :1])
        return (x * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1)

class AdapterProj(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln = nn.LayerNorm(E)
        s.fc1 = nn.Linear(E, 2048)
        s.fc2 = nn.Linear(2048, LH * NT)
        s.sc = nn.Parameter(torch.ones(1) * PS)
        s.nt = NT
    def forward(s, x):
        x = s.ln(x)
        x = F.gelu(s.fc1(x))
        x = s.fc2(x)
        return x.view(-1, s.nt, LH) * s.sc

class MM(nn.Module):
    def __init__(s, llm, enc, proj, dropout_prob=DP):
        super().__init__()
        s.llm = llm
        s.enc = enc
        s.proj = proj
        s.config = llm.config
        s.nmt = proj.nt
        s.dropout_prob = dropout_prob
    def set_dropout_prob(s, p):
        s.dropout_prob = p
    def forward(s,input_ids=None,attention_mask=None,labels=None,genus_ids=None,genus_mask=None,sample_weights=None,**kw):
        B=genus_ids.shape[0];d=next(s.parameters()).device;nt=s.nmt
        input_ids=input_ids.to(d);attention_mask=attention_mask.to(d)
        if labels is not None:labels=labels.to(d)
        genus_ids=genus_ids.to(d);genus_mask=genus_mask.to(d)
        S=input_ids.shape[1]
        me=s.enc(genus_ids,genus_mask).to(next(s.proj.parameters()).dtype)
        mt=s.proj(me)
        if s.training and s.dropout_prob>0:
            mask=torch.rand(B,1,1,device=mt.device)>s.dropout_prob;mt=mt*mask.float()
        te=s.llm.base_model.model.model.embed_tokens(input_ids);mt=mt.to(te.dtype)
        ce=torch.cat([mt,te],dim=1)
        if labels is not None:
            nl=torch.full((B,S+nt),-100,device=labels.device,dtype=labels.dtype);nl[:,nt:]=labels
        else:nl=None
        if attention_mask is not None:
            nm=torch.ones(B,S+nt,device=attention_mask.device,dtype=attention_mask.dtype);nm[:,nt:]=attention_mask
        else:nm=None
        pid=torch.arange(ce.shape[1],dtype=torch.long,device=ce.device).unsqueeze(0)
        o=s.llm(inputs_embeds=ce,attention_mask=nm,position_ids=pid,labels=nl,**kw)
        if sample_weights is not None and o.loss is not None:
            sw=sample_weights.to(d);lo=o.logits[:,:-1,:].contiguous()
            sl2=nl[:,1:].contiguous() if nl is not None else None
            tl=F.cross_entropy(lo.view(-1,lo.size(-1)),sl2.view(-1),reduction='none').view(B,-1)
            vm=(sl2!=-100).float();tl=tl*vm
            sloss=tl.sum(dim=1)/vm.sum(dim=1).clamp(min=1);o.loss=(sloss*sw).sum()/sw.sum()
        return o

def load_data():
    td,ed=[],[]
    with open(f'{DATA_DIR}/train_nl.jsonl') as f:
        for l in f:td.append(json.loads(l))
    with open(f'{DATA_DIR}/test_nl.jsonl') as f:
        for l in f:ed.append(json.loads(l))
    ts=np.load(f'{DATA_DIR}/train_genus_sequences.npy');xs=np.load(f'{DATA_DIR}/test_genus_sequences.npy')
    tm=np.load(f'{DATA_DIR}/train_genus_masks.npy');xm=np.load(f'{DATA_DIR}/test_genus_masks.npy')
    return td,ed,ts,xs,tm,xm

def build_ds(td,ed,ts,xs,tm,xm,tok):
    class DS(Dataset):
        def __init__(s,data,seqs,mks):
            s.data=data;s.seqs=seqs;s.mks=mks;s.enc=[];s.sw=[]
            for it in data:
                msgs=it['messages']
                fi=tok.apply_chat_template(msgs,tokenize=True,max_length=ML,truncation=True,add_generation_prompt=False).input_ids
                pi=tok.apply_chat_template([msgs[0]],tokenize=True,max_length=ML,truncation=True,add_generation_prompt=True).input_ids
                ul=len(pi);lb=[-100]*len(fi)
                for j in range(ul,len(fi)):lb[j]=fi[j]
                s.enc.append({'ids':fi,'lb':lb});s.sw.append(DW if it.get('label','Healthy')=='Disease' else 1.0)
        def __len__(s):return len(s.data)
        def __getitem__(s,i):
            e=s.enc[i];sq=s.seqs[i].astype(np.int64);mk=s.mks[i]
            return {'input_ids':e['ids'],'attention_mask':[1]*len(e['ids']),'labels':e['lb'],'genus_ids':sq,'genus_mask':mk,'sample_weights':s.sw[i]}
    class Coll:
        def __init__(s,tok):s.tok=tok;s.pid=tok.pad_token_id or 0
        def __call__(s,b):
            ids=[x['input_ids'] for x in b];am=[x['attention_mask'] for x in b];lb=[x['labels'] for x in b]
            gi=[x['genus_ids'] for x in b];gm=[x['genus_mask'] for x in b];sw=[x['sample_weights'] for x in b]
            ml2=min(max(len(i) for i in ids),ML);pi2,pm,pl=[],[],[]
            for i in range(len(ids)):
                d=ids[i][:ml2];m=am[i][:ml2];l=lb[i][:ml2];p=ml2-len(d)
                pi2.append(d+[s.pid]*p if p>0 else d);pm.append(m+[0]*p if p>0 else m);pl.append(l+[-100]*p if p>0 else l)
            tg=[g[:SL] for g in gi];tmm=[m[:SL] for m in gm];mgl=max(len(g) for g in tg);pg,pm2=[],[]
            for i in range(len(tg)):
                g=tg[i];m=tmm[i];p=mgl-len(g)
                pg.append(np.pad(g,(0,p),constant_values=0) if p>0 else g);pm2.append(np.pad(m,(0,p),constant_values=False) if p>0 else m)
            return {'input_ids':torch.tensor(pi2,dtype=torch.long),'attention_mask':torch.tensor(pm,dtype=torch.long),
                    'labels':torch.tensor(pl,dtype=torch.long),'genus_ids':torch.tensor(np.array(pg),dtype=torch.long),
                    'genus_mask':torch.tensor(np.array(pm2),dtype=torch.bool),'sample_weights':torch.tensor(sw,dtype=torch.float32)}
    return DS(td,ts,tm),DS(ed,xs,xm),Coll(tok)

@torch.no_grad()
def evaluate_mm(mm,test_data,xs,xm,mode,tok,device,max_tok=128):
    mm.eval();correct=0;total=0
    for i in range(len(test_data)):
        sq=xs[i];mk=xm[i]
        gi=torch.from_numpy(np.asarray(sq).astype(np.int64)).long().unsqueeze(0).to(device)
        gm=torch.from_numpy(np.asarray(mk)).bool().unsqueeze(0).to(device)
        me=mm.enc(gi,gm).to(next(mm.proj.parameters()).dtype);mt=mm.proj(me)
        if mode=='dropout':mt=mt*0.0
        msgs=test_data[i]['messages']
        prompt=tok.apply_chat_template([msgs[0]],tokenize=False,add_generation_prompt=True)
        pi=tok(prompt,return_tensors='pt',truncation=True,max_length=ML).to(device)
        te=mm.llm.base_model.model.model.embed_tokens(pi['input_ids']);mt=mt.to(te.dtype)
        ce=torch.cat([mt,te],dim=1)
        sl=ce.shape[1];pid=torch.arange(0,sl,dtype=torch.long,device=device).unsqueeze(0)
        o=mm.llm(inputs_embeds=ce,position_ids=pid,use_cache=True)
        nt2=torch.argmax(o.logits[:,-1,:],dim=-1,keepdim=True);generated=[nt2];cur_len=sl
        for _ in range(max_tok):
            pos=torch.full((1,1),cur_len,dtype=torch.long,device=device)
            out=mm.llm(input_ids=nt2,position_ids=pos,past_key_values=o.past_key_values,use_cache=True)
            nt2=torch.argmax(out.logits[:,-1,:],dim=-1,keepdim=True)
            if nt2.item()==tok.eos_token_id:break
            generated.append(nt2);cur_len+=1;o.past_key_values=out.past_key_values
        gen_text=tok.decode(torch.cat(generated,dim=1)[0],skip_special_tokens=True)
        pred=el(gen_text);true=test_data[i]['label']
        if pred:total+=1;correct+=1 if pred==true else 0
    return {'accuracy':correct/max(total,1),'correct':correct,'total':total}

if __name__=='__main__':
    print("="*60);print("B: Two-Stage Training (clean_2538)");print("="*60)
    tok=AutoTokenizer.from_pretrained(LLM_PATH,trust_remote_code=True)
    td,ed,ts,xs,tm,xm=load_data();train_ds,_,coll=build_ds(td,ed,ts,xs,tm,xm,tok)
    print(f"train={len(td)} test={len(ed)} device={DEVICE}")
    results=[]
    for seed in SEEDS:
        t0=time.time();print(f"\n{'='*40}\nSeed={seed}\n{'='*40}")
        torch.manual_seed(seed);np.random.seed(seed)
        enc=SimpleEmbEnc()
        for p in enc.parameters():p.requires_grad=False
        enc.to(DEVICE);proj=AdapterProj().to(DEVICE)
        llm=AutoModelForCausalLM.from_pretrained(LLM_PATH,dtype=torch.bfloat16,trust_remote_code=True,device_map={'':DEVICE})
        lc=LoraConfig(r=LR_R,lora_alpha=LR_A,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],lora_dropout=LR_D,bias='none',task_type=TaskType.CAUSAL_LM)
        lm=get_peft_model(llm,lc);mm=MM(lm,enc,proj).to(DEVICE)

        # ── Stage 1: freeze LoRA, train Adapter only ──
        print("  Stage 1: Adapter only (Qwen frozen)")
        for n,p in lm.named_parameters():p.requires_grad=False
        for p in proj.parameters():p.requires_grad=True
        opt1=torch.optim.AdamW([p for p in proj.parameters() if p.requires_grad],lr=LR*10)  # higher LR for adapter-only
        mm.set_dropout_prob(0.0)  # no dropout during adapter-only phase
        for ep in range(STAGE1_EP):
            dl=DataLoader(train_ds,batch_size=BS,shuffle=True,collate_fn=coll);mm.train();n=0;ep_loss=0;t1=time.time()
            for si,batch in enumerate(dl):
                out=mm(**batch);loss=out.loss/GA;loss.backward();ep_loss+=loss.item()*GA;n+=1
                if (si+1)%GA==0:torch.nn.utils.clip_grad_norm_(mm.parameters(),1.0);opt1.step();opt1.zero_grad()
            print(f'    S1 Ep{ep+1}/{STAGE1_EP} loss={ep_loss/max(n,1):.4f}')

        # ── Stage 2: unfreeze LoRA, joint training ──
        print("  Stage 2: Joint training (Adapter + LoRA)")
        del opt1; gc.collect(); torch.cuda.empty_cache()
        mm.llm.gradient_checkpointing_enable()
        for n,p in lm.named_parameters():p.requires_grad=True
        opt2=torch.optim.AdamW([p for p in mm.parameters() if p.requires_grad],lr=LR)
        mm.set_dropout_prob(DP)
        for ep in range(STAGE2_EP):
            dl=DataLoader(train_ds,batch_size=BS,shuffle=True,collate_fn=coll);mm.train();n=0;ep_loss=0;t1=time.time()
            for si,batch in enumerate(dl):
                out=mm(**batch);loss=out.loss/GA;loss.backward();ep_loss+=loss.item()*GA;n+=1
                if (si+1)%GA==0:torch.nn.utils.clip_grad_norm_(mm.parameters(),1.0);opt2.step();opt2.zero_grad()
            print(f'    S2 Ep{ep+1}/{STAGE2_EP} loss={ep_loss/max(n,1):.4f}')

        mm.set_dropout_prob(0.0);enc_nl=evaluate_mm(mm,ed,xs,xm,'normal',tok,DEVICE)
        mm.set_dropout_prob(1.0);nl_only=evaluate_mm(mm,ed,xs,xm,'dropout',tok,DEVICE)
        gap=enc_nl['accuracy']-nl_only['accuracy']
        print(f"  Enc+NL={enc_nl['accuracy']:.4f} NL-only={nl_only['accuracy']:.4f} Gap={gap:.4f} Time={time.time()-t0:.0f}s")
        results.append({'seed':seed,'enc_nl':enc_nl,'nl_only':nl_only,'gap':gap,'method':'two_stage'})
        del mm;gc.collect();torch.cuda.empty_cache()
    accs=[r['enc_nl']['accuracy'] for r in results]
    print(f"\nMEAN: {np.mean(accs):.4f} ±{np.std(accs):.4f}")
    with open(f'{RESULT_DIR}/two_stage.json','w') as f:json.dump(results,f,indent=2,default=str)
    print("DONE")
