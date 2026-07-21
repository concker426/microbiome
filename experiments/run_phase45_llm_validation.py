#!/usr/bin/env python3
"""
Phase 4.5: LLM Explanation Validation
- 3 prompt variants: Raw / SHAP / SHAP+Literature
- Metrics: hallucination rate, specificity, literature fidelity, input consistency
"""
import sys,os,csv,json,pickle,re
sys.path.insert(0,'/hd/liujx/microbiome_llm_project')
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR='/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis'
MODEL_PATH='/hd/liujx/microbiome_llm_project/models/qwen2-7b'
DATA_DIR='/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
os.makedirs(OUT_DIR,exist_ok=True)

DEVICE='cuda:0'; N_SAMPLES=50; MAX_NEW=300
SEED=42; torch.manual_seed(SEED); np.random.seed(SEED)

print("="*60)
print("Phase 4.5: LLM Explanation Validation")
print("="*60)

# ── Load model ──
print("Loading Qwen2-7B...")
tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,torch_dtype=torch.float16,
    device_map={'':DEVICE},trust_remote_code=True)
model.eval()
print(f"GPU mem: {torch.cuda.memory_allocated(0)/1024**3:.1f} GB")

# ── Load data ──
with open('/hd/liujx/microbiome_llm_project/data/qiita_ibd/combined_info.json') as f:
    GENUS_NAMES=json.load(f)['genus_names']

test_data=[]
with open(f'{DATA_DIR}/test_nl.jsonl') as f:
    for l in f: test_data.append(json.loads(l))

with open(f'{OUT_DIR}/../backbone/predictions.csv') as f:
    pred_data={r['sample_id']:r for r in csv.DictReader(f) if r['split']=='test'}

with open(f'{OUT_DIR}/shap_data_full.pkl','rb') as f:
    shap_by_id={}
    for s in pickle.load(f)['all_samples']:
        shap_by_id[s['sample_id']]={'label':s['label'],'importance':s['importance']}

# ── Build literature knowledge base ──
lit_kb={}
with open(f'{OUT_DIR}/literature_ground_truth.csv') as f:
    for r in csv.DictReader(f):
        g=r['Genus'].strip()
        lit_kb[g.lower()]={
            'direction':r['Direction'].strip(),
            'mechanism':r['Mechanism'].strip(),
            'evidence':r['Evidence_Level'].strip()
        }
print(f"Literature KB: {len(lit_kb)} genera")

# ── Select samples ──
test_h=[d for d in test_data if d['label']=='Healthy']
test_d=[d for d in test_data if d['label']=='Disease']
rng=np.random.RandomState(SEED)
sel=list(rng.choice(test_h,min(25,len(test_h)),replace=False))
sel+=list(rng.choice(test_d,min(25,len(test_d)),replace=False))
rng.shuffle(sel)
samples=sel[:N_SAMPLES]
print(f"Samples: {len(samples)} ({sum(1 for s in samples if s['label']=='Healthy')}H/{sum(1 for s in samples if s['label']=='Disease')}D)")

# ── Load test sequences ──
xs=np.load(f'{DATA_DIR}/test_genus_sequences.npy')
xm=np.load(f'{DATA_DIR}/test_genus_masks.npy')
test_idx_map={d['sample_id']:i for i,d in enumerate(test_data)}

def get_genera(sid):
    if sid not in test_idx_map: return []
    i=test_idx_map[sid]
    valid=xm[i].astype(bool)
    genera=[]
    for pos in range(len(xs[i])):
        if valid[pos] and xs[i][pos]>0:
            gid=int(xs[i][pos])
            gname=GENUS_NAMES[gid-1] if gid-1<len(GENUS_NAMES) else f'genus_{gid}'
            genera.append(gname)
    return genera

# ── Prompt builders ──
def build_prompt_a(genera):
    glist=', '.join(genera[:25])
    return f"""You are a gut microbiome analyst. A patient's gut microbiome contains these bacterial genera (ranked by abundance): {glist}.

Based on this genus composition, analyze whether this microbiome pattern is associated with Inflammatory Bowel Disease (IBD). Be specific about which genera suggest health or disease, and explain the biological reasoning.

Respond concisely in 3-5 sentences."""

def build_prompt_b(genera,pred_label,confidence,prob,shap_list):
    glist=', '.join(genera[:25])
    shap_str='\n'.join(f"  • {g['genus_name']}: {'INCREASED' if g['importance']>0 else 'DECREASED'} (importance={abs(g['importance']):.4f})" for g in shap_list[:15])

    return f"""You are a gut microbiome analyst. A machine learning classifier analyzed this patient's gut microbiome.

PATIENT DATA:
Genera (by abundance): {glist}

CLASSIFIER OUTPUT:
Prediction: {pred_label} (confidence: {confidence})
Disease probability: {prob:.3f}

IMPORTANT GENERA (SHAP feature importance):
{shap_str}

Based on these findings, explain why the classifier made this prediction. Reference specific genera and their known roles in gut health. Be specific and cite biological mechanisms.

Respond concisely in 4-6 sentences."""

def build_prompt_c(genera,pred_label,confidence,prob,shap_list):
    glist=', '.join(genera[:25])
    parts=[]
    for g in shap_list[:15]:
        gname=g['genus_name']; imp=g['importance']
        direction='INCREASED' if imp>0 else 'DECREASED'
        line=f"  • {gname}: {direction} (importance={abs(imp):.4f})"
        # Add literature context if available
        gn_lower=gname.strip().lower()
        if gn_lower in lit_kb:
            lk=lit_kb[gn_lower]
            line+=f" [LITERATURE: {lk['direction']} in IBD — {lk['mechanism']} ({lk['evidence']})]"
        parts.append(line)
    shap_str='\n'.join(parts)

    return f"""You are a gut microbiome analyst. A machine learning classifier analyzed this patient's gut microbiome.

PATIENT DATA:
Genera (by abundance): {glist}

CLASSIFIER OUTPUT:
Prediction: {pred_label} (confidence: {confidence})

IMPORTANT GENERA with literature context:
{shap_str}

Based on these findings AND the provided literature evidence, explain why the classifier made this prediction. Reference specific genera and literature-supported mechanisms. Discuss whether the model's findings are consistent with known IBD biology.

Respond concisely in 4-6 sentences."""

# ── Generate ──
@torch.no_grad()
def generate(prompt):
    messages=[{"role":"user","content":prompt}]
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=tokenizer(text,return_tensors="pt",truncation=True,max_length=2048).to(DEVICE)
    outputs=model.generate(
        **inputs,max_new_tokens=MAX_NEW,do_sample=True,
        temperature=0.7,top_p=0.9,pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()

results=[]
for i,d in enumerate(samples):
    sid=d['sample_id']; label=d['label']
    pred=pred_data.get(sid,{})
    prob=float(pred.get('prob_disease',0.5))
    pred_label='IBD' if prob>0.5 else 'HEALTHY'
    confidence=f'{max(prob,1-prob)*100:.1f}%'

    genera=get_genera(sid)
    if not genera: continue

    shap_top=[]
    if sid in shap_by_id:
        for g in shap_by_id[sid]['importance'][:15]:
            shap_top.append({'genus_name':g['genus_name'],'importance':g['importance']})

    print(f"\n[{i+1}/{len(samples)}] {sid} ({label}) pred={pred_label} conf={confidence}")

    try:
        # Variant A: Raw genus list
        prompt_a=build_prompt_a(genera)
        resp_a=generate(prompt_a)

        # Variant B: SHAP only
        prompt_b=build_prompt_b(genera,pred_label,confidence,prob,shap_top)
        resp_b=generate(prompt_b)

        # Variant C: SHAP + Literature
        prompt_c=build_prompt_c(genera,pred_label,confidence,prob,shap_top)
        resp_c=generate(prompt_c)

        results.append({
            'sample_id':sid,'ground_truth':label,'predicted':pred_label,
            'prob_disease':prob,'confidence':confidence,
            'genera':genera,'shap_top':shap_top,
            'prompt_a':prompt_a,'response_a':resp_a,
            'prompt_b':prompt_b,'response_b':resp_b,
            'prompt_c':prompt_c,'response_c':resp_c,
        })
        print(f"  A: {resp_a[:120]}...")
        print(f"  B: {resp_b[:120]}...")
        print(f"  C: {resp_c[:120]}...")
    except Exception as e:
        print(f"  ERROR: {e}")

# ── Evaluation ──
print(f"\n{'='*60}")
print("EVALUATION METRICS")
print(f"{'='*60}")

def count_genus_mentions(response,genus_set):
    """Count how many unique genus names from input are mentioned in response"""
    resp_lower=response.lower()
    mentioned=set()
    for g in genus_set:
        if g.lower() in resp_lower:
            mentioned.add(g.lower())
    return len(mentioned)

def count_hallucinated_genus(response,genus_set,all_genus_names):
    """Count genus names in response that are NOT in the input set"""
    resp_lower=response.lower()
    hallucinated=0
    # Check all known genus names
    genus_pattern=re.findall(r'[A-Z][a-z]+(?: [a-z]+)?',response)
    for g in genus_pattern:
        gl=g.lower()
        if gl not in genus_set and len(gl)>3:
            # Check if it's a known genus
            for known in all_genus_names:
                if known.lower()==gl:
                    hallucinated+=1
                    break
    return hallucinated

def count_specific_sentences(response):
    """Count sentences that contain specific biological mechanisms (not generic)"""
    sentences=re.split(r'[.!?]+',response)
    sentences=[s.strip() for s in sentences if len(s.strip())>20]
    if not sentences: return 0,0

    # Keywords indicating specificity
    specific_kw=['scfa','butyrate','inflammation','barrier','permeability',
        'immune','pathogen','dysbiosis','cytokine','mucosal','microbial',
        'decreased','increased','reduced','elevated','associated',
        'fermentation','metabolite','anti-inflammatory','pro-inflammatory']
    spec_count=0
    for s in sentences:
        for kw in specific_kw:
            if kw in s.lower():
                spec_count+=1
                break
    return spec_count,len(sentences)

def check_consistency(response,pred_label):
    """Check if response is consistent with the prediction"""
    resp_lower=response.lower()
    if pred_label=='IBD':
        disease_kw=['disease','ibd','inflammation','dysbiosis','crohn','colitis','altered','abnormal']
        healthy_kw_neg=['healthy','normal','balanced','homeostasis']
        disease_score=sum(1 for kw in disease_kw if kw in resp_lower)
        healthy_score=sum(1 for kw in healthy_kw_neg if kw in resp_lower)
        return 'consistent' if disease_score>=healthy_score else 'inconsistent'
    else:
        healthy_kw=['healthy','normal','balanced','homeostasis','commensal','beneficial']
        disease_kw_neg=['disease','ibd','inflammation','dysbiosis']
        healthy_score=sum(1 for kw in healthy_kw if kw in resp_lower)
        disease_score=sum(1 for kw in disease_kw_neg if kw in resp_lower)
        return 'consistent' if healthy_score>=disease_score else 'inconsistent'

# Build all genus names set for hallucination check
all_genus_set=set(g.lower() for g in GENUS_NAMES if len(g)>3)

# Compute metrics
metrics={'A':{'hallucination':[],'specificity':[],'genus_mentions':[],'consistency':[]},
         'B':{'hallucination':[],'specificity':[],'genus_mentions':[],'consistency':[]},
         'C':{'hallucination':[],'specificity':[],'genus_mentions':[],'consistency':[]}}

for r in results:
    genus_set=set(g.lower() for g in r['genera'])

    for variant,resp_key,resp in [('A','response_a',r['response_a']),
                                    ('B','response_b',r['response_b']),
                                    ('C','response_c',r['response_c'])]:
        # Hallucination
        n_hall=count_hallucinated_genus(resp,genus_set,all_genus_set)
        metrics[variant]['hallucination'].append(n_hall)

        # Genus mentions (how many input genera are referenced)
        n_mention=count_genus_mentions(resp,genus_set)
        metrics[variant]['genus_mentions'].append(n_mention)

        # Specificity
        n_spec,n_total=count_specific_sentences(resp)
        spec_ratio=n_spec/max(n_total,1)
        metrics[variant]['specificity'].append(spec_ratio)

        # Consistency
        cons=check_consistency(resp,r['predicted'])
        metrics[variant]['consistency'].append(cons)

# Print summary
for variant,label in [('A','Raw genus list'),('B','SHAP only'),('C','SHAP + Literature')]:
    m=metrics[variant]
    mean_hall=np.mean(m['hallucination'])
    mean_mention=np.mean(m['genus_mentions'])
    mean_spec=np.mean(m['specificity'])
    n_consistent=sum(1 for c in m['consistency'] if c=='consistent')
    n_total=len(m['consistency'])

    print(f"\n  {label}:")
    print(f"    Hallucinated genera:  {mean_hall:.1f}/response")
    print(f"    Input genera mentioned: {mean_mention:.1f}/response")
    print(f"    Specificity ratio:    {mean_spec:.3f} (higher = more specific)")
    print(f"    Prediction consistent: {n_consistent}/{n_total} ({n_consistent/n_total*100:.0f}%)")

# ── Save results ──
# Serialize metrics properly
metrics_serializable={}
for v,m in metrics.items():
    metrics_serializable[v]={}
    for k,vv in m.items():
        if isinstance(vv,list) and vv and isinstance(vv[0],(int,float,np.floating)):
            metrics_serializable[v][k]=float(np.mean(vv))
        elif isinstance(vv,list) and vv and isinstance(vv[0],str):
            metrics_serializable[v][k]=sum(1 for x in vv if x=='consistent')
        else:
            metrics_serializable[v][k]=vv

with open(f'{OUT_DIR}/phase45_validation.json','w') as f:
    json.dump({'results':results,'metrics':metrics_serializable},f,indent=2,default=str)

# Human-readable report
with open(f'{OUT_DIR}/phase45_human_review.txt','w') as f:
    for i,r in enumerate(results):
        f.write(f"{'='*70}\nSample {i+1}: {r['sample_id']}\n")
        f.write(f"Truth: {r['ground_truth']} | Pred: {r['predicted']} (p={r['prob_disease']:.3f}, conf={r['confidence']})\n")
        f.write(f"Genera: {', '.join(r['genera'][:15])}...\n")
        f.write(f"\n─── A: RAW ───\n{r['response_a']}\n")
        f.write(f"\n─── B: SHAP ───\n{r['response_b']}\n")
        f.write(f"\n─── C: SHAP+LIT ───\n{r['response_c']}\n\n")

print(f"\nSaved: {OUT_DIR}/phase45_validation.json")
print(f"Saved: {OUT_DIR}/phase45_human_review.txt")
print("\nPHASE 4.5 DONE")
