#!/usr/bin/env python3
"""Phase 4 Pilot: LLM explanation — 50 test samples, Raw vs SHAP+LLM"""
import sys,os,csv,json
sys.path.insert(0,'/hd/liujx/microbiome_llm_project')
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR='/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis'
MODEL_PATH='/hd/liujx/microbiome_llm_project/models/qwen2-7b'
DATA_DIR='/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
os.makedirs(OUT_DIR,exist_ok=True)

DEVICE='cuda:0'
N_SAMPLES=50
MAX_NEW=256

print("="*60)
print("Phase 4 Pilot: LLM Explanation (50 samples)")
print("="*60)

# ── Load model ──
print("Loading Qwen2-7B...")
tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,torch_dtype=torch.float16,
    device_map={'':DEVICE},trust_remote_code=True
)
model.eval()
print(f"Model loaded on {DEVICE}. GPU mem: {torch.cuda.memory_allocated(0)/1024**3:.1f} GB")

# ── Load test data ──
test_data=[]
with open(f'{DATA_DIR}/test_nl.jsonl') as f:
    for l in f: test_data.append(json.loads(l))

# Load predictions
with open(f'{OUT_DIR}/../backbone/predictions.csv') as f:
    pred_data={r['sample_id']:r for r in csv.DictReader(f) if r['split']=='test'}

# Load SHAP patient data
import pickle
with open(f'{OUT_DIR}/shap_data_full.pkl','rb') as f:
    shap_data=pickle.load(f)
shap_by_id={}
for s in shap_data['all_samples']:
    shap_by_id[s['sample_id']]={'label':s['label'],'importance':s['importance']}

# Load genus names
with open('/hd/liujx/microbiome_llm_project/data/qiita_ibd/combined_info.json') as f:
    GENUS_NAMES=json.load(f)['genus_names']

# ── Select 50 test samples (balanced) ──
test_healthy=[d for d in test_data if d['label']=='Healthy']
test_disease=[d for d in test_data if d['label']=='Disease']
rng=np.random.RandomState(42)
sel_healthy=rng.choice(test_healthy,min(25,len(test_healthy)),replace=False)
sel_disease=rng.choice(test_disease,min(25,len(test_disease)),replace=False)
selected_samples=list(sel_healthy)+list(sel_disease)
rng.shuffle(selected_samples)
selected_samples=selected_samples[:N_SAMPLES]

print(f"Selected {len(selected_samples)} samples ({sum(1 for d in selected_samples if d['label']=='Healthy')}H/{sum(1 for d in selected_samples if d['label']=='Disease')}D)")

# ── Build prompts ──
@torch.no_grad()
def generate(prompt):
    messages=[{"role":"user","content":prompt}]
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=tokenizer(text,return_tensors="pt").to(DEVICE)
    outputs=model.generate(
        **inputs,max_new_tokens=MAX_NEW,do_sample=True,
        temperature=0.7,top_p=0.9,pad_token_id=tokenizer.eos_token_id
    )
    response=tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],skip_special_tokens=True)
    return response.strip()

def decode_genus_ids(genus_ids,genus_mask):
    """Convert genus IDs to readable list"""
    genera=[]
    valid=genus_mask.astype(bool)
    for pos in range(len(genus_ids)):
        if valid[pos] and genus_ids[pos]>0:
            gid=int(genus_ids[pos])
            gname=GENUS_NAMES[gid-1] if gid-1<len(GENUS_NAMES) else f'genus_{gid}'
            genera.append(gname)
    return genera

results=[]

for i,d in enumerate(selected_samples):
    sid=d['sample_id']
    label=d['label']
    pred=pred_data.get(sid,{})
    prob=float(pred.get('prob_disease',0.5))
    pred_label='Disease' if prob>0.5 else 'Healthy'

    # Get genus list
    # Find this sample in test data
    test_idx=None
    for j,td in enumerate(test_data):
        if td['sample_id']==sid:
            test_idx=j; break
    if test_idx is None: continue

    xs=np.load(f'{DATA_DIR}/test_genus_sequences.npy')
    xm=np.load(f'{DATA_DIR}/test_genus_masks.npy')
    genera=decode_genus_ids(xs[test_idx],xm[test_idx])
    genera_str=', '.join(genera[:20])  # top 20 most abundant

    # Get SHAP top 20
    shap_top20=[]
    if sid in shap_by_id:
        for g in shap_by_id[sid]['importance'][:20]:
            direction='↑' if g['importance']>0 else '↓'
            shap_top20.append(f"{g['genus_name']} ({direction}, importance={abs(g['importance']):.4f})")

    # ── Prompt A: Raw genus list ──
    prompt_a=f"""You are a gut microbiome analyst. A patient's gut microbiome sample contains the following bacterial genera (listed by abundance):

{genera_str}

Please explain whether this microbiome pattern is associated with Inflammatory Bowel Disease (IBD). What biological mechanisms might be involved?"""

    # ── Prompt B: SHAP-guided ──
    shap_str='\n'.join(f"  {i+1}. {g}" for i,g in enumerate(shap_top20)) if shap_top20 else '(no SHAP data available)'

    prompt_b=f"""You are a gut microbiome analyst. A machine learning classifier has analyzed a patient's gut microbiome and predicted: {pred_label} (probability: {prob:.2f}).

The classifier identified these genera as most important for the prediction (SHAP feature importance):

{shap_str}

Please explain the possible biological mechanisms linking these specific genera to IBD. What does the pattern of these genera suggest about the patient's gut health?"""

    # Generate
    try:
        print(f"\n  [{i+1}/{len(selected_samples)}] {sid} ({label})...")
        resp_a=generate(prompt_a)
        resp_b=generate(prompt_b)

        results.append({
            'sample_id':sid,
            'ground_truth':label,
            'predicted':pred_label,
            'prob_disease':prob,
            'genera':genera_str,
            'shap_top20':shap_top20,
            'prompt_a':prompt_a,
            'response_a':resp_a,
            'prompt_b':prompt_b,
            'response_b':resp_b
        })

        print(f"    A (raw):  {resp_a[:150]}...")
        print(f"    B (SHAP): {resp_b[:150]}...")

    except Exception as e:
        print(f"    ERROR: {e}")
        continue

# ── Save ──
with open(f'{OUT_DIR}/phase4_llm_pilot.json','w') as f:
    json.dump(results,f,indent=2,default=str)
print(f"\nSaved: {OUT_DIR}/phase4_llm_pilot.json ({len(results)} samples)")

# ── Quick evaluation ──
print(f"\n{'='*60}")
print("Preliminary Evaluation")
print(f"{'='*60}")

hallucination_a=0; hallucination_b=0
total_a=0; total_b=0
for r in results:
    genera_set=set(g.strip().lower() for g in r['genera'].split(','))

    # Check response A for mentioned genera not in input
    resp_a_lower=r['response_a'].lower()
    for g in genera_set:
        if g and g in resp_a_lower:
            total_a+=1
    # Rough hallucination check: mentions of "studies show", "known to", etc. without genus name
    # Simple check: count how many genus names from response appear in input
    resp_a_words=set(resp_a_lower.replace(',',' ').replace('.',' ').split())
    unknown_a=0
    for w in resp_a_words:
        if len(w)>3 and w not in genera_set and any(w in g for g in genera_set):
            pass  # fuzzy match
    total_a+=1

    # Similar for B
    resp_b_lower=r['response_b'].lower()

print(f"  (Detailed evaluation requires human review)")
print(f"  Saved responses for manual inspection")

# Save readable format
with open(f'{OUT_DIR}/phase4_human_review.txt','w') as f:
    for i,r in enumerate(results):
        f.write(f"{'='*70}\n")
        f.write(f"Sample {i+1}: {r['sample_id']}\n")
        f.write(f"Ground Truth: {r['ground_truth']} | Predicted: {r['predicted']} (p={r['prob_disease']:.3f})\n")
        f.write(f"\n--- RAW LLM (Baseline A) ---\n{r['response_a']}\n")
        f.write(f"\n--- SHAP+LLM (Method B) ---\n{r['response_b']}\n")
        f.write(f"\nTop SHAP genera: {r['shap_top20'][:5] if r['shap_top20'] else 'N/A'}\n\n")
print(f"Saved: {OUT_DIR}/phase4_human_review.txt")

print("\nPHASE 4 PILOT DONE")
