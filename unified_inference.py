#!/usr/bin/env python3
"""
Unified inference pipeline: cascade Healthy/Disease binary → CD/UC subtype.

Loads:
  - procyon_nl_7b           (binary classifier)
  - procyon_subtype_7b      (CD/UC sub-typer; only invoked on Disease)
  - sample_retriever (TF-IDF index over training samples)

Usage:
    pipeline = MicrobiomePipeline().load()
    result = pipeline.infer(genus_ids, genus_mask, top_genera_for_retrieval)

CLI:
    python3 unified_inference.py --sample-id 10317.000065620
    python3 unified_inference.py --random
"""
import os, json, argparse, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["TORCH_FLASH_ATTN_ENABLED"] = "0"

import fix_flash_attn  # noqa: F401
import accelerate.utils.imports as _acc_imports
_acc_imports.is_deepspeed_available = lambda: False
import accelerate.utils.other as _acc_other
_acc_other.is_deepspeed_available = lambda: False

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from mgm_encoder import MGMEncoder
from run_microbiome_nl_7b import (
    ProjectionLayer as NL_Projection, MultimodalNLModel,
    extract_label as nl_extract_label,
    MAX_SEQ_LEN, VOCAB_SIZE, EMBED_DIM, MGM_LAYERS, MGM_HEADS,
    MGM_FFN_DIM, MGM_DROPOUT, MODEL_PATH, MAX_LENGTH,
)
from run_microbiome_subtype_7b import (
    ProjectionLayer as SUB_Projection, MultimodalSubtypeModel,
    extract_label as subtype_extract_label,
    SUBTYPE_PROMPT,
)
from sample_retriever import SampleRetriever

BASE = "/hd/liujx/microbiome_llm_project"
NL_DIR = os.path.join(BASE, "saved_models/procyon_nl_7b")
SUBTYPE_DIR = os.path.join(BASE, "saved_models/procyon_subtype_7b")
TRAIN_SET = os.path.join(BASE, "data/agp_ftp_processed/train_set.jsonl")
TEST_SET = os.path.join(BASE, "data/agp_ftp_processed/test_set.jsonl")
TRAIN_SEQ = os.path.join(BASE, "data/agp_ftp_processed/train_genus_sequences.npy")
TRAIN_MSK = os.path.join(BASE, "data/agp_ftp_processed/train_genus_masks.npy")
TEST_SEQ = os.path.join(BASE, "data/agp_ftp_processed/test_genus_sequences.npy")
TEST_MSK = os.path.join(BASE, "data/agp_ftp_processed/test_genus_masks.npy")
TEST_VEC = os.path.join(BASE, "data/agp_ftp_processed/test_set_vectors.npy")
TRAIN_VEC = os.path.join(BASE, "data/agp_ftp_processed/train_set_vectors.npy")
GENUS_NAMES = os.path.join(BASE, "data/agp_ftp_processed/genus_names.npy")


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(line) for line in f]


def _load_multimodal(adapter_dir, projection_cls):
    """Load LLM+LoRA + MGMEncoder + projection from a saved checkpoint dir."""
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    lora = PeftModel.from_pretrained(base, adapter_dir)
    lora.config.use_cache = True
    encoder = MGMEncoder(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
        n_layers=MGM_LAYERS, n_heads=MGM_HEADS, ffn_dim=MGM_FFN_DIM,
        max_seq_len=MAX_SEQ_LEN, dropout=MGM_DROPOUT,
    )
    projection = projection_cls()
    ck = torch.load(os.path.join(adapter_dir, "multimodal_components.pt"),
                    map_location="cpu")
    encoder.load_state_dict(ck["encoder_state_dict"])
    projection.load_state_dict(ck["projection_state_dict"])
    encoder.to(lora.device, dtype=torch.bfloat16)
    projection.to(lora.device, dtype=torch.bfloat16)
    return lora, encoder, projection


@torch.no_grad()
def _generate(model, tokenizer, prompt_text, genus_ids, genus_mask,
              device, max_new_tokens=128):
    micro = model.encoder(genus_ids, genus_mask)
    micro = micro.to(model.projection.proj.weight.dtype)
    mt = model.projection(micro).unsqueeze(1)

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False, add_generation_prompt=True,
    )
    pin = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=MAX_LENGTH).to(device)
    te = model.llm.base_model.model.model.embed_tokens(pin["input_ids"])
    mt = mt.to(te.dtype)
    comb = torch.cat([mt, te], dim=1)
    L = comb.shape[1]
    pos = torch.arange(0, L, dtype=torch.long, device=device).unsqueeze(0)
    out = model.llm(inputs_embeds=comb, position_ids=pos, use_cache=True)
    nt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    gen_ids = [nt]
    cur = L
    for _ in range(max_new_tokens):
        pid = torch.full((1, 1), cur, dtype=torch.long, device=device)
        o = model.llm(input_ids=nt, position_ids=pid,
                      past_key_values=out.past_key_values, use_cache=True)
        nt = torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)
        if nt.item() == tokenizer.eos_token_id:
            break
        gen_ids.append(nt)
        cur += 1
        out.past_key_values = o.past_key_values
    gen_ids = torch.cat(gen_ids, dim=1)
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()


def _build_genus_str(vec, genus_names, top_n=15):
    idx = np.argsort(-vec)[:top_n]
    return "，".join(f"{genus_names[i]} ({vec[i]:.2f}%)" for i in idx if vec[i] > 0)


class MicrobiomePipeline:
    def __init__(self):
        self.nl_model = None
        self.subtype_model = None
        self.tokenizer = None
        self.retriever = None
        self.genus_names = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self, with_retrieval=True):
        print("Loading NL binary model...")
        nl_lora, nl_enc, nl_proj = _load_multimodal(NL_DIR, NL_Projection)
        self.nl_model = MultimodalNLModel(nl_lora, nl_enc, nl_proj)
        self.nl_model.eval()

        print("Loading Subtype model...")
        st_lora, st_enc, st_proj = _load_multimodal(SUBTYPE_DIR, SUB_Projection)
        self.subtype_model = MultimodalSubtypeModel(st_lora, st_enc, st_proj)
        self.subtype_model.eval()

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(NL_DIR)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if with_retrieval:
            print("Loading retrieval index...")
            self.retriever = SampleRetriever().load()

        self.genus_names = np.load(GENUS_NAMES, allow_pickle=True)

        print("✅ Pipeline ready.\n")
        return self

    def infer(self, genus_ids_np, genus_mask_np, sample_vec=None,
              with_retrieval=True, k_retrieve=3):
        """Run cascade: NL binary → (if Disease) Subtype → optionally retrieval.
        Args:
          genus_ids_np: 1-D numpy array of token ids
          genus_mask_np: 1-D numpy array of mask (bool)
          sample_vec: 1222-D abundance vector (optional, used for retrieval doc)
        Returns:
          dict with predicted_label, predicted_subtype (if any), generated_text,
          retrieved_samples (list)
        """
        device = self.device
        gid = torch.from_numpy(np.asarray(genus_ids_np[:MAX_SEQ_LEN]).astype(np.int64)).long().unsqueeze(0).to(device)
        gmk = torch.from_numpy(np.asarray(genus_mask_np[:MAX_SEQ_LEN])).bool().unsqueeze(0).to(device)

        result = {}

        # Stage 1: binary diagnosis
        genus_str = _build_genus_str(sample_vec, self.genus_names) if sample_vec is not None else "（菌属向量）"
        binary_prompt = (
            "你是一位专业的肠道微生物分析师。请分析样本的菌群数据。\n\n"
            f"【主要菌属构成】: {genus_str}\n\n"
            "请判断该样本的健康状态（Healthy 或 Disease），并简要说明理由。"
        )
        nl_text = _generate(self.nl_model, self.tokenizer, binary_prompt, gid, gmk, device)
        nl_label = nl_extract_label(nl_text)
        result["stage1_binary"] = {"label": nl_label, "text": nl_text}

        # Stage 2: subtype (only if Disease)
        if nl_label == "Disease":
            sub_prompt = SUBTYPE_PROMPT.format(genus_str=genus_str)
            sub_text = _generate(self.subtype_model, self.tokenizer, sub_prompt, gid, gmk, device)
            sub_label = subtype_extract_label(sub_text)
            result["stage2_subtype"] = {"label": sub_label, "text": sub_text}
        else:
            result["stage2_subtype"] = None

        # Stage 3: retrieval (text-based, uses query derived from result)
        if with_retrieval and self.retriever is not None and sample_vec is not None:
            top_idx = np.argsort(-sample_vec)[:5]
            top_part = " ".join(str(self.genus_names[i]) for i in top_idx if sample_vec[i] > 0)
            query_parts = [nl_label or "", top_part]
            if result.get("stage2_subtype") and result["stage2_subtype"].get("label"):
                query_parts.append(result["stage2_subtype"]["label"])
            query = " ".join(p for p in query_parts if p)
            hits = self.retriever.search(query, k=k_retrieve)
            result["stage3_retrieval"] = {"query": query, "hits": hits}
        else:
            result["stage3_retrieval"] = None

        return result


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sample-id", help="evaluate a specific sample by sample_id (search train+test)")
    g.add_argument("--random", action="store_true", help="pick a random Disease sample")
    g.add_argument("--first-disease", action="store_true", help="run on first Disease sample in test")
    ap.add_argument("--no-retrieval", action="store_true")
    args = ap.parse_args()

    test_items = load_jsonl(TEST_SET)
    train_items = load_jsonl(TRAIN_SET)
    test_seqs = np.load(TEST_SEQ).astype(np.int64)
    test_masks = np.load(TEST_MSK)
    train_seqs = np.load(TRAIN_SEQ).astype(np.int64)
    train_masks = np.load(TRAIN_MSK)
    test_vecs = np.load(TEST_VEC)
    train_vecs = np.load(TRAIN_VEC)

    chosen_split, chosen_idx = None, None
    if args.sample_id:
        for i, it in enumerate(test_items):
            if it.get("sample_id") == args.sample_id:
                chosen_split, chosen_idx = "test", i
                break
        if chosen_idx is None:
            for i, it in enumerate(train_items):
                if it.get("sample_id") == args.sample_id:
                    chosen_split, chosen_idx = "train", i
                    break
        if chosen_idx is None:
            raise SystemExit(f"sample_id {args.sample_id} not found")
    elif args.random:
        disease = [i for i, it in enumerate(test_items) if it.get("label") == "Disease"]
        chosen_split, chosen_idx = "test", random.choice(disease)
    else:
        # default: first disease in test
        for i, it in enumerate(test_items):
            if it.get("label") == "Disease":
                chosen_split, chosen_idx = "test", i
                break

    if chosen_split == "test":
        item, seq, msk, vec = test_items[chosen_idx], test_seqs[chosen_idx], test_masks[chosen_idx], test_vecs[chosen_idx]
    else:
        item, seq, msk, vec = train_items[chosen_idx], train_seqs[chosen_idx], train_masks[chosen_idx], train_vecs[chosen_idx]

    print("=" * 60)
    print(f"  Sample: {item.get('sample_id')} ({chosen_split} #{chosen_idx})")
    print(f"  Ground truth: label={item.get('label')}, detail={item.get('label_detail') or '-'}, "
          f"diagnosis={item.get('diagnosis')}")
    print("=" * 60)

    pipe = MicrobiomePipeline().load(with_retrieval=not args.no_retrieval)
    out = pipe.infer(seq, msk, sample_vec=vec, with_retrieval=not args.no_retrieval)

    print("\n[Stage 1: Binary diagnosis]")
    print(f"  Predicted: {out['stage1_binary']['label']}")
    print(f"  Generation: {out['stage1_binary']['text'][:300]}")

    if out["stage2_subtype"]:
        print("\n[Stage 2: CD/UC subtype]")
        print(f"  Predicted: {out['stage2_subtype']['label']}")
        print(f"  Generation: {out['stage2_subtype']['text'][:300]}")
    else:
        print("\n[Stage 2: skipped (Healthy)]")

    if out["stage3_retrieval"]:
        print(f"\n[Stage 3: Retrieval] (query='{out['stage3_retrieval']['query']}')")
        for j, h in enumerate(out["stage3_retrieval"]["hits"], 1):
            print(f"  [{j}] score={h['score']:.4f}  {h['sample_id']} "
                  f"({h['label']} {h['label_detail'] or '-'})")

    print()


if __name__ == "__main__":
    main()
