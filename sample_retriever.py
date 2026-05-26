#!/usr/bin/env python3
"""
Sample retrieval: given a textual query (symptom description, genus name,
diagnosis hint, etc.), return the K most similar samples from the
training set with their labels and dominant genera.

Uses TF-IDF + cosine similarity over per-sample document descriptions.
A character-level analyzer handles Chinese without needing jieba.

Usage as a library:
    retr = SampleRetriever()
    retr.build_index()
    hits = retr.search("克罗恩 Faecalibacterium 减少", k=5)

Usage as CLI:
    python3 sample_retriever.py "症状关键词"
    python3 sample_retriever.py --rebuild
"""
import os, json, argparse, pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = "/hd/liujx/microbiome_llm_project"
SRC_TRAIN = os.path.join(BASE, "data/agp_ftp_processed/train_set.jsonl")
SRC_TEST = os.path.join(BASE, "data/agp_ftp_processed/test_set.jsonl")
TRAIN_VEC = os.path.join(BASE, "data/agp_ftp_processed/train_set_vectors.npy")
TEST_VEC = os.path.join(BASE, "data/agp_ftp_processed/test_set_vectors.npy")
GENUS = os.path.join(BASE, "data/agp_ftp_processed/genus_names.npy")

INDEX_PATH = os.path.join(BASE, "data/sample_retrieval_index.pkl")


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(line) for line in f]


def build_doc(item, vec, genus_names, top_n=15):
    """Build a search document for a sample combining label, diagnosis,
    and top-N genera (name + abundance %)."""
    label = item.get("label", "")
    detail = item.get("label_detail", "")
    diagnosis = item.get("diagnosis", "")
    top_idx = np.argsort(-vec)[:top_n]
    genus_part = "，".join(
        f"{genus_names[i]} {vec[i]:.1f}%" for i in top_idx if vec[i] > 0
    )
    parts = [label, detail or "", diagnosis or "", genus_part]
    if label == "Disease":
        parts.append("疾病 患者 异常 失调")
    if detail == "CD":
        parts.append("克罗恩 克罗恩病 Crohn IBD 炎症 回肠 结肠")
    if detail == "UC":
        parts.append("溃疡性结肠炎 ulcerative IBD 炎症 远端结肠")
    if label == "Healthy":
        parts.append("健康 正常")
    return " ".join(p for p in parts if p)


class SampleRetriever:
    def __init__(self, index_path=INDEX_PATH):
        self.index_path = index_path
        self.docs = None
        self.metadata = None
        self.vectorizer = None
        self.matrix = None

    def build_index(self):
        print("Building retrieval index...")
        train_items = load_jsonl(SRC_TRAIN)
        test_items = load_jsonl(SRC_TEST)
        train_vec = np.load(TRAIN_VEC)
        test_vec = np.load(TEST_VEC)
        genus_names = np.load(GENUS, allow_pickle=True)
        if len(genus_names) > train_vec.shape[1]:
            genus_names = genus_names[:train_vec.shape[1]]

        docs, metas = [], []
        for items, vecs, split in [(train_items, train_vec, "train"),
                                    (test_items, test_vec, "test")]:
            for i, item in enumerate(items):
                docs.append(build_doc(item, vecs[i], genus_names))
                metas.append({
                    "split": split,
                    "row": i,
                    "sample_id": item.get("sample_id"),
                    "label": item.get("label"),
                    "label_detail": item.get("label_detail", ""),
                    "diagnosis": item.get("diagnosis", ""),
                    "top_genera": [
                        {"name": str(genus_names[j]), "abundance_pct": float(vecs[i][j])}
                        for j in np.argsort(-vecs[i])[:5] if vecs[i][j] > 0
                    ],
                })

        # char n-gram TF-IDF handles both Chinese and Latin tokens uniformly
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            max_features=80000, lowercase=True,
        )
        matrix = vectorizer.fit_transform(docs)
        print(f"  Indexed {len(docs)} samples, vocab={len(vectorizer.vocabulary_)}")

        self.docs = docs
        self.metadata = metas
        self.vectorizer = vectorizer
        self.matrix = matrix

        with open(self.index_path, "wb") as f:
            pickle.dump({
                "docs": docs, "metadata": metas,
                "vectorizer": vectorizer, "matrix": matrix,
            }, f)
        print(f"  Saved → {self.index_path}")
        return self

    def load(self):
        if self.matrix is not None:
            return self
        if not os.path.exists(self.index_path):
            return self.build_index()
        with open(self.index_path, "rb") as f:
            d = pickle.load(f)
        self.docs = d["docs"]
        self.metadata = d["metadata"]
        self.vectorizer = d["vectorizer"]
        self.matrix = d["matrix"]
        return self

    def search(self, query: str, k: int = 5):
        self.load()
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = np.argsort(-scores)[:k]
        return [{
            "score": float(scores[i]),
            **self.metadata[i],
        } for i in top_idx if scores[i] > 0]


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="*", help="search query")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args()

    retr = SampleRetriever()
    if args.rebuild or not os.path.exists(INDEX_PATH):
        retr.build_index()
    else:
        retr.load()

    if not args.query:
        print("Usage: python3 sample_retriever.py 'symptom keywords' [-k N]")
        return
    query = " ".join(args.query)
    hits = retr.search(query, k=args.k)
    print(f'\nQuery: "{query}"')
    print(f"Top-{len(hits)} matches:\n")
    for j, h in enumerate(hits, 1):
        print(f"  [{j}] score={h['score']:.4f}  sample={h['sample_id']}")
        print(f"      label={h['label']} detail={h['label_detail'] or '-'}  diagnosis={h['diagnosis']}")
        gen = ", ".join(f"{g['name']}({g['abundance_pct']:.1f}%)" for g in h["top_genera"][:5])
        print(f"      top genera: {gen}\n")


if __name__ == "__main__":
    cli()
