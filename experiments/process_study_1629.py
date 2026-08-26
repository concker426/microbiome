import h5py, numpy as np, json
P = "/hd/liujx/microbiome_llm_project"
base = "/hd/liujx/microbiome_data/study_1629/BIOM"
info = json.load(open(f"{P}/data/qiita_ibd/combined_info.json"))
vocab = info["genus_names"]
vindex = {g: i for i, g in enumerate(vocab)}
print("vocab size:", len(vocab))

def gname(b):
    if isinstance(b, bytes):
        s = b.decode()
    else:
        s = str(b)
    s = s.strip()
    if s.startswith("g__"):
        s = s[3:]
    return s

def load_otu(sub):
    f = h5py.File(f"{base}/{sub}/otu_table.biom", "r")
    obs_ids = [str(x) for x in f["/observation/ids"]]
    samp_ids = [str(x) for x in f["/sample/ids"]]
    tax = f["/observation/metadata/taxonomy"][()]
    data = f["/observation/matrix/data"][()]
    idx = f["/observation/matrix/indices"][()]
    indptr = f["/observation/matrix/indptr"][()]
    f.close()
    n_samp = len(samp_ids)
    mat = np.zeros((n_samp, len(vocab)), dtype=np.float64)
    matched = unmatched = unclassified = 0
    uniq = set()
    for o in range(len(obs_ids)):
        g = gname(tax[o][5]) if len(tax[o]) > 5 else ""
        if not g or g == "":
            unclassified += 1
            continue
        uniq.add(g)
        gi = vindex.get(g)
        if gi is None:
            unmatched += 1
            continue
        matched += 1
        vals = data[indptr[o]:indptr[o+1]]
        cols = idx[indptr[o]:indptr[o+1]]
        for v, c in zip(vals, cols):
            if c < n_samp:
                mat[c, gi] += v
    print(f"{sub}: obs={len(obs_ids)} samples={n_samp} | matched={matched} unmatched={unmatched} unclassified={unclassified} | unique genera={len(uniq)}")
    mat = mat / mat.sum(1).clip(min=1)[:, None] * 100.0
    return samp_ids, mat

s1, m1 = load_otu("19865")
s2, m2 = load_otu("43623")
all_ids = s1 + s2
all_mat = np.vstack([m1, m2])
seen = {}
keep_ids = []
keep_mat = []
for sid, vec in zip(all_ids, all_mat):
    if sid not in seen:
        seen[sid] = True
        keep_ids.append(sid)
        keep_mat.append(vec)
vecs = np.array(keep_mat)
print("unique samples:", len(keep_ids), "| vectors:", vecs.shape, "| richness mean: %.1f" % (vecs > 0).sum(1).mean())
np.save(f"{P}/data/study_1629_vectors.npy", vecs)
json.dump({"ids": keep_ids, "labels": ["Disease"] * len(keep_ids)}, open(f"{P}/data/study_1629_labels.json", "w"), indent=1)
print("Saved study_1629_vectors.npy + labels.json")
