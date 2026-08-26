import pandas as pd, numpy as np
from scipy.stats import binomtest
P = "/hd/liujx/microbiome_llm_project"
def load(p):
    return pd.read_csv(p)
cl = load(f"{P}/experiments/results/decontaminated_groupcv_classical_no_xgboost_20260804/predictions_by_fold.csv")
se = load(f"{P}/experiments/results/decontaminated_groupcv_simpleemb_mlp_20260804/predictions_by_fold.csv")
mg = load(f"{P}/experiments/results/decontaminated_groupcv_mgm_20260804/predictions_by_fold.csv")
cl = cl.rename(columns={"model": "model_name"})
se = se.rename(columns={"model": "model_name"})
mg = mg.rename(columns={"model": "model_name"})
cl["model"] = cl["model_name"]
se["model"] = "SimpleEmb"
mg["model"] = "MGM+" + mg["model_name"].str.replace("MGM_", "").str.replace("LogisticRegression", "LR").str.replace("RandomForest", "RF")
df = pd.concat([cl, se, mg], ignore_index=True)
print("models:", sorted(df["model"].unique()))
key = ["seed", "fold", "sample_id"]
piv = df.pivot_table(index=key, columns="model", values="predicted_label", aggfunc="first")
print("pivot shape:", piv.shape)
models = [m for m in piv.columns if m != "SimpleEmb"]
rows = []
for m in models:
    sub = piv[["SimpleEmb", m]].dropna()
    if len(sub) < 10: continue
    a = sub["SimpleEmb"].astype(int); b_ = sub[m].astype(int)
    disc_b = ((a == 1) & (b_ == 0)).sum()  # SimpleEmb correct, other wrong
    disc_c = ((a == 0) & (b_ == 1)).sum()  # other correct, SimpleEmb wrong
    n = disc_b + disc_c
    pv = binomtest(max(disc_b, disc_c), n, 0.5, alternative="two-sided").pvalue if n > 0 else 1.0
    acc_se = (a == piv["SimpleEmb"]).mean() if False else (a == sub["SimpleEmb"]).mean()
    acc_m = (b_ == sub[m]).mean()
    rows.append({"comparison": f"SimpleEmb vs {m}", "n": int(n), "disc_B": int(disc_b), "disc_C": int(disc_c), "p_raw": pv, "acc_SimpleEmb": acc_se, "acc_other": acc_m})
res = pd.DataFrame(rows)
# Benjamini-Hochberg
res = res.sort_values("p_raw").reset_index(drop=True)
m = len(res)
res["q_bh"] = [min(1.0, res["p_raw"][j] * m / (j + 1)) for j in range(m)]
# enforce monotonicity
for j in range(m - 2, -1, -1):
    res.loc[j, "q_bh"] = min(res.loc[j, "q_bh"], res.loc[j + 1, "q_bh"])
res = res.sort_values("comparison").reset_index(drop=True)
pd.set_option("display.width", 160)
print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
out = res.to_dict("records")
import json
json.dump(out, open(f"{P}/ProCyon_v2/analysis/fdr_mcnemar_results.json", "w"), indent=2)
print("Saved fdr_mcnemar_results.json")
