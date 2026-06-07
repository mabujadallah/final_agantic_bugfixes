"""
NC2 — Is a wasted agent bug-fix predictable at submission time?
Population: agent fix PRs, matched repos, closed (decided). Target: rejected (1) vs merged (0).
Temporal split (train past / test future), no leakage. Reports AUC, PR-AUC, calibration, importances.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve

BASE = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes"
FIG = os.path.join(BASE, "figures")
df = pd.read_parquet(os.path.join(BASE, "analysis_clean.parquet"))

# population: agent fixes, matched repos, decided (closed)
d = df[(df.group == "agent") & df.matched & df.closed].copy()
d["created_dt"] = pd.to_datetime(d["created_dt"], utc=True)
d = d.sort_values("created_dt")

# time-causal repo-history feature: prior rejection rate among earlier agent fixes in the same repo
d["prior_n"]   = d.groupby("repo_id").cumcount()
d["prior_rej"] = (d.groupby("repo_id")["rejected"]
                    .apply(lambda s: s.shift().expanding().mean()).reset_index(level=0, drop=True))
d["prior_rej"] = d["prior_rej"].fillna(d["rejected"].mean())

# temporal split: train < 2025-10, test 2025-10..2025-12 (mature/decided); drop censored 2026
tr = d[d.created_dt < "2025-10-01"]
te = d[(d.created_dt >= "2025-10-01") & (d.created_dt < "2026-01-01")]
y_tr, y_te = tr["rejected"].astype(int), te["rejected"].astype(int)
print(f"train={len(tr):,} (rej {100*y_tr.mean():.1f}%) | test={len(te):,} (rej {100*y_te.mean():.1f}%)")

num  = ["log_churn","n_files","n_commits","title_len","body_len","prior_rej","prior_n"]
binr = ["has_test","links_issue","has_files"]
cat  = ["agent","btype"]
FEATS = num + binr + cat
Xtr, Xte = tr[FEATS], te[FEATS]
pre = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("bin", "passthrough", binr),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])

models = {
    "LogReg": Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))]),
    "GBM":    Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08))]),
}
base = y_te.mean()
print(f"baseline (predict-all-reject precision = positive rate): {base:.3f}")
results = {}
for name, m in models.items():
    m.fit(Xtr, y_tr)
    p = m.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(y_te, p); ap = average_precision_score(y_te, p); brier = brier_score_loss(y_te, p)
    results[name] = (p, auc, ap, brier)
    print(f"{name}: ROC-AUC={auc:.3f}  PR-AUC={ap:.3f} (base {base:.3f})  Brier={brier:.3f}")

# permutation importance (GBM, on test)
gbm = models["GBM"]
pi = permutation_importance(gbm, Xte, y_te, scoring="roc_auc", n_repeats=5, random_state=0, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=FEATS).sort_values(ascending=False)
print("\npermutation importance (drop in ROC-AUC):")
print(imp.round(4).to_string())

# ---- figures: ROC + importances ----
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for name, (p, auc, ap, brier) in results.items():
    fpr, tpr, _ = roc_curve(y_te, p); ax[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
ax[0].plot([0,1],[0,1],"k--",alpha=.4); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
ax[0].set_title("NC2: predicting agent bug-fix rejection (held-out future)"); ax[0].legend(); ax[0].grid(alpha=.3)
imp.head(10)[::-1].plot.barh(ax=ax[1], color="#d62728")
ax[1].set_xlabel("Importance (drop in ROC-AUC when shuffled)"); ax[1].set_title("NC2: top predictive signals (GBM)")
ax[1].grid(axis="x", alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "nc2_predict.png"), dpi=130); plt.close()

# logistic coefficients (direction, interpretable)
lr = models["LogReg"]; ohe = lr.named_steps["pre"].named_transformers_["cat"]
names = num + binr + list(ohe.get_feature_names_out(cat))
coef = pd.Series(lr.named_steps["clf"].coef_[0], index=names).sort_values()
print("\nlogistic coefficients (most protective / most rejection-prone):")
print(pd.concat([coef.head(6), coef.tail(6)]).round(3).to_string())
print("\nsaved figures/nc2_predict.png")
