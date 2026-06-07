"""
NC4 — Did agents improve, or did the tasks get harder?
Kitagawa-Oaxaca-Blinder (3-fold) decomposition of each agent's early->late change in rejection rate
into: COMPOSITION (shift in task mix = bug-type x churn-size) vs RATE (true change at fixed task mix)
vs interaction. Bootstrap CIs. Dataset-only (analysis_clean).
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes"; FIG = os.path.join(BASE, "figures")
AGENTS = ["Claude_Code", "Cursor", "Copilot", "Devin"]
clean = pd.read_parquet(os.path.join(BASE, "analysis_clean.parquet"))
d = clean[(clean.group == "agent") & clean.closed & clean.mature].copy()
d["cbin"] = pd.cut(d["churn"], [-1, 10, 50, 1e12], labels=["small", "med", "large"])
d["stratum"] = d["btype"].astype(str) + "|" + d["cbin"].astype(str)
EARLY = ("2025-01", "2025-03"); LATE = ("2025-10", "2025-12")
d = d[d.month.between(EARLY[0], EARLY[1]) | d.month.between(LATE[0], LATE[1])].copy()
d["period"] = np.where(d.month.between(EARLY[0], EARLY[1]), "early", "late")

def decomp(df):
    e = df[df.period == "early"]; l = df[df.period == "late"]
    if len(e) < 50 or len(l) < 50: return None
    we = e.groupby("stratum").size() / len(e); wl = l.groupby("stratum").size() / len(l)
    re_ = e.groupby("stratum")["rejected"].mean(); rl = l.groupby("stratum")["rejected"].mean()
    S = we.index.union(wl.index)
    we = we.reindex(S, fill_value=0); wl = wl.reindex(S, fill_value=0)
    re_ = re_.reindex(S).fillna(e["rejected"].mean()); rl = rl.reindex(S).fillna(l["rejected"].mean())
    E = ((wl - we) * re_).sum()                 # composition
    C = (we * (rl - re_)).sum()                 # rate (within-stratum)
    I = ((wl - we) * (rl - re_)).sum()          # interaction
    return dict(delta=(l.rejected.mean()-e.rejected.mean())*100, comp=E*100, rate=C*100, inter=I*100)

rng = np.random.default_rng(0)
print(f"early={EARLY}, late={LATE}  | decomposition of rejection-rate change (pts)\n")
rows = {}
for a in AGENTS:
    df = d[d.agent == a]
    base = decomp(df)
    if base is None:
        print(f"  {a:12s}: insufficient volume"); continue
    # bootstrap composition share
    boots = []
    for _ in range(300):
        bs = df.sample(len(df), replace=True, random_state=rng.integers(1e9))
        r = decomp(bs)
        if r and abs(r["delta"]) > 1e-9: boots.append(r["comp"]/r["delta"])
    cs = np.percentile(boots, [2.5, 97.5]) if boots else [np.nan, np.nan]
    rows[a] = base
    print(f"  {a:12s}: Delta={base['delta']:+.1f}  | composition={base['comp']:+.1f}  "
          f"rate={base['rate']:+.1f}  interaction={base['inter']:+.1f}  "
          f"| comp share 95%CI [{cs[0]:.2f},{cs[1]:.2f}]")

# ---- figure: stacked composition vs rate (+interaction) per agent ----
ags = list(rows); comp = [rows[a]["comp"] for a in ags]; rate = [rows[a]["rate"] for a in ags]
inter = [rows[a]["inter"] for a in ags]; delta = [rows[a]["delta"] for a in ags]
x = np.arange(len(ags))
plt.figure(figsize=(9, 5))
plt.bar(x, comp, label="Composition (task mix)", color="#1f77b4")
plt.bar(x, rate, bottom=comp, label="Rate (true change)", color="#d62728")
plt.bar(x, inter, bottom=np.array(comp)+np.array(rate), label="Interaction", color="#999999")
plt.plot(x, delta, "ko", label="Total Δ")
plt.xticks(x, ags, rotation=15); plt.axhline(0, color="k", lw=.8)
plt.ylabel("Change in rejection rate, early->late (pts)")
plt.title("NC4: is the per-agent change real skill (rate) or task drift (composition)?")
plt.legend(); plt.grid(axis="y", alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(FIG, "nc4_decomp.png"), dpi=130); plt.close()
print("\nsaved figures/nc4_decomp.png")
