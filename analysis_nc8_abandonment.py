"""
NC8 — Silent abandonment: fixes that are neither merged nor closed.
The rejection rate only sees decided PRs; here we measure the share of fix PRs still UNDECIDED
60 days after creation — maintainer attention debt. Uniform 60-day window for every PR (created
>= 60d before dataset end), so age cannot bias the comparison. Agent vs human in matched repos,
per agent over time. Dataset-only.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, chi2_contingency

BASE = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes"
D = os.path.join(BASE, "data"); FIG = os.path.join(BASE, "figures")
END = pd.Timestamp("2026-02-28", tz="UTC"); WIN = 60; MIN_N = 100

clean = pd.read_parquet(os.path.join(BASE, "analysis_clean.parquet"))
ts = pd.read_parquet(os.path.join(D, "fix_prs_only.parquet"), columns=["id", "merged_at", "closed_at"])
for c in ["merged_at", "closed_at"]:
    ts[c] = pd.to_datetime(ts[c], utc=True, errors="coerce")
d = clean.merge(ts, on="id", how="left")

# uniform window: every PR observed for exactly WIN days
d = d[d.created_dt <= END - pd.Timedelta(days=WIN)].copy()
dec = d[["merged_at", "closed_at"]].min(axis=1)
d["undecided"] = ~((dec.notna()) & ((dec - d.created_dt).dt.days <= WIN))

m = d[d.matched & d.group.isin(["agent", "human"])]
ct = pd.crosstab(m.group, m.undecided)
p = chi2_contingency(ct)[1]
print(f"fix PRs with a full {WIN}d observation window: {len(d):,}")
print(f"\nundecided at {WIN} days (matched repos, chi2 p={p:.2e}):")
for g in ["agent", "human"]:
    s = m[m.group == g]
    print(f"  {g:6s}: {100*s.undecided.mean():5.1f}%  (n={len(s):,})")

# trend per month (matched)
print(f"\ntrend of undecided-at-{WIN}d share (Spearman over months, matched):")
series = {}
for g in ["agent", "human"]:
    s = m[m.group == g]
    mo = s.groupby("month")["undecided"].agg(["size", "mean"])
    mo = mo[mo["size"] >= MIN_N]["mean"] * 100
    series[g] = mo
    rho, pp = spearmanr(range(len(mo)), mo.values)
    print(f"  {g:6s}: {mo.iloc[0]:5.1f}% -> {mo.iloc[-1]:5.1f}%   rho={rho:+.2f} p={pp:.3f}")

# per agent (all agent data)
ag = d[d.group == "agent"]
print(f"\nundecided at {WIN}d per agent (all agent data):")
agser = {}
for a in ["Claude_Code", "Cursor", "Copilot", "Devin"]:
    s = ag[ag.agent == a]
    mo = s.groupby("month")["undecided"].agg(["size", "mean"])
    mo = mo[mo["size"] >= MIN_N]["mean"] * 100
    if len(mo) >= 4:
        rho, pp = spearmanr(range(len(mo)), mo.values)
        agser[a] = mo
        print(f"  {a:12s}: overall {100*s.undecided.mean():5.1f}% (n={len(s):,}) | "
              f"trend {mo.iloc[0]:5.1f}% -> {mo.iloc[-1]:5.1f}%  rho={rho:+.2f} p={pp:.3f}")

# how rejection rate would shift if undecided counted as rejected (pessimistic bound, matched)
print("\npessimistic bound (undecided counted as rejected, matched):")
for g in ["agent", "human"]:
    s = m[m.group == g]
    dec_s = s[~s.undecided]
    rej_dec = 100 * (dec_s.rejected & ((s.loc[dec_s.index, 'closed_at'] - s.loc[dec_s.index, 'created_dt']).dt.days <= WIN)).mean()
    rej_pess = 100 * (s.undecided | s.rejected).mean()
    print(f"  {g:6s}: decided-only rejection {rej_dec:5.1f}%  ->  with undecided {rej_pess:5.1f}%")

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for g, col in [("agent", "#d62728"), ("human", "#1f77b4")]:
    ax[0].plot(series[g].index, series[g].values, "o-", color=col, label=g)
ax[0].set_ylabel(f"% undecided at {WIN} days"); ax[0].set_title(
    "NC8: silent abandonment, agent vs human\n(matched repos, uniform 60d window)")
cols = {"Claude_Code": "#9467bd", "Cursor": "#2ca02c", "Copilot": "#ff7f0e", "Devin": "#8c564b"}
for a, mo in agser.items():
    ax[1].plot(mo.index, mo.values, "o-", color=cols[a], label=a, alpha=.85)
ax[1].set_ylabel(f"% undecided at {WIN} days"); ax[1].set_title("NC8: abandonment per agent (all agent fixes)")
for a_ in ax:
    a_.legend(); a_.grid(alpha=.3); a_.tick_params(axis="x", rotation=60)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "nc8_abandonment.png"), dpi=130); plt.close()
print("\nsaved figures/nc8_abandonment.png")
