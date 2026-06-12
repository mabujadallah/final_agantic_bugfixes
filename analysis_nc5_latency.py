"""
NC5 — Rubber-stamping: are agent fixes decided faster and faster?
Time-to-decision (merged_at/closed_at - created_at) for agent vs human fixes in matched repos,
plus the share of merged fixes accepted within 10 minutes ("fast-merge" = effectively no human
review). Trends tested with Spearman. Dataset-only (analysis_clean + timestamps from fix_prs_only).
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu

BASE = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes"
D = os.path.join(BASE, "data"); FIG = os.path.join(BASE, "figures")
FAST_MIN = 10                      # merged within 10 minutes = "fast-merge"
MIN_N = 100                        # min PRs per month-cell to plot

clean = pd.read_parquet(os.path.join(BASE, "analysis_clean.parquet"))
ts = pd.read_parquet(os.path.join(D, "fix_prs_only.parquet"), columns=["id", "merged_at", "closed_at"])
for c in ["merged_at", "closed_at"]:
    ts[c] = pd.to_datetime(ts[c], utc=True, errors="coerce")
d = clean.merge(ts, on="id", how="left")

# decided population
d = d[d.closed & d.mature].copy()
d["decision_dt"] = np.where(d.merged, d.merged_at, d.closed_at)
d["decision_dt"] = pd.to_datetime(d["decision_dt"], utc=True)
d["lat_h"] = (d.decision_dt - d.created_dt).dt.total_seconds() / 3600
d = d[d.lat_h.notna() & (d.lat_h >= 0)].copy()           # drop missing/negative-clock glitches
d["fast"] = d.merged & (d.lat_h <= FAST_MIN / 60)

m = d[d.matched & d.group.isin(["agent", "human"])]
print(f"decided fixes with timestamps: {len(d):,} | matched agent={len(m[m.group=='agent']):,} "
      f"human={len(m[m.group=='human']):,}")

# ---- headline: latency by group (matched) ----
for g in ["agent", "human"]:
    s = m[m.group == g]
    mg, rj = s[s.merged], s[s.rejected]
    print(f"  {g:6s}: median time-to-merge {mg.lat_h.median():6.1f} h | "
          f"time-to-rejection {rj.lat_h.median():6.1f} h | "
          f"fast-merge (<= {FAST_MIN} min) {100*mg.fast.mean():.1f}% of merges")
u = mannwhitneyu(m[m.group=='agent' ].loc[m.merged, 'lat_h'],
                 m[m.group=='human'].loc[m.merged, 'lat_h'])
print(f"  Mann-Whitney agent vs human time-to-merge: p={u.pvalue:.2e}")

# ---- trends over months ----
def monthly(df, col, fn="median"):
    g = df.groupby("month")[col]
    out = g.median() if fn == "median" else g.mean()
    return out[df.groupby("month").size() >= MIN_N]

print("\ntrends (Spearman over monthly series, matched repos):")
series = {}
for g in ["agent", "human"]:
    s = m[(m.group == g) & m.merged]
    tt = monthly(s, "lat_h"); fast = monthly(s, "fast", "mean") * 100
    series[g] = (tt, fast)
    for name, ser in [("median time-to-merge (h)", tt), ("fast-merge share (%)", fast)]:
        rho, p = spearmanr(range(len(ser)), ser.values)
        print(f"  {g:6s} {name:26s}: {ser.iloc[0]:7.1f} -> {ser.iloc[-1]:7.1f}   rho={rho:+.2f} p={p:.3f}")

# per-agent fast-merge trend (all agents, decided+merged)
print("\nfast-merge share per agent (all agent data, merged fixes):")
ag = d[(d.group == "agent") & d.merged]
agfast = {}
for a in ["Claude_Code", "Cursor", "Copilot", "Devin"]:
    s = ag[ag.agent == a]
    f = monthly(s, "fast", "mean") * 100
    if len(f) >= 4:
        rho, p = spearmanr(range(len(f)), f.values)
        agfast[a] = f
        print(f"  {a:12s}: {f.iloc[0]:5.1f}% -> {f.iloc[-1]:5.1f}%   rho={rho:+.2f} p={p:.3f} "
              f"(overall {100*s.fast.mean():.1f}%, n={len(s):,})")

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for g, col in [("agent", "#d62728"), ("human", "#1f77b4")]:
    tt, fast = series[g]
    ax[0].plot(tt.index, tt.values, "o-", color=col, label=g)
    ax[1].plot(fast.index, fast.values, "o-", color=col, label=f"{g} (matched)")
ax[0].set_ylabel("Median time-to-merge (hours)"); ax[0].set_yscale("log")
ax[0].set_title("NC5: how long until an agent/human fix is merged\n(matched repos, log scale)")
for a, col in [("Claude_Code", "#9467bd"), ("Copilot", "#ff7f0e")]:
    if a in agfast:
        ax[1].plot(agfast[a].index, agfast[a].values, "--", alpha=.7, color=col, label=a)
ax[1].set_ylabel(f"Share of merges within {FAST_MIN} min (%)")
ax[1].set_title("NC5: fast-merges = accepted with effectively no review")
for a_ in ax:
    a_.legend(); a_.grid(alpha=.3); a_.tick_params(axis="x", rotation=60)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "nc5_latency.png"), dpi=130); plt.close()
print("\nsaved figures/nc5_latency.png")
