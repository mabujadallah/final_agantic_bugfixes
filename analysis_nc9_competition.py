"""
NC9 — Within-repo agent competition: do repos consolidate on one agent, and does the better
agent win? RQ2a showed the global market shift; here we look inside multi-agent repos:
(1) consolidation — HHI of the agent mix, first half vs second half of each repo's agent activity;
(2) head-to-head win matrix — among repos where A and B both compete early, who dominates at the end;
(3) merit — does the agent with the lower early rejection rate win the repo? Dataset-only.
"""
import os, itertools, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, binomtest

BASE = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes"; FIG = os.path.join(BASE, "figures")
AGENTS = ["Claude_Code", "Cursor", "Copilot", "Devin"]
MIN_FIXES = 10

clean = pd.read_parquet(os.path.join(BASE, "analysis_clean.parquet"))
ag = clean[clean.group == "agent"].sort_values("created_dt").copy()

# multi-agent repos with enough history
g = ag.groupby("repo_id").agg(n=("id", "size"), k=("agent", "nunique"))
multi = g[(g.n >= MIN_FIXES) & (g.k >= 2)].index
d = ag[ag.repo_id.isin(multi)].copy()
d["half"] = d.groupby("repo_id").cumcount() < d.groupby("repo_id")["id"].transform("size") // 2
d["half"] = np.where(d["half"], "early", "late")
n_repos_any = ag.repo_id.nunique()
print(f"agent repos: {n_repos_any:,} | with >={MIN_FIXES} fixes & >=2 agents: {len(multi):,} "
      f"({len(d):,} fixes)")

# ---- (1) consolidation: HHI early vs late ----
def hhi(s): p = s.value_counts(normalize=True); return float((p ** 2).sum())
H = d.groupby(["repo_id", "half"])["agent"].apply(hhi).unstack()
H = H.dropna()
w = wilcoxon(H["early"], H["late"])
print(f"\nconsolidation (HHI of agent mix, 1.0 = single agent): "
      f"early median {H['early'].median():.2f} -> late {H['late'].median():.2f} "
      f"(Wilcoxon p={w.pvalue:.2e})")
mono_late = (H["late"] == 1.0).mean()
print(f"  repos fully consolidated on ONE agent in their late half: {100*mono_late:.1f}%")
# NOTE: the full-sample HHI drop is partly mechanical (a repo enters the sample even if its 2nd
# agent only appears late). Unbiased test: repos where >=2 agents already compete in the early half.
k_early = d[d.half == "early"].groupby("repo_id")["agent"].nunique()
comp = H.loc[H.index.intersection(k_early[k_early >= 2].index)]
w2 = wilcoxon(comp["early"], comp["late"])
print(f"  unbiased (>=2 agents already early, n={len(comp)}): HHI {comp['early'].median():.2f} -> "
      f"{comp['late'].median():.2f} (Wilcoxon p={w2.pvalue:.3f}); "
      f"fully consolidated late: {100*(comp['late']==1.0).mean():.1f}%")

# ---- (2) head-to-head: among repos where A and B both appear early, who dominates late ----
dom_late = d[d.half == "late"].groupby("repo_id")["agent"] \
            .agg(lambda s: s.value_counts().idxmax()).rename("winner")
early_set = d[d.half == "early"].groupby("repo_id")["agent"].agg(set).rename("early")
hh = pd.concat([early_set, dom_late], axis=1).dropna()
W = pd.DataFrame(0, index=AGENTS, columns=AGENTS)
for a, b in itertools.permutations(AGENTS, 2):
    both = hh[hh.early.map(lambda s: a in s and b in s)]
    W.loc[a, b] = int((both.winner == a).sum())
print("\nhead-to-head wins (row agent ends up dominant in repos where both competed early):")
print(W.to_string())
print("win rate when facing each other (row vs col):")
WR = (W / (W + W.T).replace(0, np.nan) * 100).round(0)
print(WR.to_string())

# ---- (3) merit: does the early-better agent win the repo? ----
e = d[(d.half == "early") & d.closed & d.mature]
er = e.groupby(["repo_id", "agent"])["rejected"].agg(["size", "mean"])
er = er[er["size"] >= 3]            # need >=3 decided early fixes per agent
cases = []
for rid, sub in er.groupby(level=0):
    if len(sub) < 2 or rid not in dom_late.index: continue
    sub = sub.droplevel(0).sort_values("mean")
    best, worst = sub.index[0], sub.index[-1]
    if sub["mean"].iloc[0] == sub["mean"].iloc[-1]: continue   # tie -> uninformative
    cases.append(dom_late.loc[rid] == best)
n_win = int(np.sum(cases)); n_tot = len(cases)
bt = binomtest(n_win, n_tot, 0.5)
print(f"\nmerit test: repos where 2+ agents have >=3 decided early fixes and unequal rejection rates: {n_tot}")
print(f"  early-better agent ends up dominant in {n_win}/{n_tot} = {100*n_win/n_tot:.0f}% "
      f"(binomial vs 50%: p={bt.pvalue:.3f})")

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].hist([H["early"], H["late"]], bins=np.linspace(0.25, 1.0, 16),
           label=["early half", "late half"], color=["#1f77b4", "#d62728"])
ax[0].set_xlabel("HHI of agent mix (1.0 = one agent only)"); ax[0].set_ylabel("repos")
ax[0].set_title("NC9: multi-agent repos consolidate over time")
ax[0].legend(); ax[0].grid(alpha=.3)
im = ax[1].imshow(WR.values.astype(float), cmap="RdYlGn", vmin=0, vmax=100)
ax[1].set_xticks(range(4), AGENTS, rotation=30); ax[1].set_yticks(range(4), AGENTS)
for i in range(4):
    for j in range(4):
        if i != j and not np.isnan(WR.values[i, j]):
            ax[1].text(j, i, f"{WR.values[i,j]:.0f}%\n(n={W.values[i,j]+W.values[j,i]})",
                       ha="center", va="center", fontsize=9)
ax[1].set_title("NC9: head-to-head — row agent wins the repo vs column agent")
plt.colorbar(im, ax=ax[1], label="win rate (%)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "nc9_competition.png"), dpi=130); plt.close()
print("\nsaved figures/nc9_competition.png")
