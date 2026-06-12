"""
NC7 — Re-fix recurrence: a durability measure with actual signal (reverts are ~0.1%).
A merged fix is "re-fixed" if ANY non-test, non-generated file it touched is touched again by a
LATER fix PR (any author) in the same repo within 90 days of the merge. Fixed window (merge
>= 90 days before dataset end). Agent vs human in matched repos, with size and bug-type controls
and a within-repo paired check. Dataset-only.
"""
import os, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, wilcoxon

BASE = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes"
D = os.path.join(BASE, "data"); FIG = os.path.join(BASE, "figures")
END = pd.Timestamp("2026-02-28", tz="UTC"); WIN = 90

clean = pd.read_parquet(os.path.join(BASE, "analysis_clean.parquet"))
ts = pd.read_parquet(os.path.join(D, "fix_prs_only.parquet"), columns=["id", "merged_at"])
ts["merged_at"] = pd.to_datetime(ts["merged_at"], utc=True, errors="coerce")
clean = clean.merge(ts, on="id", how="left")

# file touches: every (fix PR, code file) pair — same NOISE filter as build_clean, plus test files
det = pd.read_parquet(os.path.join(D, "fix_pr_commit_details.parquet"), columns=["pr_id", "filename"])
det["fn"] = det["filename"].fillna("")
NOISE = re.compile(
    r"package-lock\.json|yarn\.lock|pnpm-lock|npm-shrinkwrap|go\.sum|Cargo\.lock|poetry\.lock|"
    r"Gemfile\.lock|composer\.lock|\.min\.(js|css)|/vendor/|/dist/|/build/|generated|\.pb\.go$|"
    r"_pb2\.py$|\.snap$|\.lock$", re.I)
TEST_RE = re.compile(r"(^|/)(tests?|__tests__|spec)(/|\.)|(_test\.|test_|\.test\.|\.spec\.|_spec\.|\.tests\.)", re.I)
det = det[(det.fn != "") & ~det.fn.map(lambda f: bool(NOISE.search(f)) or bool(TEST_RE.search(f)))]
det = det[["pr_id", "fn"]].drop_duplicates()

meta = clean[["id", "repo_id", "group", "agent", "merged", "matched", "btype",
              "created_dt", "merged_at", "n_files_clean"]]
touch = det.merge(meta, left_on="pr_id", right_on="id", how="inner")
print(f"(PR, code-file) touches: {len(touch):,} across {touch.pr_id.nunique():,} fix PRs")

# events = merged fixes in matched repos, agent|human, with a full 90d window after merge
ev = touch[touch.merged & touch.matched & touch.group.isin(["agent", "human"])
           & (touch.merged_at <= END - pd.Timedelta(days=WIN))].copy()

# first later fix-PR touch of the same (repo, file): merge_asof forward on creation time
later = touch[["repo_id", "fn", "created_dt"]].sort_values("created_dt")
ev = ev.sort_values("merged_at")
hit = pd.merge_asof(ev, later.rename(columns={"created_dt": "next_touch"}),
                    by=["repo_id", "fn"], left_on="merged_at", right_on="next_touch",
                    direction="forward", allow_exact_matches=False)
hit["refixed_file"] = (hit.next_touch - hit.merged_at).dt.days <= WIN

# file hotness: how many distinct fix PRs ever touch this (repo, file) — ceiling control
hot = touch.groupby(["repo_id", "fn"])["pr_id"].nunique().rename("fn_touches")
hit = hit.merge(hot, left_on=["repo_id", "fn"], right_index=True, how="left")
hit["gap_days"] = (hit.next_touch - hit.merged_at).dt.days

# PR-level: re-fixed if any of its files is re-touched in the window
pr = (hit.groupby("pr_id").agg(refixed=("refixed_file", "any"), repo_id=("repo_id", "first"),
                               group=("group", "first"), agent=("agent", "first"),
                               btype=("btype", "first"), nf=("n_files_clean", "first"),
                               gap=("gap_days", "min"), max_hot=("fn_touches", "max")))
print(f"merged fixes scored: {len(pr):,} (agent {int((pr.group=='agent').sum()):,} | "
      f"human {int((pr.group=='human').sum()):,})")

ct = pd.crosstab(pr.group, pr.refixed)
chi2, p = chi2_contingency(ct)[:2]
ra = 100 * pr[pr.group == "agent"].refixed.mean(); rh = 100 * pr[pr.group == "human"].refixed.mean()
print(f"\nre-fixed within {WIN}d of merge:  agent {ra:.1f}%  vs  human {rh:.1f}%   (chi2 p={p:.2e})")

# window sensitivity: quick re-fixes are likelier true bug recurrence than file churn
print("\nre-fix rate by window (agent vs human):")
for w in [7, 14, 30, 90]:
    a = 100 * (pr[pr.group == "agent"].gap <= w).mean(); h = 100 * (pr[pr.group == "human"].gap <= w).mean()
    print(f"  <= {w:3d}d : agent {a:5.1f}%  human {h:5.1f}%  (gap {a-h:+.1f} pts)")

# cold-file subset: every touched file is touched by <=3 fix PRs ever — here a re-touch is
# far likelier to mean the bug came back, not just a hot file churning
cold = pr[pr.max_hot <= 3]
ca = 100 * cold[cold.group == "agent"].refixed.mean(); ch = 100 * cold[cold.group == "human"].refixed.mean()
cp = chi2_contingency(pd.crosstab(cold.group, cold.refixed))[1]
print(f"\nCOLD-FILE fixes only (all files touched by <=3 fix PRs ever; "
      f"n agent={int((cold.group=='agent').sum()):,}, human={int((cold.group=='human').sum()):,}):")
print(f"  re-fixed within {WIN}d:  agent {ca:.1f}%  vs  human {ch:.1f}%   (chi2 p={cp:.2e})")

# size control: stratify by number of clean files touched
pr["fbin"] = pd.cut(pr.nf, [-1, 1, 3, 10, 1e9], labels=["1 file", "2-3", "4-10", ">10"])
strat = pr.groupby(["fbin", "group"], observed=True)["refixed"].agg(["size", "mean"]).unstack()
strat_pct = (strat["mean"] * 100).round(1); strat_pct["n_agent"] = strat["size"]["agent"]
print(f"\nby files touched (controls 'more files = more chances'):")
print(strat_pct.to_string())

# within-repo paired check: repos with >=20 scored fixes from each group
cnt = pr.groupby(["repo_id", "group"]).size().unstack(fill_value=0)
big = cnt[(cnt.get("agent", 0) >= 20) & (cnt.get("human", 0) >= 20)].index
pair = pr[pr.repo_id.isin(big)].groupby(["repo_id", "group"])["refixed"].mean().unstack()
w = wilcoxon(pair["agent"], pair["human"])
print(f"\nwithin-repo paired ({len(pair)} repos with >=20 scored fixes each): "
      f"agent median {100*pair['agent'].median():.1f}% vs human {100*pair['human'].median():.1f}% "
      f"(Wilcoxon p={w.pvalue:.3f})")

# by bug type (agent vs human gap)
bt = pr.groupby(["btype", "group"])["refixed"].agg(["size", "mean"]).unstack()
ok = bt["size"].min(axis=1) >= 150
gap = ((bt["mean"]["agent"] - bt["mean"]["human"]) * 100)[ok].sort_values()
print("\nagent - human re-fix gap by bug type (pts, n>=150 each):")
print(gap.round(1).to_string())

# per agent
pa = pr[pr.group == "agent"].groupby("agent")["refixed"].agg(["size", "mean"])
pa["mean"] *= 100
print("\nre-fix rate per agent:"); print(pa.round(1).to_string())

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sp = strat["mean"] * 100
sp.plot.bar(ax=ax[0], color={"agent": "#d62728", "human": "#1f77b4"})
ax[0].set_ylabel(f"% merged fixes re-fixed within {WIN}d"); ax[0].set_xlabel("Code files touched")
ax[0].set_title("NC7: re-fix recurrence, agent vs human\n(matched repos, merged fixes)")
ax[0].legend(); ax[0].grid(axis="y", alpha=.3); ax[0].tick_params(axis="x", rotation=0)
gap.plot.barh(ax=ax[1], color=np.where(gap.values > 0, "#d62728", "#2ca02c"))
ax[1].set_xlabel("Agent − human re-fix rate (pts)"); ax[1].axvline(0, color="k", lw=.8)
ax[1].set_title("NC7: where agent fixes get re-fixed more (red) or less (green)")
ax[1].grid(axis="x", alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "nc7_refix.png"), dpi=130); plt.close()
print("\nsaved figures/nc7_refix.png")
