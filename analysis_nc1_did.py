"""
NC1 step 2 — does adopting an agent-instruction file reduce agent bug-fix rejection?
Difference-in-differences (two-way fixed effects) on agent fix PRs, treated repos (adopt during
window, with agent fixes before & after) vs never-adopt controls. Plus event-study (parallel-trends)
and a placebo. Consumes data/repo_adoption.parquet + analysis_clean.parquet.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

BASE = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes"; FIG = os.path.join(BASE, "figures")
clean = pd.read_parquet(os.path.join(BASE, "analysis_clean.parquet"))
ad = pd.read_parquet(os.path.join(BASE, "data", "repo_adoption.parquet"))

def midx(s): return pd.PeriodIndex(s, freq="M").astype("int64")
d = clean[(clean.group == "agent") & clean.closed & clean.mature].copy()
d["mi"] = midx(d["month"])
ad["ai"] = midx(ad["adopt_month"])
d = d.merge(ad[["repo_id", "ai"]], on="repo_id", how="left")
d["rej"] = d["rejected"].astype(int)

# treated repos: adopt AND have agent fixes both before and at/after adoption
g = d.dropna(subset=["ai"]).groupby("repo_id")
ok = g.apply(lambda x: (x.mi < x.ai.iloc[0]).any() and (x.mi >= x.ai.iloc[0]).any())
treated = set(ok[ok].index)
# controls: never adopt, with >= 8 agent fixes (stable), sample to keep TWFE tractable
never = d[d.ai.isna()]
ctrl_counts = never.groupby("repo_id").size()
ctrl_pool = ctrl_counts[ctrl_counts >= 8].index
rng = np.random.default_rng(0)
ctrl = set(rng.choice(ctrl_pool, size=min(500, len(ctrl_pool)), replace=False))

d["treated"] = d.repo_id.isin(treated)
sub = d[d.repo_id.isin(treated | ctrl)].copy()
sub["post"] = (sub.treated & (sub.mi >= sub.ai)).astype(int)
print(f"treated repos: {len(treated):,} | control repos: {len(ctrl):,} | agent fixes used: {len(sub):,}")

# 2x2 sanity: treated pre vs post
tt = sub[sub.treated]
print(f"treated rejection: pre={100*tt[tt.post==0].rej.mean():.1f}%  post={100*tt[tt.post==1].rej.mean():.1f}%")

# ---- TWFE DiD: rej ~ post + repo FE + month FE, cluster by repo ----
m = smf.ols("rej ~ post + C(repo_id) + C(mi)", data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["repo_id"]})
b, se = m.params["post"], m.bse["post"]
print(f"\nDiD effect of instruction-file adoption on rejection: {b*100:+.2f} pts "
      f"(95% CI [{(b-1.96*se)*100:+.2f}, {(b+1.96*se)*100:+.2f}], p={m.pvalues['post']:.3f})")

# secondary outcomes
for out, col in [("log_churn", "log_churn"), ("has_test", None)]:
    sub["_y"] = sub[out].astype(float) if out=="log_churn" else sub["has_test"].astype(int)
    mm = smf.ols("_y ~ post + C(repo_id) + C(mi)", data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["repo_id"]})
    print(f"  effect on {out}: {mm.params['post']:+.3f} (p={mm.pvalues['post']:.3f})")

# ---- event study (treated only): rej ~ C(rel) + repo FE + month FE ----
ts = sub[sub.treated].copy()
ts["rel"] = (ts.mi - ts.ai).clip(-4, 4).astype(int)
ev = smf.ols("rej ~ C(rel, Treatment(reference=-1)) + C(repo_id) + C(mi)", data=ts).fit(
        cov_type="cluster", cov_kwds={"groups": ts["repo_id"]})
rels = [r for r in range(-4, 5) if r != -1]
pts = {r: (ev.params.get(f"C(rel, Treatment(reference=-1))[T.{r}]", 0.0),
           ev.bse.get(f"C(rel, Treatment(reference=-1))[T.{r}]", 0.0)) for r in rels}
xs = [-4,-3,-2,-1,0,1,2,3,4]
ys = [0 if r==-1 else pts[r][0]*100 for r in xs]
es = [0 if r==-1 else pts[r][1]*100 for r in xs]
plt.figure(figsize=(9,5))
plt.errorbar(xs, ys, yerr=[1.96*e for e in es], fmt="o-", color="#9467bd", capsize=3)
plt.axvline(-0.5, ls="--", color="k", alpha=.5); plt.axhline(0, color="grey", lw=.8)
plt.xlabel("Months relative to instruction-file adoption"); plt.ylabel("Δ rejection rate vs month -1 (pts)")
plt.title("NC1 event study: agent bug-fix rejection around instruction-file adoption\n(flat pre-trend = parallel-trends OK)")
plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(os.path.join(FIG, "nc1_eventstudy.png"), dpi=130); plt.close()

# ---- placebo: fake adoption 4 months earlier, pre-period only (should be ~0) ----
pl = sub[sub.treated].copy()
pl = pl[pl.mi < pl.ai]                       # restrict to true pre-period
pl["fake_post"] = (pl.mi >= (pl.ai - 4)).astype(int)
if pl["fake_post"].nunique() > 1:
    pm = smf.ols("rej ~ fake_post + C(repo_id) + C(mi)", data=pl).fit(
            cov_type="cluster", cov_kwds={"groups": pl["repo_id"]})
    print(f"placebo (fake adoption -4mo, pre-period only): {pm.params['fake_post']*100:+.2f} pts "
          f"(p={pm.pvalues['fake_post']:.3f})  [want ~0]")
print("saved figures/nc1_eventstudy.png")
