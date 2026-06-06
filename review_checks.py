"""Reviewer validity probes for the notebook's claims."""
import os, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr

DATA = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes\data"
AGENTS = ["Claude_Code","Cursor","Copilot","Devin"]

prs = pd.read_parquet(os.path.join(DATA,"fix_prs_only.parquet"),
        columns=["id","state","created_at","merged_at","is_agent","agent"])
prs["month"] = prs["created_at"].str.slice(0,7)
prs["closed"] = prs["state"]=="closed"
mna = prs["merged_at"].isna() | prs["merged_at"].astype(str).isin(["","NaT","None"])
prs["merged"] = ~mna
prs["rejected"] = prs["closed"] & ~prs["merged"]
months = sorted(prs["month"].dropna().unique()); xi = np.arange(len(months))

print("="*70)
print("CHECK A — RQ1a: is the 'agents flat' aggregate hiding per-agent trends? (Simpson)")
for a in AGENTS:
    d = prs[(prs.agent==a) & prs.closed]
    r = d.groupby("month")["rejected"].mean().reindex(months)*100
    valid = r.dropna()
    rho,p = spearmanr(np.arange(len(valid)), valid.values)
    print(f"  {a:12s}: {valid.iloc[0]:5.1f}% -> {valid.iloc[-1]:5.1f}%  rho={rho:+.2f} p={p:.3f}")
agg = prs[prs.is_agent & prs.closed].groupby("month")["rejected"].mean().reindex(months)*100
rho,p = spearmanr(xi, agg.values)
print(f"  AGG (all agents): rho={rho:+.2f} p={p:.3f}  <- aggregate trend")

print("="*70)
print("CHECK B — RQ1b: churn coverage by group + median among COVERED only")
det = pd.read_parquet(os.path.join(DATA,"fix_pr_commit_details.parquet"),
        columns=["pr_id","additions","deletions"])
det["lines"] = det["additions"].fillna(0)+det["deletions"].fillna(0)
churn = det.groupby("pr_id")["lines"].sum()
prs["churn"] = prs["id"].map(churn)
for g,mask in [("Agent",prs.is_agent),("Human",~prs.is_agent)]:
    sub = prs[mask]
    cov = sub["churn"].notna().mean()*100
    med_all = sub["churn"].fillna(0).median()
    med_cov = sub["churn"].dropna().median()
    print(f"  {g}: coverage={cov:4.1f}%  median(fill0)={med_all:5.0f}  median(covered only)={med_cov:5.0f}")
print("  NOTE: churn = sum over ALL commit-file rows -> multi-commit PRs counted cumulatively (not net diff).")

print("="*70)
print("CHECK C — RQ4: is 'agents reverted less' an age (exposure) confound?")
com = pd.read_parquet(os.path.join(DATA,"pr_commits.parquet"), columns=["sha","pr_id","message"])
fix_ids = set(prs["id"])
fixcom = com[com["pr_id"].isin(fix_ids)]
sha_to_pr = dict(zip(fixcom["sha"].str.slice(0,12), fixcom["pr_id"]))
REV = re.compile(r"reverts commit ([0-9a-f]{7,40})", re.I)
rev_ids=set()
for pr_id,msg in zip(com["pr_id"],com["message"]):
    if not isinstance(msg,str) or "revert" not in msg.lower(): continue
    for ref in REV.findall(msg):
        o = sha_to_pr.get(ref[:12])
        if o is not None and o!=pr_id: rev_ids.add(o)

m = prs[prs.merged].copy()
m["reverted"] = m["id"].isin(rev_ids)
END = pd.Timestamp("2026-02-28", tz="UTC")
m["merged_dt"] = pd.to_datetime(m["merged_at"], utc=True, errors="coerce")
m["exposure_days"] = (END - m["merged_dt"]).dt.days
print(f"  median exposure(days since merge): Agent={m[m.is_agent].exposure_days.median():.0f}  "
      f"Human={m[~m.is_agent].exposure_days.median():.0f}")
def rate(df):
    return f"{100*df.reverted.mean():.3f}% ({int(df.reverted.sum())}/{len(df)})"
print(f"  OVERALL revert rate:  Agent={rate(m[m.is_agent])}  Human={rate(m[~m.is_agent])}")
# fixed cohort: merged in 2025-H1 (>=8 months exposure for both)
coh = m[(m.merged_dt>=pd.Timestamp('2025-01-01',tz='UTC')) & (m.merged_dt<pd.Timestamp('2025-07-01',tz='UTC'))]
print(f"  COHORT merged 2025-H1: Agent={rate(coh[coh.is_agent])}  Human={rate(coh[~coh.is_agent])}")
print("="*70)
