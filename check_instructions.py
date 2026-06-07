"""RQ3 viability: do fix PRs touch agent-instruction files, and is there an over-time signal?"""
import re, pandas as pd, os
DATA = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes\data"

det = pd.read_parquet(os.path.join(DATA, "fix_pr_commit_details.parquet"), columns=["pr_id","filename"])
INSTR = re.compile(
    r"(copilot-instructions\.md|(^|/)CLAUDE\.md|(^|/)AGENTS?\.md|(^|/)GEMINI\.md|"
    r"\.cursorrules|\.cursor/rules|\.windsurfrules|\.github/instructions/|"
    r"(^|/)CONVENTIONS\.md|\.aider)", re.I)
det["instr"] = det["filename"].fillna("").map(lambda f: bool(INSTR.search(f)))
hit_prs = set(det.loc[det.instr, "pr_id"])
print(f"fix-PR file rows: {len(det):,}")
print(f"fix PRs that touch an agent-instruction file: {len(hit_prs):,}")

# join to PR metadata for repo/time/outcome
prs = pd.read_parquet(os.path.join(DATA, "fix_prs_only.parquet"),
                      columns=["id","repo_id","created_at","state","merged_at","is_agent"])
prs["month"] = prs["created_at"].str.slice(0,7)
prs["touch_instr"] = prs["id"].isin(hit_prs)
ag = prs[prs.is_agent]
print(f"\nagent fix PRs touching instructions: {int(ag.touch_instr.sum()):,} / {len(ag):,} ({100*ag.touch_instr.mean():.2f}%)")
print(f"distinct repos with such a fix PR: {ag.loc[ag.touch_instr,'repo_id'].nunique():,}")
# over time (share of agent fix PRs touching instructions, by quarter)
ag = ag.copy(); ag["q"] = pd.PeriodIndex(pd.to_datetime(ag["created_at"], utc=True, errors="coerce"), freq="Q").astype(str)
share = ag.groupby("q")["touch_instr"].mean()*100
print("\nshare of agent fix PRs touching an instruction file, by quarter:")
print(share.round(3).to_string())

# show some example filenames matched
ex = det.loc[det.instr, "filename"].value_counts().head(12)
print("\ntop matched instruction filenames:")
print(ex.to_string())
