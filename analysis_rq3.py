"""RQ3: do developers increasingly pair agentic bug-fixes with agent-instruction files, over time?
Dataset-only (fix-PR file changes). Saves figures/rq3_instructions.png."""
import re, os
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DATA = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes\data"
FIG = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes\figures"

det = pd.read_parquet(os.path.join(DATA, "fix_pr_commit_details.parquet"), columns=["pr_id","filename"])
INSTR = re.compile(
    r"(copilot-instructions\.md|(^|/)CLAUDE\.md|(^|/)AGENTS\.md|(^|/)GEMINI\.md|"
    r"\.cursorrules|\.cursor/rules|\.windsurfrules|\.github/instructions/|"
    r"(^|/)CONVENTIONS\.md|\.aider)", re.I)
det["instr"] = det["filename"].fillna("").map(lambda f: bool(INSTR.search(f)))
hit = set(det.loc[det.instr, "pr_id"])

prs = pd.read_parquet(os.path.join(DATA, "fix_prs_only.parquet"),
                      columns=["id","repo_id","created_at","is_agent"])
ag = prs[prs.is_agent].copy()
ag["touch"] = ag["id"].isin(hit)
ag["q"] = pd.PeriodIndex(pd.to_datetime(ag["created_at"], utc=True, errors="coerce"), freq="Q").astype(str)
share = ag.groupby("q")["touch"].mean()*100
cnt = ag.groupby("q")["touch"].sum()

print(f"agent fix PRs touching an instruction file: {int(ag.touch.sum()):,}/{len(ag):,} "
      f"({100*ag.touch.mean():.2f}%) across {ag.loc[ag.touch,'repo_id'].nunique():,} repos")
print(share.round(3).to_string())

qs = list(share.index)
plt.figure(figsize=(9,5))
b = plt.bar(qs, share.values, color="#9467bd")
for bar,v,c in zip(b, share.values, cnt.values):
    plt.text(bar.get_x()+bar.get_width()/2, v, f"{v:.2f}%\n(n={int(c)})", ha="center", va="bottom", fontsize=8)
plt.ylabel("% of agent bug-fix PRs that also edit an instruction file")
plt.xlabel("Quarter (PR created)")
plt.title("RQ3: developers increasingly pair bug-fixes with agent-instruction files\n(CLAUDE.md, copilot-instructions.md, AGENTS.md, .cursorrules, ...)")
plt.ylim(0, max(share.values)*1.25); plt.grid(axis="y", alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(FIG, "rq3_instructions.png"), dpi=130); plt.close()
print("saved figures/rq3_instructions.png")
