"""
NC1 step 1 (LOCAL, robust fallback) — repo instruction-file adoption dates from LOCAL fix-PR file
changes only (no network). Lower bound: only sees instruction files edited *inside a fix PR*, so it
misses repos that added them via non-fix PRs and dates adoption late. Stronger all-PR dating needs an
HF token to scan pr_commit_details (78 GB) — blocked by HTTP 429 unauthenticated.
"""
import os, re
import pandas as pd

D = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes\data"
BASE = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes"
INSTR = re.compile(
    r"(copilot-instructions\.md|(^|/)CLAUDE\.md|(^|/)AGENTS\.md|(^|/)GEMINI\.md|"
    r"\.cursorrules|\.cursor/rules|\.windsurfrules|\.github/instructions/|"
    r"(^|/)CONVENTIONS\.md|\.aider)", re.I)

det = pd.read_parquet(os.path.join(D, "fix_pr_commit_details.parquet"), columns=["pr_id","filename"])
hit = set(det.loc[det["filename"].fillna("").map(lambda f: bool(INSTR.search(f))), "pr_id"])

clean = pd.read_parquet(os.path.join(BASE, "analysis_clean.parquet"), columns=["id","repo_id","month"])
ins = clean[clean.id.isin(hit)]
adopt = ins.groupby("repo_id")["month"].min().rename("adopt_month").reset_index()
adopt.to_parquet(os.path.join(D, "repo_adoption.parquet"), index=False)
print(f"instruction-touch fix PRs: {len(hit):,} | repos with adoption (local proxy): {len(adopt):,}")
print(adopt["adopt_month"].value_counts().sort_index().to_string())
print("saved data/repo_adoption.parquet (LOCAL proxy)")
