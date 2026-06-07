import pandas as pd

LLM = r"C:\Users\Mahmoudabujadallah\final_replication\GitHub-Agentic-PR-Dataset\pr_task_type.parquet"
t = pd.read_parquet(LLM)
print("pr_task_type.parquet columns:", list(t.columns), "| rows:", f"{len(t):,}")
if "type" in t:
    print("\ntype distribution:"); print(t["type"].value_counts().to_string())
if "confidence" in t:
    print("\nmedian confidence:", t["confidence"].median())
if "agent" in t:
    print("\nagent distribution:"); print(t["agent"].value_counts().to_string())

# join LLM fix labels to our outcomes (fix_prs_only has id, state, merged_at, is_agent)
our = pd.read_parquet(r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes\data\fix_prs_only.parquet",
                      columns=["id","state","merged_at","is_agent"])
mna = our["merged_at"].isna() | our["merged_at"].astype(str).isin(["","NaT","None","nan"])
our["merged"] = ~mna; our["closed"] = our["state"]=="closed"; our["rej"] = our["closed"] & mna

llm_fix_ids = set(t[t["type"]=="fix"]["id"]) if "type" in t else set()
print(f"\nLLM 'fix' PRs: {len(llm_fix_ids):,}")
ov = our[our.is_agent & our.id.isin(llm_fix_ids)]
print(f"  of those present in our agent fix_prs_only (regex-fix) set: {len(ov):,}")
if len(ov):
    cl = ov[ov.closed]
    print(f"  rejection on LLM-fix ∩ regex-fix (agent, closed): {100*cl['rej'].mean():.1f}% (n={len(cl):,})")
# overall regex-fix agent rejection for reference
ag = our[our.is_agent]; cla = ag[ag.closed]
print(f"\nfor reference, regex-fix agent rejection: {100*cla['rej'].mean():.1f}% (n={len(cla):,})")
