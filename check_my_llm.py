import pandas as pd, glob, os

d = r"C:\Users\Mahmoudabujadallah\Downloads\Telegram Desktop"
files = glob.glob(os.path.join(d, "*_pr_task_type.csv"))
print("found CSVs:", [os.path.basename(f) for f in files])
parts = []
for f in files:
    df = pd.read_csv(f)
    parts.append(df)
    print(f"  {os.path.basename(f)}: {len(df):,} rows, cols={list(df.columns)}")
llm = pd.concat(parts, ignore_index=True)
print(f"\nTOTAL LLM-labelled agent PRs: {len(llm):,}")
print("type distribution:"); print(llm["type"].value_counts().to_string())
if "confidence" in llm: print("median confidence:", llm["confidence"].median())

# do these ids match our dataset?
our = pd.read_parquet(r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes\data\fix_prs_only.parquet",
                      columns=["id","state","merged_at","is_agent","agent"])
mna = our["merged_at"].isna() | our["merged_at"].astype(str).isin(["","NaT","None","nan"])
our["merged"]=~mna; our["closed"]=our["state"]=="closed"; our["rej"]=our["closed"]&mna
our_ids = set(our["id"])
llm_fix = llm[llm["type"]=="fix"]
print(f"\nLLM 'fix' PRs: {len(llm_fix):,}")
inter = llm_fix[llm_fix["id"].isin(our_ids)]
print(f"  LLM-fix ids also in our regex-fix table: {len(inter):,} ({100*len(inter)/max(len(llm_fix),1):.0f}%)")
# rejection on LLM-fix that we can match to outcomes
j = our[our.id.isin(set(llm_fix['id']))]
cl = j[j.closed]
if len(cl): print(f"  rejection on matched LLM-fix (closed): {100*cl['rej'].mean():.1f}% (n={len(cl):,})")
print(f"\n(reference) regex-fix agent rejection: {100*our[our.is_agent & our.closed]['rej'].mean():.1f}%")
