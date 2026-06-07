import pandas as pd

# Replicate the paper exactly
pr = pd.read_parquet("hf://datasets/hao-li/AIDev/pull_request.parquet",
                     columns=["id","agent","state","merged_at","html_url"])
tt = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_task_type.parquet", columns=["id","type"])
fix_ids = set(tt[tt.type == "fix"]["id"])
m = pr[pr.id.isin(fix_ids) & (pr.state == "closed") & (pr.agent != "OpenAI_Codex")].copy()
m["rej"] = m["merged_at"].isna()
print(f"PAPER replicate: closed fix PRs={len(m):,}  rejection={100*m.rej.mean():.1f}%  (expect ~46.4%)")

# Same PRs by GitHub URL, OUR fresh outcomes
our = pd.read_parquet(r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes\data\fix_prs_only.parquet",
                      columns=["html_url","state","merged_at"])
mna = our["merged_at"].isna() | our["merged_at"].astype(str).isin(["","NaT","None","nan"])
our["merged"] = ~mna; our["closed"] = our["state"]=="closed"
our_by_url = our.set_index("html_url")
paper_urls = m["html_url"]
found = paper_urls[paper_urls.isin(our_by_url.index)]
print(f"\nSAME PRs by URL found in OUR data: {len(found):,}/{len(m):,}")
sub = our_by_url.loc[found.values]
print(f"  our outcomes: merged={int(sub.merged.sum()):,}  open={int((sub.state=='open').sum()):,}  closed={int(sub.closed.sum()):,}")
cl = sub[sub.closed]
rej = cl["merged"].eq(False)
print(f"  rejection (closed) in OUR data = {100*rej.mean():.1f}%   (paper said 46.4% for these)")
