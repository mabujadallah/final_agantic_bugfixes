import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

fs = HfFileSystem()
base = "datasets/mabujadallah/GitHub-Agentic-PR-Dataset"

files = [
    "fix_prs_only.parquet",
    "fix_classified_prs.parquet",
    "fix_pr_commits.parquet",
    "fix_pr_commit_details.parquet",
    "agent_pull_requests.parquet",
    "all_pull_requests.parquet",
    "human_pull_requests.parquet",
    "pr_commits.parquet",
    "pr_commit_details.parquet",
]

for f in files:
    path = f"{base}/{f}"
    try:
        with fs.open(path, "rb") as fo:
            pf = pq.ParquetFile(fo)
            schema = pf.schema_arrow
            n = pf.metadata.num_rows
            print(f"\n===== {f}  | rows={n:,} | cols={len(schema)} =====")
            for field in schema:
                print(f"  {field.name}: {field.type}")
    except Exception as e:
        print(f"\n===== {f}  | ERROR: {type(e).__name__}: {e}")
