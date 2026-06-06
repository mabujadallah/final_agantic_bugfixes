"""Fetch the bug-fix Parquet tables used by the analysis from Hugging Face.

These files are large (~1.1 GB total) and are NOT stored in git. Run this once:
    python download_data.py
"""
import os
from huggingface_hub import hf_hub_download

REPO = "mabujadallah/GitHub-Agentic-PR-Dataset"
FILES = ["fix_prs_only.parquet", "fix_pr_commit_details.parquet", "pr_commits.parquet"]
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

os.makedirs(DEST, exist_ok=True)
for f in FILES:
    path = hf_hub_download(repo_id=REPO, filename=f, repo_type="dataset", local_dir=DEST)
    print("downloaded", path)
print("done")
