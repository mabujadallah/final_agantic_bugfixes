"""
NC1 step 1 — build repo-level instruction-file adoption dates (dataset-only, robust via DuckDB).
DuckDB reads ONLY the needed columns of the remote Parquets over HTTP (column projection + its own
retrying HTTP layer), pushes the instruction-file regex down to the scan, and returns the tiny
matching pr_id set. Caches data/repo_adoption.parquet + data/instr_pr_ids.parquet.
"""
import os, time, duckdb

OUT = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes\data"
B = "https://huggingface.co/datasets/mabujadallah/GitHub-Agentic-PR-Dataset/resolve/main"
DET = f"{B}/pr_commit_details.parquet"
APR = f"{B}/all_pull_requests.parquet"
# lowercased instruction-file pattern (RE2); we match on lower(filename)
PAT = (r"copilot-instructions\.md|(^|/)claude\.md|(^|/)agents\.md|(^|/)gemini\.md|"
       r"\.cursorrules|\.cursor/rules|\.windsurfrules|\.github/instructions/|"
       r"(^|/)conventions\.md|\.aider")

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("SET http_keep_alive=false; SET http_retries=5; SET http_timeout=120000;")
con.execute(f"SET threads={os.cpu_count() or 4};")

t0 = time.time()
print("scanning pr_commit_details (filename regex pushdown) for instruction-file PRs ...")
con.execute(f"""
CREATE TEMP TABLE ins AS
SELECT DISTINCT pr_id
FROM read_parquet('{DET}')
WHERE regexp_matches(lower(filename), '{PAT}')
""")
n_ins = con.execute("SELECT count(*) FROM ins").fetchone()[0]
print(f"  instruction-touching PRs: {n_ins:,}  ({time.time()-t0:.0f}s)")
con.execute(f"COPY ins TO '{os.path.join(OUT,'instr_pr_ids.parquet')}' (FORMAT parquet);")

print("dating adoptions via all_pull_requests ...")
adopt = con.execute(f"""
SELECT a.repo_id AS repo_id, substr(min(a.created_at),1,7) AS adopt_month
FROM read_parquet('{APR}') a
WHERE a.id IN (SELECT pr_id FROM ins)
GROUP BY a.repo_id
""").fetchdf()
adopt.to_parquet(os.path.join(OUT, "repo_adoption.parquet"), index=False)
print(f"repos that ever adopt an instruction file: {len(adopt):,}  ({time.time()-t0:.0f}s)")
print(adopt["adopt_month"].value_counts().sort_index().to_string())
print("saved data/repo_adoption.parquet, data/instr_pr_ids.parquet")
