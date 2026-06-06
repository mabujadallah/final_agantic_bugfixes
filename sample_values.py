import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from collections import Counter

fs = HfFileSystem()
base = "datasets/mabujadallah/GitHub-Agentic-PR-Dataset"

path = f"{base}/fix_prs_only.parquet"
cols = ["agent", "is_agent", "type", "source", "state", "merged_at", "created_at"]
with fs.open(path, "rb") as fo:
    pf = pq.ParquetFile(fo)
    print("num_row_groups:", pf.num_row_groups, "| total rows:", pf.metadata.num_rows)
    tbl = pf.read_row_group(0, columns=cols)

import pyarrow.compute as pc
d = tbl.to_pydict()
n = len(d["agent"])
print("sample rows from row-group 0:", n)

def vc(col, top=15):
    c = Counter(d[col])
    print(f"\n-- {col} (distinct={len(c)}) --")
    for k, v in c.most_common(top):
        print(f"   {repr(k)}: {v}")

vc("agent")
vc("is_agent")
vc("type")
vc("source")
vc("state")

# rejection signal: state==closed and merged_at is null/empty
merged = d["merged_at"]
state = d["state"]
rej = sum(1 for s, m in zip(state, merged) if (s == "closed") and (m in (None, "", "NaT")))
mer = sum(1 for m in merged if m not in (None, "", "NaT"))
print(f"\nmerged (merged_at present): {mer} / {n}")
print(f"closed & not merged (rejected): {rej} / {n}")

print("\nexample created_at values:", d["created_at"][:5])
print("example merged_at values:", merged[:5])
