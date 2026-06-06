import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from collections import defaultdict

fs = HfFileSystem()
base = "datasets/mabujadallah/GitHub-Agentic-PR-Dataset"
path = f"{base}/fix_prs_only.parquet"

cols = ["created_at", "agent", "is_agent", "state", "merged_at"]
with fs.open(path, "rb") as fo:
    tbl = pq.ParquetFile(fo).read_row_group(0, columns=cols)
d = tbl.to_pydict()

created = d["created_at"]
agent = d["agent"]
state = d["state"]
merged = d["merged_at"]

months = [c[:7] for c in created if c]  # YYYY-MM
print("date min:", min(created), "| max:", max(created))
print("distinct months:", sorted(set(months)))

# monthly agent-fix counts + rejection rate (agents only)
counts = defaultdict(int)
rej = defaultdict(int)
clo = defaultdict(int)
for c, a, s, m in zip(created, agent, state, merged):
    if a == "human":
        continue
    mo = c[:7]
    counts[(mo, a)] += 1
    if s == "closed":
        clo[mo] += 1
        if m in (None, "", "NaT"):
            rej[mo] += 1

print("\n-- agent fix PRs per month (by agent) --")
for mo in sorted(set(m for m, _ in counts)):
    row = {a: counts[(mo, a)] for a in ["Claude_Code", "Cursor", "Copilot", "Devin"]}
    print(f"  {mo}: {row}")

print("\n-- monthly agent rejection rate (closed&unmerged / closed) --")
for mo in sorted(clo):
    r = 100 * rej[mo] / clo[mo] if clo[mo] else 0
    print(f"  {mo}: {rej[mo]}/{clo[mo]} = {r:.1f}%")
