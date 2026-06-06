"""
RQ2b: model-substitution / relay  (agent A's fix rejected -> a later fix on the SAME issue merged)
RQ4:  fix durability  (do merged fixes get reverted?)
Dataset-only. Outputs: figures/rq2b_*.png, figures/rq4_*.png  and appends to RESULTS.md
"""
import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes"
DATA = os.path.join(BASE, "data"); FIG = os.path.join(BASE, "figures")
AGENTS = ["Claude_Code", "Cursor", "Copilot", "Devin"]
RESCUERS = AGENTS + ["human"]
out = ["\n\n---\n"]
def w(s=""): out.append(s); print(s)

# ---------- load PRs ----------
prs = pd.read_parquet(os.path.join(DATA, "fix_prs_only.parquet"),
        columns=["id","number","title","body","state","created_at","merged_at","repo_id","is_agent","agent"])
mna = prs["merged_at"].isna() | prs["merged_at"].astype(str).isin(["", "NaT", "None"])
prs["merged"]   = ~mna
prs["rejected"] = (prs["state"] == "closed") & ~prs["merged"]
prs["label"]    = np.where(prs["is_agent"], prs["agent"], "human")

# ================= RQ2b: relay / model substitution =================
w("## RQ2b — Model-substitution / relay (does switching agent rescue a failed fix?)")
# extract issue numbers this PR claims to close (GitHub closing keywords)
KEY = re.compile(r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s+#(\d+)", re.I)
def issue_refs(title, body):
    txt = f"{title or ''}\n{body or ''}"
    return set(int(m) for m in KEY.findall(txt))
prs["refs"] = [issue_refs(t, b) for t, b in zip(prs["title"], prs["body"])]
cov = (prs["refs"].map(len) > 0).mean() * 100

# explode to (repo_id, issue) -> attempts
recs = []
for r in prs.itertuples(index=False):
    for iss in r.refs:
        recs.append((r.repo_id, iss, r.created_at, r.label, r.is_agent, r.merged, r.rejected))
ex = pd.DataFrame(recs, columns=["repo_id","issue","created_at","label","is_agent","merged","rejected"])
ex = ex.sort_values("created_at")
g = ex.groupby(["repo_id","issue"])
sizes = g.size()
multi = sizes[sizes >= 2].index            # issues with >=2 fix attempts
w(f"- Coverage: {cov:.1f}% of fix PRs reference an issue via a closing keyword; "
  f"{len(multi):,} issues received >=2 fix attempts (multi-attempt issues).")

# focus: issues whose FIRST attempt is an agent fix that was REJECTED
same=diff=human=norec=0
matrix = pd.DataFrame(0, index=AGENTS, columns=RESCUERS)  # failed agent -> rescuer
n_failed_first = 0
for key in multi:
    sub = g.get_group(key)
    first = sub.iloc[0]
    if not (first.is_agent and first.rejected):
        continue
    n_failed_first += 1
    later = sub.iloc[1:]
    win = later[later["merged"]]
    if len(win) == 0:
        norec += 1; continue
    rescuer = win.iloc[0]["label"]          # first merged attempt after the failure
    if rescuer == first.label:    same += 1
    elif rescuer == "human":      human += 1
    else:                         diff += 1
    if first.label in AGENTS:
        matrix.loc[first.label, rescuer] += 1

rec = same + diff + human
w(f"- Issues whose FIRST fix was an agent and got rejected: **{n_failed_first:,}**.")
if n_failed_first:
    w(f"- Of those, **{100*rec/n_failed_first:.1f}%** were later fixed by a merged PR (recovered):")
    w(f"    - different agent (**substitution**): {diff:,} ({100*diff/n_failed_first:.1f}%)")
    w(f"    - same agent (retry): {same:,} ({100*same/n_failed_first:.1f}%)")
    w(f"    - human takeover: {human:,} ({100*human/n_failed_first:.1f}%)")
    w(f"    - never recovered: {norec:,} ({100*norec/n_failed_first:.1f}%)")

# fig: recovery breakdown
plt.figure(figsize=(8,5))
labels = ["Different agent\n(substitution)","Same agent\n(retry)","Human\ntakeover","Never\nrecovered"]
vals = [diff, same, human, norec]
cols = ["#2ca02c","#1f77b4","#ff7f0e","#d62728"]
bars = plt.bar(labels, vals, color=cols)
for b,v in zip(bars,vals):
    pct = 100*v/n_failed_first if n_failed_first else 0
    plt.text(b.get_x()+b.get_width()/2, v, f"{v}\n({pct:.0f}%)", ha="center", va="bottom")
plt.ylabel("# issues (first agent fix rejected)")
plt.title("RQ2b: What rescues a rejected agent fix on the same issue?")
plt.tight_layout(); plt.savefig(os.path.join(FIG,"rq2b_recovery_breakdown.png"), dpi=130); plt.close()

# fig: substitution matrix (row-normalized %)
mn = matrix.div(matrix.sum(axis=1).replace(0,np.nan), axis=0)*100
plt.figure(figsize=(7,5))
plt.imshow(mn.values, cmap="Greens", vmin=0, vmax=100, aspect="auto")
plt.xticks(range(len(RESCUERS)), RESCUERS, rotation=30, ha="right"); plt.yticks(range(len(AGENTS)), AGENTS)
for i in range(len(AGENTS)):
    for j in range(len(RESCUERS)):
        c = matrix.values[i,j]
        if c: plt.text(j, i, f"{c}", ha="center", va="center", color="black", fontsize=9)
plt.colorbar(label="% of that agent's recovered failures"); plt.xlabel("Rescued by"); plt.ylabel("Failed agent")
plt.title("RQ2b: Who rescues whom (counts; color=row %)")
plt.tight_layout(); plt.savefig(os.path.join(FIG,"rq2b_substitution_matrix.png"), dpi=130); plt.close()
w("- Figures: `figures/rq2b_recovery_breakdown.png`, `figures/rq2b_substitution_matrix.png`")
w("- How: parse GitHub closing keywords (`fixes/closes/resolves #N`) from title+body; group fix PRs by "
  "(repo, issue); for issues whose earliest fix attempt was an agent PR that was rejected, the first "
  "later **merged** attempt is the 'rescuer' (same agent = retry, other agent = substitution, human = takeover).\n")

# ================= RQ4: durability / reverts =================
w("## RQ4 — Fix durability (do merged fixes get reverted?)")
fix_ids = set(prs["id"].tolist())
com = pd.read_parquet(os.path.join(DATA, "pr_commits.parquet"), columns=["sha","pr_id","message"])
# map: short-sha -> pr_id for commits that belong to a FIX PR
fixcom = com[com["pr_id"].isin(fix_ids)]
sha_to_pr = dict(zip(fixcom["sha"].str.slice(0,12), fixcom["pr_id"]))
w(f"- Indexed {len(fixcom):,} commits across {fixcom['pr_id'].nunique():,} fix PRs.")

REV = re.compile(r"reverts commit ([0-9a-f]{7,40})", re.I)
reverted_pr = {}   # original fix pr_id -> reverting pr_id
for pr_id, msg in zip(com["pr_id"], com["message"]):
    if not isinstance(msg, str) or "revert" not in msg.lower():
        continue
    for ref in REV.findall(msg):
        orig = sha_to_pr.get(ref[:12])
        if orig is not None and orig != pr_id:   # external revert of a fix PR's commit
            reverted_pr.setdefault(orig, pr_id)
rev_ids = set(reverted_pr)
w(f"- Found {len(rev_ids):,} fix PRs with at least one commit reverted (via 'reverts commit <sha>').")

merged = prs[prs["merged"]].copy()
merged["reverted"] = merged["id"].isin(rev_ids)
def rate(df):
    return (100*df["reverted"].mean(), int(df["reverted"].sum()), len(df))
ag = merged[merged["is_agent"]]; hu = merged[~merged["is_agent"]]
ra, na, ta = rate(ag); rh, nh, th = rate(hu)
w(f"- **Agent merged fixes reverted: {na:,}/{ta:,} = {ra:.3f}%** | Human: {nh:,}/{th:,} = {rh:.3f}%.")
per_agent = {a: rate(merged[merged["agent"]==a]) for a in AGENTS}
for a,(r,n,t) in per_agent.items():
    w(f"    - {a}: {n:,}/{t:,} = {r:.3f}%")

# fig: revert rate agent vs human + per agent
plt.figure(figsize=(8,5))
labels = ["Human","Agent (all)"] + AGENTS
vals = [rh, ra] + [per_agent[a][0] for a in AGENTS]
cols = ["#1f77b4","#d62728"] + ["#9467bd","#2ca02c","#ff7f0e","#8c564b"]
bars = plt.bar(labels, vals, color=cols)
for b,v in zip(bars,vals): plt.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
plt.ylabel("Revert rate (% of merged fixes)"); plt.title("RQ4: Merged bug-fix revert rate")
plt.xticks(rotation=20, ha="right"); plt.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"rq4_revert_rate.png"), dpi=130); plt.close()
w("- Figure: `figures/rq4_revert_rate.png`")
w("- How: collect commit SHAs of all fix PRs; scan ALL PR commit messages for `reverts commit <sha>`; "
  "a merged fix PR is 'reverted' if one of its commits is referenced by a revert in a different PR. "
  "Revert rate = reverted / merged, per group. (Lower bound: only reverts done via PRs are visible.)\n")

with open(os.path.join(BASE,"RESULTS.md"), "a", encoding="utf-8") as f:
    f.write("\n".join(out))
print("\n[done] appended to RESULTS.md and wrote rq2b_*/rq4_* figures")
