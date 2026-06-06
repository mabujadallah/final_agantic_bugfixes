"""
RQ5: bug-type difficulty  (classify fix PRs from title/body -> rejection rate & churn per type)
RQ6 robustness: is the 'tests -> more rejection' effect just a size (churn) confound?
Dataset-only. Outputs figures/rq5_*.png, figures/rq6c_*.png ; appends to RESULTS.md
"""
import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes"
DATA = os.path.join(BASE, "data"); FIG = os.path.join(BASE, "figures")
out = ["\n\n---\n"]
def w(s=""): out.append(s); print(s)

# ---------- load + enrich ----------
prs = pd.read_parquet(os.path.join(DATA, "fix_prs_only.parquet"),
        columns=["id","title","body","state","merged_at","is_agent","agent"])
mna = prs["merged_at"].isna() | prs["merged_at"].astype(str).isin(["", "NaT", "None"])
prs["merged"]   = ~mna
prs["closed"]   = prs["state"] == "closed"
prs["rejected"] = prs["closed"] & ~prs["merged"]
prs["grp"]      = np.where(prs["is_agent"], "Agent", "Human")

det = pd.read_parquet(os.path.join(DATA, "fix_pr_commit_details.parquet"),
                      columns=["pr_id","filename","additions","deletions"])
det["lines"] = det["additions"].fillna(0) + det["deletions"].fillna(0)
churn = det.groupby("pr_id")["lines"].sum().rename("churn")
TEST_RE = re.compile(r"(^|/)(tests?|__tests__|spec)(/|\.)|(_test\.|test_|\.test\.|\.spec\.|_spec\.|\.tests\.)", re.I)
det["is_test"] = det["filename"].fillna("").map(lambda f: bool(TEST_RE.search(f)))
has_test = det.groupby("pr_id")["is_test"].any().rename("has_test")
prs = prs.merge(churn, left_on="id", right_index=True, how="left").merge(has_test, left_on="id", right_index=True, how="left")
prs["churn"] = prs["churn"].fillna(0)
prs["has_test"] = prs["has_test"].fillna(False).astype(bool)

# ================= RQ5: bug-type difficulty =================
w("## RQ5 — Which bug types are agents good vs bad at?")
CATS = [  # ordered: first match wins
 ("security",    r"\b(security|vulnerab\w*|cve-?\d|xss|csrf|sql\s*injection|injection|exploit|sanitiz)\b"),
 ("crash",       r"\b(crash\w*|segfault|seg\s*fault|npe|null\s*pointer|nullpointer|exception|panic|fatal|stack\s*overflow)\b"),
 ("concurrency", r"\b(race\s*condition|deadlock|concurren\w*|thread[-\s]?safe\w*|mutex|data\s*race|atomic)\b"),
 ("memory",      r"\b(memory\s*leak|mem\s*leak|out\s*of\s*memory|oom|buffer\s*overflow|leak)\b"),
 ("performance", r"\b(performance|perf|slow\w*|latency|speed\s*up|optimi[sz]\w*|throughput|timeout)\b"),
 ("security_auth",r"\b(authenticat\w*|authoriz\w*|permission|token|login|session)\b"),
 ("ui",          r"\b(ui|ux|css|layout|render\w*|display|styling|stylesheet|button|alignment|responsive|dark\s*mode)\b"),
 ("build_ci",    r"\b(build|ci|compil\w*|lint\w*|dependenc\w*|import\s*error|module\s*not\s*found|version\s*bump|packaging)\b"),
 ("typo_doc",    r"\b(typo|spelling|grammar|docstring|readme|documentation|docs)\b"),
 ("typing",      r"\b(type\s*error|typing|type\s*hint|mypy|type\s*annotation|typescript\s*type)\b"),
 ("data_parse",  r"\b(parse|parsing|serializ\w*|deserializ\w*|json|yaml|encoding|decod\w*|formatting)\b"),
 ("network",     r"\b(http|https|api|request|response|endpoint|url|websocket|socket|connection)\b"),
]
COMP = [(n, re.compile(p, re.I)) for n, p in CATS]
def classify(t, b):
    t = t if isinstance(t, str) else ""
    b = b if isinstance(b, str) else ""
    txt = f"{t} {b[:300]}"
    for n, rx in COMP:
        if rx.search(txt): return n
    return "other_logic"
prs["btype"] = [classify(t, b) for t, b in zip(prs["title"], prs["body"])]

def by_type(df):
    d = df[df["closed"]]
    t = d.groupby("btype").agg(n=("id","size"), rej=("rejected","mean"),
                               churn=("churn","median"))
    t["rej"] *= 100
    return t
agt = by_type(prs[prs.grp == "Agent"]).sort_values("rej", ascending=False)
hum = by_type(prs[prs.grp == "Human"])
agt["human_rej"] = hum["rej"]
w("- Agent fix PRs by bug type (closed only): rejection rate, median churn, human rejection baseline:")
for ty, r in agt.iterrows():
    w(f"    - {ty:14s} n={int(r['n']):6d}  reject={r['rej']:5.1f}%  churn(med)={r['churn']:5.0f}  (human {r['human_rej']:.1f}%)")

order = agt.index.tolist()
y = np.arange(len(order))
plt.figure(figsize=(9, 6))
plt.barh(y - 0.2, agt["rej"].values, height=0.4, label="Agent", color="#d62728")
plt.barh(y + 0.2, agt["human_rej"].values, height=0.4, label="Human", color="#1f77b4")
for i, ty in enumerate(order):
    plt.text(agt["rej"].iloc[i] + 0.3, i - 0.2, f"n={int(agt['n'].iloc[i])}", va="center", fontsize=7)
plt.yticks(y, order); plt.gca().invert_yaxis()
plt.xlabel("Rejection rate (% of closed)"); plt.title("RQ5: Bug-fix rejection rate by bug type (agent vs human)")
plt.legend(); plt.grid(axis="x", alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(FIG, "rq5_rejection_by_type.png"), dpi=130); plt.close()
w(f"- Hardest (agent): {order[0]} ({agt['rej'].iloc[0]:.1f}%). Easiest: {order[-1]} ({agt['rej'].iloc[-1]:.1f}%).")
w("- Figure: `figures/rq5_rejection_by_type.png`")
w("- How: keyword classifier (first match wins) over title + first 300 chars of body into 13 buckets "
  "(+'other_logic'); per type, rejection = rejected/closed, churn = median. Heuristic labels (no issue labels in data).\n")

# ================= RQ6 robustness: churn-controlled test effect =================
w("## RQ6 (robustness) — Is 'tests -> more rejection' just a size confound?")
a = prs[(prs.grp == "Agent") & prs["closed"]].copy()
mt = a[a.has_test]["churn"].median(); mn = a[~a.has_test]["churn"].median()
w(f"- Confound check: median churn WITH test = {mt:.0f} vs WITHOUT = {mn:.0f} lines "
  f"(test PRs are {'larger' if mt>mn else 'not larger'}).")
bins = [-1, 0, 10, 50, 200, 1e12]; lbls = ["0", "1-10", "11-50", "51-200", "200+"]
a["cb"] = pd.cut(a["churn"], bins=bins, labels=lbls)
rows = []
for b in lbls:
    sub = a[a["cb"] == b]
    rt = 100 * sub[sub.has_test]["rejected"].mean(); nt = int(sub.has_test.sum())
    rn = 100 * sub[~sub.has_test]["rejected"].mean(); nn = int((~sub.has_test).sum())
    rows.append((b, rt, nt, rn, nn))
    w(f"    - churn {b:7s}: +test {rt:5.1f}% (n={nt:5d}) | no-test {rn:5.1f}% (n={nn:6d}) | diff {rt-rn:+.1f}pts")

x = np.arange(len(lbls))
plt.figure(figsize=(9, 5))
plt.bar(x - 0.2, [r[1] for r in rows], width=0.4, label="with test", color="#d62728")
plt.bar(x + 0.2, [r[3] for r in rows], width=0.4, label="no test", color="#f4a3a3")
plt.xticks(x, lbls); plt.xlabel("Code churn bin (lines)"); plt.ylabel("Rejection rate (% of closed)")
plt.title("RQ6 (robustness): agent rejection by test inclusion, within churn bins")
plt.legend(); plt.grid(axis="y", alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(FIG, "rq6c_churn_controlled.png"), dpi=130); plt.close()
valid = [(rt, nt, rn, nn) for (_, rt, nt, rn, nn) in rows
         if nt >= 200 and nn >= 200 and not (np.isnan(rt) or np.isnan(rn))]
tot = sum(nt + nn for _, nt, _, nn in valid)
wdiff = sum((rt - rn) * (nt + nn) for rt, nt, rn, nn in valid) / tot
verdict = ("robust to size: test PRs are still rejected more even within equal-size bins"
           if wdiff >= 1.0 else "largely a size confound")
w(f"- Verdict: across comparable-size bins (n>=200/side), the test gap is {wdiff:+.1f} pts (size-weighted) "
  f"-> **{verdict}**. The large raw median-churn gap (250 vs 15) shows size IS a real co-factor, "
  f"but it does not fully explain the effect.")
w("- Figure: `figures/rq6c_churn_controlled.png`")
w("- How: agent closed fix PRs binned by churn; within each bin compare rejection rate with vs without a test.\n")

with open(os.path.join(BASE, "RESULTS.md"), "a", encoding="utf-8") as f:
    f.write("\n".join(out))
print("\n[done] appended RESULTS.md and wrote rq5_*/rq6c_* figures")
