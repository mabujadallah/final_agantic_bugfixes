"""
Re-run the agent-vs-human comparisons on a MATCHED set: only repos that contain
BOTH agent and human bug-fix PRs (within-project comparison). Prints full vs matched
and saves *_matched.png figures. Affects RQ1a, RQ1b, RQ4, RQ5, RQ6.
"""
import os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr

DATA = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes\data"
FIG = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes\figures"
AGENTS = ["Claude_Code","Cursor","Copilot","Devin"]

prs = pd.read_parquet(os.path.join(DATA,"fix_prs_only.parquet"),
        columns=["id","state","created_at","merged_at","repo_id","is_agent","agent","title","body"])
prs["month"] = prs["created_at"].str.slice(0,7)
prs["closed"] = prs["state"]=="closed"
mna = prs["merged_at"].isna() | prs["merged_at"].astype(str).isin(["","NaT","None"])
prs["merged"] = ~mna
prs["rejected"] = prs["closed"] & mna
prs["grp"] = np.where(prs["is_agent"],"Agent","Human")
months = sorted(prs["month"].dropna().unique()); xi = np.arange(len(months))

A = set(prs[prs.is_agent].repo_id); H = set(prs[~prs.is_agent].repo_id)
matched = A & H
prs["m"] = prs["repo_id"].isin(matched)
print(f"matched repos (have both): {len(matched):,}")
print(f"matched PRs: agent={int((prs.m & prs.is_agent).sum()):,}  human={int((prs.m & ~prs.is_agent).sum()):,}")
print("="*72)

# churn + test (needed for RQ1b/RQ5/RQ6)
det = pd.read_parquet(os.path.join(DATA,"fix_pr_commit_details.parquet"),
        columns=["pr_id","filename","additions","deletions"])
det["lines"] = det["additions"].fillna(0)+det["deletions"].fillna(0)
prs["churn"] = prs["id"].map(det.groupby("pr_id")["lines"].sum())
TEST_RE = re.compile(r"(^|/)(tests?|__tests__|spec)(/|\.)|(_test\.|test_|\.test\.|\.spec\.|_spec\.|\.tests\.)", re.I)
det["is_test"] = det["filename"].fillna("").map(lambda f: bool(TEST_RE.search(f)))
prs["has_test"] = prs["id"].map(det.groupby("pr_id")["is_test"].any()).fillna(False).astype(bool)

def rej_overall(df):
    d = df[df.closed]; return 100*d["rejected"].mean()

# ---------- RQ1a ----------
print("RQ1a  rejection rate (agent vs human)")
for tag,mask in [("FULL ",pd.Series(True,index=prs.index)),("MATCH",prs.m)]:
    sub = prs[mask]
    a = rej_overall(sub[sub.is_agent]); h = rej_overall(sub[~sub.is_agent])
    print(f"  {tag}: agent={a:5.1f}%  human={h:5.1f}%  gap={a-h:+.1f}pts")
# matched trend per group
sm = prs[prs.m]
def mrej(df): g=df[df.closed].groupby("month"); return (100*g["rejected"].mean()).reindex(months)
arej, hrej = mrej(sm[sm.is_agent]), mrej(sm[~sm.is_agent])
plt.figure(figsize=(10,5))
plt.plot(months,arej.values,"o-",label="Agent",color="#d62728")
plt.plot(months,hrej.values,"s-",label="Human",color="#1f77b4")
plt.ylabel("Rejection rate (% of closed)"); plt.xlabel("Month"); plt.xticks(rotation=45,ha="right")
plt.title("RQ1a (matched repos): rejection rate over time"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"rq1a_matched.png"),dpi=130); plt.close()
print("="*72)

# ---------- RQ1b ----------
print("RQ1b  median code churn (covered PRs only)")
for tag,mask in [("FULL ",pd.Series(True,index=prs.index)),("MATCH",prs.m)]:
    sub = prs[mask]
    a = sub[sub.is_agent]["churn"].median(); h = sub[~sub.is_agent]["churn"].median()
    print(f"  {tag}: agent={a:5.0f}  human={h:5.0f}")
print("="*72)

# ---------- RQ4 ----------
print("RQ4  revert rate of merged fixes")
com = pd.read_parquet(os.path.join(DATA,"pr_commits.parquet"), columns=["sha","pr_id","message"])
fixcom = com[com["pr_id"].isin(set(prs["id"]))]
sha_to_pr = dict(zip(fixcom["sha"].str.slice(0,12), fixcom["pr_id"]))
REV = re.compile(r"reverts commit ([0-9a-f]{7,40})", re.I)
rev_ids=set()
for pid,msg in zip(com["pr_id"],com["message"]):
    if not isinstance(msg,str) or "revert" not in msg.lower(): continue
    for ref in REV.findall(msg):
        o=sha_to_pr.get(ref[:12])
        if o is not None and o!=pid: rev_ids.add(o)
prs["reverted"] = prs["id"].isin(rev_ids)
for tag,mask in [("FULL ",pd.Series(True,index=prs.index)),("MATCH",prs.m)]:
    sub = prs[mask & prs.merged]
    a=100*sub[sub.is_agent]["reverted"].mean(); h=100*sub[~sub.is_agent]["reverted"].mean()
    print(f"  {tag}: agent={a:.3f}%  human={h:.3f}%")
mm = prs[prs.m & prs.merged]
labels=["Human","Agent"]+AGENTS
vals=[100*mm[~mm.is_agent]["reverted"].mean(),100*mm[mm.is_agent]["reverted"].mean()]+\
     [100*mm[mm.agent==a]["reverted"].mean() for a in AGENTS]
plt.figure(figsize=(8,5))
b=plt.bar(labels,vals,color=["#1f77b4","#d62728","#9467bd","#2ca02c","#ff7f0e","#8c564b"])
for bar,v in zip(b,vals): plt.text(bar.get_x()+bar.get_width()/2,v,f"{v:.2f}%",ha="center",va="bottom",fontsize=9)
plt.ylabel("Revert rate (% of merged)"); plt.title("RQ4 (matched repos): revert rate")
plt.xticks(rotation=20,ha="right"); plt.grid(axis="y",alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(FIG,"rq4_matched.png"),dpi=130); plt.close()
print("="*72)

# ---------- RQ5 ----------
print("RQ5  rejection by bug type (matched) -- agent vs human, top gaps")
CATS = [
 ("security",r"\b(security|vulnerab\w*|cve-?\d|xss|csrf|sql\s*injection|injection|exploit|sanitiz)\b"),
 ("crash",r"\b(crash\w*|segfault|seg\s*fault|npe|null\s*pointer|nullpointer|exception|panic|fatal|stack\s*overflow)\b"),
 ("concurrency",r"\b(race\s*condition|deadlock|concurren\w*|thread[-\s]?safe\w*|mutex|data\s*race|atomic)\b"),
 ("memory",r"\b(memory\s*leak|mem\s*leak|out\s*of\s*memory|oom|buffer\s*overflow|leak)\b"),
 ("performance",r"\b(performance|perf|slow\w*|latency|speed\s*up|optimi[sz]\w*|throughput|timeout)\b"),
 ("security_auth",r"\b(authenticat\w*|authoriz\w*|permission|token|login|session)\b"),
 ("ui",r"\b(ui|ux|css|layout|render\w*|display|styling|stylesheet|button|alignment|responsive|dark\s*mode)\b"),
 ("build_ci",r"\b(build|ci|compil\w*|lint\w*|dependenc\w*|import\s*error|module\s*not\s*found|version\s*bump|packaging)\b"),
 ("typo_doc",r"\b(typo|spelling|grammar|docstring|readme|documentation|docs)\b"),
 ("typing",r"\b(type\s*error|typing|type\s*hint|mypy|type\s*annotation|typescript\s*type)\b"),
 ("data_parse",r"\b(parse|parsing|serializ\w*|deserializ\w*|json|yaml|encoding|decod\w*|formatting)\b"),
 ("network",r"\b(http|https|api|request|response|endpoint|url|websocket|socket|connection)\b"),
]
COMP=[(n,re.compile(p,re.I)) for n,p in CATS]
def classify(t,b):
    t=t if isinstance(t,str) else ""; b=b if isinstance(b,str) else ""
    txt=f"{t} {b[:300]}"
    for n,rx in COMP:
        if rx.search(txt): return n
    return "other_logic"
sm = prs[prs.m].copy()
sm["btype"]=[classify(t,b) for t,b in zip(sm["title"],sm["body"])]
def by_type(df):
    d=df[df.closed]; t=d.groupby("btype").agg(n=("id","size"),rej=("rejected","mean")); t["rej"]*=100; return t
ag=by_type(sm[sm.is_agent]).sort_values("rej",ascending=False); hu=by_type(sm[~sm.is_agent])
ag["human_rej"]=hu["rej"]; ag["gap"]=ag["rej"]-ag["human_rej"]
print(ag[["n","rej","human_rej","gap"]].round(1).to_string())
order=ag.index.tolist(); y=np.arange(len(order))
plt.figure(figsize=(9,6))
plt.barh(y-0.2,ag["rej"].values,height=0.4,label="Agent",color="#d62728")
plt.barh(y+0.2,ag["human_rej"].values,height=0.4,label="Human",color="#1f77b4")
for i in range(len(order)): plt.text(ag["rej"].iloc[i]+0.3,i-0.2,f"n={int(ag['n'].iloc[i])}",va="center",fontsize=7)
plt.yticks(y,order); plt.gca().invert_yaxis(); plt.xlabel("Rejection rate (% of closed)")
plt.title("RQ5 (matched repos): rejection by bug type"); plt.legend(); plt.grid(axis="x",alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"rq5_matched.png"),dpi=130); plt.close()
print("="*72)

# ---------- RQ6 ----------
print("RQ6  test effect (matched): rejection with vs without a test")
for tag,mask in [("FULL ",pd.Series(True,index=prs.index)),("MATCH",prs.m)]:
    sub=prs[mask]
    for g in ["Agent","Human"]:
        s=sub[sub.grp==g]
        rt=rej_overall(s[s.has_test]); rn=rej_overall(s[~s.has_test])
        print(f"  {tag} {g}: +test {rt:5.1f}%  no-test {rn:5.1f}%  diff {rt-rn:+.1f}pts")
print("="*72)

# ---------- extra matched figures (RQ1b, RQ6, RQ6b, RQ6c) ----------
smc = prs[prs.m]
def medc(df): return df.groupby("month")["churn"].median().reindex(months)
ac, hc = medc(smc[smc.is_agent]), medc(smc[~smc.is_agent])
plt.figure(figsize=(10,5))
plt.plot(months, ac.values,"o-",label="Agent",color="#d62728")
plt.plot(months, hc.values,"s-",label="Human",color="#1f77b4")
plt.ylabel("Median code churn (lines +/-)"); plt.xlabel("Month"); plt.xticks(rotation=45,ha="right")
plt.title("RQ1b (matched repos): median code churn over time"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"rq1b_matched.png"),dpi=130); plt.close()

vals=[]
for g in ["Agent","Human"]:
    s=smc[smc.grp==g]; vals += [rej_overall(s[s.has_test]), rej_overall(s[~s.has_test])]
labels=["Agent\n+test","Agent\nno-test","Human\n+test","Human\nno-test"]
plt.figure(figsize=(8,5))
b=plt.bar(labels,vals,color=["#d62728","#f4a3a3","#1f77b4","#a3c4f4"])
for bar,v in zip(b,vals): plt.text(bar.get_x()+bar.get_width()/2,v+0.3,f"{v:.1f}%",ha="center")
plt.ylabel("Rejection rate (% of closed)"); plt.title("RQ6 (matched repos): rejection by test inclusion")
plt.grid(axis="y",alpha=.3); plt.tight_layout(); plt.savefig(os.path.join(FIG,"rq6_matched.png"),dpi=130); plt.close()

ia=smc[smc.is_agent].groupby("month")["has_test"].mean().reindex(months)*100
ih=smc[~smc.is_agent].groupby("month")["has_test"].mean().reindex(months)*100
plt.figure(figsize=(10,5))
plt.plot(months,ia.values,"o-",label="Agent",color="#d62728")
plt.plot(months,ih.values,"s-",label="Human",color="#1f77b4")
plt.ylabel("% of fixes that include a test"); plt.xlabel("Month"); plt.xticks(rotation=45,ha="right")
plt.title("RQ6b (matched repos): test-inclusion rate over time"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"rq6b_matched.png"),dpi=130); plt.close()

aa=smc[(smc.grp=="Agent") & smc.closed].copy()
bins=[-1,0,10,50,200,1e12]; lbls=["0","1-10","11-50","51-200","200+"]
aa["cb"]=pd.cut(aa["churn"],bins=bins,labels=lbls)
rtv=[]; rnv=[]
for bb in lbls:
    sub=aa[aa.cb==bb]
    rtv.append(100*sub[sub.has_test]["rejected"].mean())
    rnv.append(100*sub[~sub.has_test]["rejected"].mean())
x=np.arange(len(lbls))
plt.figure(figsize=(9,5))
plt.bar(x-0.2,rtv,width=0.4,label="with test",color="#d62728")
plt.bar(x+0.2,rnv,width=0.4,label="no test",color="#f4a3a3")
plt.xticks(x,lbls); plt.xlabel("Code churn bin (lines)"); plt.ylabel("Rejection rate (% of closed)")
plt.title("RQ6 (matched, robustness): agent rejection by test inclusion, within churn bins")
plt.legend(); plt.grid(axis="y",alpha=.3); plt.tight_layout(); plt.savefig(os.path.join(FIG,"rq6c_matched.png"),dpi=130); plt.close()

print("[done] saved rq1a/rq1b/rq4/rq5/rq6/rq6b/rq6c _matched.png")
