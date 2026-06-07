"""
Phase 0 — Data cleaning & preprocessing for the Q1 study.
Produces a single cleaned analysis table (analysis_clean.parquet) consumed by NC1-NC4,
plus two review CSVs (human-login bot review, fix-label validation sample).

Cleaning: de-bot the human baseline, clean churn (drop generated/vendored, winsorize),
add decision-window maturity flag, bug-type, links-issue, has_test, #commits, matched flag.
Dataset-only; no GitHub API.
"""
import os, re
import numpy as np
import pandas as pd

D = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes\data"
OUT = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes"
AGENTS = ["Claude_Code", "Cursor", "Copilot", "Devin"]
END = pd.Timestamp("2026-02-28", tz="UTC")          # dataset end
MATURE_DAYS = 60                                      # min age to count an outcome as "decided"

# ---------- load PRs ----------
prs = pd.read_parquet(os.path.join(D, "fix_prs_only.parquet"),
        columns=["id","number","title","body","user","state","created_at","merged_at",
                 "repo_id","repo_name","html_url","is_agent","agent","source"])
prs["created_dt"] = pd.to_datetime(prs["created_at"], utc=True, errors="coerce")
prs = prs[prs["created_dt"].notna()].copy()
prs["month"] = prs["created_dt"].dt.strftime("%Y-%m")
prs["closed"]   = prs["state"] == "closed"
_mna            = prs["merged_at"].isna() | prs["merged_at"].astype(str).isin(["","NaT","None"])
prs["merged"]   = ~_mna
prs["rejected"] = prs["closed"] & _mna
prs["mature"]   = (END - prs["created_dt"]).dt.days >= MATURE_DAYS

# ---------- 1) de-bot the human baseline ----------
BOT = re.compile(
    r"\[bot\]$|(^|[-_])bot([-_]|$)|bot\d*$|machine|automation|autom|backport|renovate|"
    r"dependabot|github-actions|greenkeeper|snyk|mergify|imgbot|codecov|allcontributors|"
    r"sweep|stale|ci-?bot|release-please|semantic-release|pre-commit|netlify|vercel|"
    r"cla-?assistant|deepsource|sonarcloud|gitguardian|copybara|roomote|robobun|"
    r"kibanamachine|cbl-mariner|azurelinux|octo-sts|runway|vaadin-bot|cherrypick-robot|"
    r"ydbot|blathers", re.I)
u = prs["user"].fillna("").astype(str)
prs["is_bot"] = (~prs["is_agent"]) & u.map(lambda x: bool(BOT.search(x)))
prs["group"]  = np.where(prs["is_agent"], "agent", np.where(prs["is_bot"], "bot", "human"))

# ---------- 2) commit-file details: cleaned churn + has_test + n_files ----------
det = pd.read_parquet(os.path.join(D, "fix_pr_commit_details.parquet"),
                      columns=["pr_id","sha","filename","additions","deletions"])
det["fn"] = det["filename"].fillna("")
NOISE = re.compile(
    r"package-lock\.json|yarn\.lock|pnpm-lock|npm-shrinkwrap|go\.sum|Cargo\.lock|poetry\.lock|"
    r"Gemfile\.lock|composer\.lock|\.min\.(js|css)|/vendor/|/dist/|/build/|generated|\.pb\.go$|"
    r"_pb2\.py$|\.snap$|\.lock$", re.I)
det["noisy"]   = det["fn"].map(lambda f: bool(NOISE.search(f)))
det["lines"]   = det["additions"].fillna(0) + det["deletions"].fillna(0)
TEST_RE = re.compile(r"(^|/)(tests?|__tests__|spec)(/|\.)|(_test\.|test_|\.test\.|\.spec\.|_spec\.|\.tests\.)", re.I)
det["is_test"] = det["fn"].map(lambda f: bool(TEST_RE.search(f)))

clean = det[~det["noisy"]]
churn_raw   = det.groupby("pr_id")["lines"].sum().rename("churn_raw")
churn_clean = clean.groupby("pr_id")["lines"].sum().rename("churn")
nfiles      = det.groupby("pr_id")["fn"].nunique().rename("n_files")
nfiles_cl   = clean.groupby("pr_id")["fn"].nunique().rename("n_files_clean")
has_test    = det.groupby("pr_id")["is_test"].any().rename("has_test")
ncommits    = det.groupby("pr_id")["sha"].nunique().rename("n_commits")

for s in [churn_raw, churn_clean, nfiles, nfiles_cl, has_test, ncommits]:
    prs = prs.merge(s, left_on="id", right_index=True, how="left")
prs["has_test"]   = prs["has_test"].fillna(False).astype(bool)
prs["has_files"]  = prs["churn_raw"].notna()
for c in ["churn_raw","churn","n_files","n_files_clean","n_commits"]:
    prs[c] = prs[c].fillna(0)
# winsorize cleaned churn at p99 (per group, to be fair)
p99 = prs["churn"].quantile(0.99)
prs["churn_w"]   = prs["churn"].clip(upper=p99)
prs["log_churn"] = np.log1p(prs["churn"])

# ---------- 3) bug type + links-issue + text lengths ----------
CATS = [
 ("security",     r"\b(security|vulnerab\w*|cve-?\d|xss|csrf|sql\s*injection|injection|exploit|sanitiz)\b"),
 ("crash",        r"\b(crash\w*|segfault|seg\s*fault|npe|null\s*pointer|nullpointer|exception|panic|fatal|stack\s*overflow)\b"),
 ("concurrency",  r"\b(race\s*condition|deadlock|concurren\w*|thread[-\s]?safe\w*|mutex|data\s*race|atomic)\b"),
 ("memory",       r"\b(memory\s*leak|mem\s*leak|out\s*of\s*memory|oom|buffer\s*overflow|leak)\b"),
 ("performance",  r"\b(performance|perf|slow\w*|latency|speed\s*up|optimi[sz]\w*|throughput|timeout)\b"),
 ("security_auth",r"\b(authenticat\w*|authoriz\w*|permission|token|login|session)\b"),
 ("ui",           r"\b(ui|ux|css|layout|render\w*|display|styling|stylesheet|button|alignment|responsive|dark\s*mode)\b"),
 ("build_ci",     r"\b(build|ci|compil\w*|lint\w*|dependenc\w*|import\s*error|module\s*not\s*found|version\s*bump|packaging)\b"),
 ("typo_doc",     r"\b(typo|spelling|grammar|docstring|readme|documentation|docs)\b"),
 ("typing",       r"\b(type\s*error|typing|type\s*hint|mypy|type\s*annotation|typescript\s*type)\b"),
 ("data_parse",   r"\b(parse|parsing|serializ\w*|deserializ\w*|json|yaml|encoding|decod\w*|formatting)\b"),
 ("network",      r"\b(http|https|api|request|response|endpoint|url|websocket|socket|connection)\b"),
]
COMP = [(n, re.compile(p, re.I)) for n, p in CATS]
def classify(t, b):
    t = t if isinstance(t, str) else ""; b = b if isinstance(b, str) else ""
    txt = f"{t} {b[:300]}"
    for n, rx in COMP:
        if rx.search(txt): return n
    return "other_logic"
KEY = re.compile(r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s+#(\d+)", re.I)
prs["btype"]       = [classify(t, b) for t, b in zip(prs["title"], prs["body"])]
prs["links_issue"] = [bool(KEY.search(f"{t or ''}\n{b or ''}")) for t, b in zip(prs["title"], prs["body"])]
prs["title_len"]   = prs["title"].fillna("").str.len()
prs["body_len"]    = prs["body"].fillna("").str.len()

# ---------- 4) matched repos (after de-bot): repos with BOTH agent and (clean) human fixes ----------
A = set(prs.loc[prs.group=="agent", "repo_id"])
H = set(prs.loc[prs.group=="human", "repo_id"])
MATCHED = A & H
prs["matched"] = prs["repo_id"].isin(MATCHED)

# ---------- emit cleaned table ----------
keep = ["id","number","repo_id","repo_name","html_url","created_at","created_dt","month","mature",
        "is_agent","agent","source","group","is_bot","state","closed","merged","rejected",
        "churn_raw","churn","churn_w","log_churn","n_files","n_files_clean","has_files","has_test",
        "n_commits","btype","links_issue","title_len","body_len","matched"]
clean_tbl = prs[keep].copy()
clean_tbl.to_parquet(os.path.join(OUT, "analysis_clean.parquet"), index=False)

# ---------- review CSVs ----------
hu = prs[~prs.is_agent].copy(); hu["u"] = hu["user"].fillna("")
rev = (hu.groupby("u").agg(prs=("u","size"), repos=("repo_id","nunique"),
        is_bot=("is_bot","first")).sort_values("prs",ascending=False).head(150))
rev.to_csv(os.path.join(OUT, "review_human_logins.csv"))
samp = (prs.groupby(np.where(prs.is_agent, prs.agent, "human"), group_keys=False)
          .apply(lambda d: d.sample(min(40, len(d)), random_state=1)))
samp[["id","html_url","agent","group","btype","title"]].to_csv(
    os.path.join(OUT, "review_fix_label_sample.csv"), index=False)

# ---------- report ----------
def rate(m): d = prs[m & prs.closed]; return 100*d["rejected"].mean() if len(d) else float("nan")
print("="*70)
print(f"rows: {len(prs):,}")
print(f"groups: agent={int((prs.group=='agent').sum()):,} | "
      f"human(clean)={int((prs.group=='human').sum()):,} | bot(removed)={int((prs.group=='bot').sum()):,}")
print(f"bots removed from human: {int((prs.group=='bot').sum()):,} "
      f"({100*(prs.group=='bot').sum()/(~prs.is_agent).sum():.1f}% of original human)")
print(f"matched repos after de-bot: {len(MATCHED):,} (was 1,218 before cleaning)")
print(f"mature PRs (>= {MATURE_DAYS}d): {int(prs.mature.sum()):,} ({100*prs.mature.mean():.1f}%)")
print(f"churn p99 winsor cap: {p99:.0f} | median churn raw={prs.churn_raw.median():.0f} clean={prs.churn.median():.0f}")
print(f"agent rej (matched,mature): {rate(prs.matched & prs.mature & prs.is_agent):.1f}% | "
      f"human rej (matched,mature): {rate(prs.matched & prs.mature & (prs.group=='human')):.1f}%")
print("wrote analysis_clean.parquet, review_human_logins.csv, review_fix_label_sample.csv")
print("="*70)
