"""
Quick-trio analysis on the GitHub-Agentic-PR bug-fix dataset.
RQ1: rejection rate + code churn over time (agent vs human)
RQ2a: agent share over time (which agent people use to fix bugs)
RQ6: does shipping a test correlate with acceptance
Outputs: figures/*.png  and  RESULTS.md
"""
import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, chi2_contingency

warnings.filterwarnings("ignore")
BASE = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes"
DATA = os.path.join(BASE, "data")
FIG = os.path.join(BASE, "figures"); os.makedirs(FIG, exist_ok=True)
AGENTS = ["Claude_Code", "Cursor", "Copilot", "Devin"]
out = []  # lines for RESULTS.md
def w(s=""): out.append(s); print(s)

# ---------- load PRs ----------
prs = pd.read_parquet(os.path.join(DATA, "fix_prs_only.parquet"),
                      columns=["id","state","created_at","merged_at","is_agent","agent"])
prs["month"] = prs["created_at"].str.slice(0, 7)
prs = prs[prs["month"].notna() & (prs["month"] != "")]
prs["closed"] = prs["state"] == "closed"
mna = prs["merged_at"].isna() | (prs["merged_at"].astype(str).isin(["", "NaT", "None"]))
prs["merged"] = ~mna
prs["rejected"] = prs["closed"] & ~prs["merged"]
prs["grp"] = np.where(prs["is_agent"], "Agent", "Human")
months = sorted(prs["month"].unique())
midx = {m: i for i, m in enumerate(months)}

w("# Bug-Fix Agentic-PRs — Temporal Findings (Quick Trio)\n")
w(f"**Dataset:** `mabujadallah/GitHub-Agentic-PR-Dataset` -> `fix_prs_only.parquet` "
  f"({len(prs):,} bug-fix PRs, type=fix). **Span:** {months[0]} to {months[-1]} ({len(months)} months).")
w(f"**Groups:** Agent = is_agent==True ({int(prs['is_agent'].sum()):,} PRs: "
  f"{', '.join(AGENTS)}); Human = is_agent==False ({int((~prs['is_agent']).sum()):,} PRs).\n")
w("## How everything is calculated (shared definitions)")
w("- **Month** = first 7 chars of `created_at` (`YYYY-MM`), i.e. the PR's creation month.")
w("- **Rejected** = `state=='closed'` AND `merged_at` is null/empty (closed without merge).")
w("- **Merged** = `merged_at` is present. **Decided/closed** = `state=='closed'` (merged + rejected).")
w("- **Rejection rate** = rejected / closed (open PRs excluded, so undecided recent PRs don't deflate it).")
w("- **Trend test** = Spearman correlation between month index (0..N) and the monthly value; "
  "reports rho and p (monthly points are the observations).")
w("- *Caveat:* the most recent month(s) are right-censored (slow merges still pending), so read the last point loosely.\n")

# ================= RQ1a: rejection rate over time =================
def monthly_rate(df):
    g = df[df["closed"]].groupby("month")
    rej = g["rejected"].sum(); clo = g["rejected"].count()
    return (100 * rej / clo).reindex(months)

agent_rej = monthly_rate(prs[prs["grp"] == "Agent"])
human_rej = monthly_rate(prs[prs["grp"] == "Human"])

plt.figure(figsize=(10, 5))
plt.plot(months, agent_rej.values, "o-", label="Agent", color="#d62728")
plt.plot(months, human_rej.values, "s-", label="Human", color="#1f77b4")
plt.ylabel("Rejection rate (% of closed)"); plt.xlabel("Month (PR created)")
plt.title("RQ1a: Bug-fix PR rejection rate over time"); plt.xticks(rotation=45, ha="right")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(FIG, "rq1a_rejection_over_time.png"), dpi=130); plt.close()

xi = np.arange(len(months))
ra, pa = spearmanr(xi, agent_rej.values)
rh, ph = spearmanr(xi, human_rej.values)
w("## RQ1a — Does the rejection rate change over time?")
w(f"**Answer:** Agent rejection rate goes from **{agent_rej.iloc[0]:.1f}%** ({months[0]}) to "
  f"**{agent_rej.iloc[-1]:.1f}%** ({months[-1]}); mean **{agent_rej.mean():.1f}%**, range "
  f"{agent_rej.min():.1f}-{agent_rej.max():.1f}%. Trend Spearman rho={ra:.2f} (p={pa:.3f}). "
  f"Human baseline mean **{human_rej.mean():.1f}%** (rho={rh:.2f}, p={ph:.3f}).")
w(f"- Interpretation: {'a significant' if pa<0.05 else 'no significant'} monotonic time trend for agents (alpha=.05).")
w("- Figure: `figures/rq1a_rejection_over_time.png`")
w("- How: per month, rejected/closed within each group (defs above); trend = Spearman(month-index, monthly rate).\n")

# ================= RQ2a: agent share over time =================
agt = prs[prs["is_agent"]].copy()
piv = agt[agt["agent"].isin(AGENTS)].pivot_table(index="month", columns="agent",
        values="id", aggfunc="count", fill_value=0).reindex(months).fillna(0)
share = piv.div(piv.sum(axis=1), axis=0) * 100

plt.figure(figsize=(10, 5))
plt.stackplot(months, [share[a].values for a in AGENTS], labels=AGENTS,
              colors=["#9467bd", "#2ca02c", "#ff7f0e", "#8c564b"])
plt.ylabel("Share of agent bug-fix PRs (%)"); plt.xlabel("Month (PR created)")
plt.title("RQ2a: Which agent people use to fix bugs, over time")
plt.xticks(rotation=45, ha="right"); plt.legend(loc="upper left", ncol=4)
plt.margins(x=0); plt.tight_layout()
plt.savefig(os.path.join(FIG, "rq2a_agent_share_over_time.png"), dpi=130); plt.close()

first, last = share.iloc[0], share.iloc[-1]
w("## RQ2a — Do people switch which agent they use to fix bugs?")
w("**Answer:** Yes, the mix shifts a lot. Share of agent bug-fix PRs (start -> end month):")
for a in AGENTS:
    w(f"  - {a}: {first[a]:.0f}% -> {last[a]:.0f}%  (peak {share[a].max():.0f}% in {share[a].idxmax()})")
w(f"- Copilot's volume is tiny until ~2025-05 then explodes (platform launch effect); "
  f"Cursor & Copilot spike together in 2025-07; Devin declines after spring 2025.")
w("- Figure: `figures/rq2a_agent_share_over_time.png`")
w("- How: among agent PRs, monthly count per agent / total agent PRs that month = share%.\n")

# ================= load commit-file details (churn + tests) =================
det = pd.read_parquet(os.path.join(DATA, "fix_pr_commit_details.parquet"),
                      columns=["pr_id", "filename", "additions", "deletions"])
det["additions"] = det["additions"].fillna(0); det["deletions"] = det["deletions"].fillna(0)
det["lines"] = det["additions"] + det["deletions"]
# churn per PR = total lines changed across all commit-file rows of the PR
churn = det.groupby("pr_id")["lines"].sum().rename("churn")
# test flag per PR
TEST_RE = re.compile(r"(^|/)(tests?|__tests__|spec)(/|\.)|(_test\.|test_|\.test\.|\.spec\.|_spec\.|\.tests\.)", re.I)
det["is_test"] = det["filename"].fillna("").map(lambda f: bool(TEST_RE.search(f)))
has_test = det.groupby("pr_id")["is_test"].any().rename("has_test")

prs = prs.merge(churn, left_on="id", right_index=True, how="left")
prs = prs.merge(has_test, left_on="id", right_index=True, how="left")
cov = prs["churn"].notna().mean() * 100
prs["churn"] = prs["churn"].fillna(0)
prs["has_test"] = prs["has_test"].fillna(False).astype(bool)

# ================= RQ1b: churn over time =================
def monthly_median_churn(df):
    return df.groupby("month")["churn"].median().reindex(months)
agent_ch = monthly_median_churn(prs[prs["grp"] == "Agent"])
human_ch = monthly_median_churn(prs[prs["grp"] == "Human"])

plt.figure(figsize=(10, 5))
plt.plot(months, agent_ch.values, "o-", label="Agent", color="#d62728")
plt.plot(months, human_ch.values, "s-", label="Human", color="#1f77b4")
plt.ylabel("Median code churn (lines +/-)"); plt.xlabel("Month (PR created)")
plt.title("RQ1b: Median bug-fix code churn over time"); plt.xticks(rotation=45, ha="right")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(FIG, "rq1b_churn_over_time.png"), dpi=130); plt.close()

rac, pac = spearmanr(xi, agent_ch.values)
w("## RQ1b — Does code churn (fix size) change over time?")
w(f"**Answer:** Agent median churn: {agent_ch.iloc[0]:.0f} ({months[0]}) -> {agent_ch.iloc[-1]:.0f} "
  f"({months[-1]}); overall agent median **{prs[prs.grp=='Agent']['churn'].median():.0f}** lines, "
  f"human median **{prs[prs.grp=='Human']['churn'].median():.0f}**. "
  f"Trend Spearman rho={rac:.2f} (p={pac:.3f}).")
w(f"- Coverage: {cov:.1f}% of fix PRs have commit-file rows; the rest treated as 0 churn (no recorded file changes).")
w("- Figure: `figures/rq1b_churn_over_time.png`")
w("- How: churn(PR) = sum(additions+deletions) over all its commit-file rows; monthly **median** per group.\n")

# ================= RQ6: does a test help? =================
def rej_rate(df):
    d = df[df["closed"]]
    return (100 * d["rejected"].mean(), len(d))
rows = []
for g in ["Agent", "Human"]:
    sub = prs[prs["grp"] == g]
    rt, nt = rej_rate(sub[sub["has_test"]])
    rn, nn = rej_rate(sub[~sub["has_test"]])
    rows.append((g, rt, nt, rn, nn))
    # chi-square on closed PRs: has_test x rejected
    c = sub[sub["closed"]]
    ct = pd.crosstab(c["has_test"], c["rejected"])
    chi2, p, _, _ = chi2_contingency(ct)
    rows[-1] = rows[-1] + (p,)

labels = ["Agent\n+test", "Agent\nno-test", "Human\n+test", "Human\nno-test"]
vals = [rows[0][1], rows[0][3], rows[1][1], rows[1][3]]
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, vals, color=["#d62728", "#f4a3a3", "#1f77b4", "#a3c4f4"])
for b, v in zip(bars, vals): plt.text(b.get_x()+b.get_width()/2, v+0.3, f"{v:.1f}%", ha="center")
plt.ylabel("Rejection rate (% of closed)")
plt.title("RQ6: Bug-fix rejection rate by whether a test was included")
plt.grid(axis="y", alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(FIG, "rq6_test_effect.png"), dpi=130); plt.close()

# test inclusion over time
ti_a = prs[prs.grp=="Agent"].groupby("month")["has_test"].mean().reindex(months)*100
ti_h = prs[prs.grp=="Human"].groupby("month")["has_test"].mean().reindex(months)*100
plt.figure(figsize=(10, 5))
plt.plot(months, ti_a.values, "o-", label="Agent", color="#d62728")
plt.plot(months, ti_h.values, "s-", label="Human", color="#1f77b4")
plt.ylabel("% of bug-fix PRs that include a test"); plt.xlabel("Month (PR created)")
plt.title("RQ6b: Test-inclusion rate over time"); plt.xticks(rotation=45, ha="right")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(FIG, "rq6b_test_inclusion_over_time.png"), dpi=130); plt.close()

w("## RQ6 — Does shipping a test correlate with acceptance?")
for g, rt, nt, rn, nn, p in rows:
    delta = rt - rn
    w(f"**{g}:** with test = **{rt:.1f}%** rejected (n={nt:,}); without test = **{rn:.1f}%** (n={nn:,}); "
      f"diff {delta:+.1f} pts; chi-square p={p:.2e} {'(significant)' if p<0.05 else '(n.s.)'}.")
w(f"- Test-inclusion (agent): {ti_a.iloc[0]:.0f}% -> {ti_a.iloc[-1]:.0f}% over the period (mean {ti_a.mean():.0f}%).")
w("- Figures: `figures/rq6_test_effect.png`, `figures/rq6b_test_inclusion_over_time.png`")
w("- How: a PR 'has_test' if ANY changed file path matches the test regex "
  r"`(^|/)(tests?|__tests__|spec)(/|\.)|(_test\.|test_|\.test\.|\.spec\.|_spec\.|\.tests\.)`; "
  "rejection rate compared with/without test; chi-square 2x2 (has_test x rejected) on closed PRs.\n")

with open(os.path.join(BASE, "RESULTS.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("\n[done] wrote RESULTS.md and figures/*.png")
