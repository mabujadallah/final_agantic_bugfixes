"""
NC3 — Agent autonomy index: behind a MERGED agent fix, how often did a human also commit?
Human-rework = a commit whose AUTHOR is not the agent/bot signature. Reliable for Copilot, Devin,
Cursor (clear bot-authored commits); Claude_Code commits under the developer's git identity, so its
author-based rework can't be separated (reported with caveat). Dataset-only (local pr_commits).
"""
import os, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes"; D = os.path.join(BASE, "data"); FIG = os.path.join(BASE, "figures")
RELIABLE = ["Copilot", "Devin"]   # CLOUD agents commit as bots → autonomy measurable.
# Cursor & Claude_Code are LOCAL/IDE agents that commit under the developer's git identity,
# so agent-vs-human authorship can't be separated for them (a finding in itself).

clean = pd.read_parquet(os.path.join(BASE, "analysis_clean.parquet"))
ag = clean[(clean.group == "agent") & clean.merged & clean.mature].copy()

c = pd.read_parquet(os.path.join(D, "pr_commits.parquet"), columns=["pr_id", "author"])
c = c[c.pr_id.isin(set(ag.id))].copy()

AGENTSIG = re.compile(
    r"copilot|cursoragent|cursor\s*agent|devin|claude|google-labs-jules|jules|web-flow|"
    r"\[bot\]|dependabot|renovate|github-actions|actions-user|sweep|gitstart|sourcery", re.I)
c["human"] = ~c["author"].fillna("").map(lambda a: bool(AGENTSIG.search(a)) or a == "")
hum = c.groupby("pr_id")["human"].sum().rename("n_human_commits")
ag = ag.merge(hum, left_on="id", right_index=True, how="left")
ag["n_human_commits"] = ag["n_human_commits"].fillna(0)
ag["has_human"] = ag["n_human_commits"] > 0
ag["autonomous"] = ~ag["has_human"]

def auto(df): return 100 * df["autonomous"].mean()
print("Autonomy index = % of merged agent fixes with ZERO human commits")
print("CLOUD agents (Copilot, Devin) commit as bots -> measurable.")
print("LOCAL agents (Cursor, Claude_Code) commit under the developer's identity -> NOT measurable.\n")
for a in ["Copilot", "Devin", "Cursor", "Claude_Code"]:
    d = ag[ag.agent == a]
    tag = "  [cloud: reliable]" if a in RELIABLE else "  [local: commits as dev -> artifact, ignore]"
    print(f"  {a:12s}: autonomy {auto(d):5.1f}%  (n={len(d):,}, median commits={d.n_commits.median():.0f}){tag}")
rel = ag[ag.agent.isin(RELIABLE)]
print(f"\nCLOUD agents combined: autonomy {auto(rel):.1f}%  (i.e. {100-auto(rel):.1f}% of merged fixes needed a human commit)")

# autonomy by bug type (reliable agents)
bt = rel.groupby("btype").agg(n=("id","size"), autonomy=("autonomous","mean"))
bt["autonomy"] *= 100; bt = bt[bt.n >= 80].sort_values("autonomy")
print("\nautonomy by bug type (cloud agents Copilot+Devin, n>=80):")
print(bt.round(1).to_string())

# autonomy over time (reliable agents)
rel = rel.copy(); rel["q"] = pd.PeriodIndex(pd.to_datetime(rel.created_dt, utc=True), freq="Q").astype(str)
ot = rel.groupby("q")["autonomous"].mean()*100

# ---- figure: autonomy by agent + by bug type ----
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
order = ["Copilot","Devin","Cursor","Claude_Code"]
vals = [auto(ag[ag.agent==a]) for a in order]
cols = ["#ff7f0e","#8c564b","#2ca02c","#9467bd"]
b = ax[0].bar(order, vals, color=cols)
for bar,v,a in zip(b,vals,order):
    ax[0].text(bar.get_x()+bar.get_width()/2, v, f"{v:.0f}%", ha="center", va="bottom")
ax[0].set_ylabel("Autonomy index (% merged fixes, no human commit)")
ax[0].set_title("NC3: agent autonomy by agent\n(Cursor/Claude commit as the developer -> artifact, not real)")
ax[0].axhline(0); ax[0].grid(axis="y", alpha=.3); ax[0].tick_params(axis="x", rotation=15)
bt["autonomy"].plot.barh(ax=ax[1], color="#ff7f0e")
ax[1].set_xlabel("Autonomy index (%)"); ax[1].set_title("NC3: autonomy by bug type (cloud agents: Copilot + Devin)")
ax[1].grid(axis="x", alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "nc3_autonomy.png"), dpi=130); plt.close()
print("\nautonomy over time (reliable):"); print(ot.round(1).to_string())
print("saved figures/nc3_autonomy.png")
