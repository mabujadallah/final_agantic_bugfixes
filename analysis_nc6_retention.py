"""
NC6 — First impressions: does a rejected first agent fix kill adoption?
For each repo, take its FIRST agent fix PR. Classify the first impression (merged vs rejected),
then measure retention = did the repo submit another agent fix within 90 days of the DECISION?
Fixed follow-up window (decision >= 90 days before dataset end) so censoring cannot bias it.
Dataset-only (analysis_clean + timestamps from fix_prs_only).
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

BASE = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes"
D = os.path.join(BASE, "data"); FIG = os.path.join(BASE, "figures")
END = pd.Timestamp("2026-02-28", tz="UTC"); WIN = 90

clean = pd.read_parquet(os.path.join(BASE, "analysis_clean.parquet"))
ts = pd.read_parquet(os.path.join(D, "fix_prs_only.parquet"), columns=["id", "merged_at", "closed_at"])
for c in ["merged_at", "closed_at"]:
    ts[c] = pd.to_datetime(ts[c], utc=True, errors="coerce")

ag = clean[clean.group == "agent"].merge(ts, on="id", how="left").sort_values("created_dt")
ag["decision_dt"] = pd.to_datetime(np.where(ag.merged, ag.merged_at, ag.closed_at), utc=True)

# first agent fix per repo; keep only decided firsts with a full follow-up window
first = ag.groupby("repo_id").first()                      # sorted by created_dt
first = first[first.closed & first.decision_dt.notna()]
first = first[first.decision_dt <= END - pd.Timedelta(days=WIN)]
first["impression"] = np.where(first.merged, "merged", "rejected")

# next agent fix in the same repo strictly after the first decision
nxt = ag.merge(first[["decision_dt"]].rename(columns={"decision_dt": "dec0"}),
               left_on="repo_id", right_index=True, how="inner")
nxt = nxt[nxt.created_dt > nxt.dec0]
gap = (nxt.created_dt - nxt.dec0).dt.days
first["days_to_next"] = nxt.assign(gap=gap).groupby("repo_id")["gap"].min()
first["retained"] = first["days_to_next"] <= WIN

# interim fixes (created before the first decision) — flag "clean" first impressions
interim = ag.merge(first[["created_dt", "decision_dt"]].rename(
        columns={"created_dt": "c0", "decision_dt": "dec0"}), left_on="repo_id", right_index=True)
n_interim = interim[(interim.created_dt > interim.c0) & (interim.created_dt <= interim.dec0)] \
                .groupby("repo_id").size()
first["clean_first"] = ~first.index.isin(n_interim.index)

print(f"repos with a decided first agent fix and {WIN}d follow-up: {len(first):,} "
      f"(merged {int(first.merged.sum()):,} | rejected {int((~first.merged).sum()):,})")

def report(df, label):
    t = df.groupby("impression").agg(n=("retained", "size"), retained=("retained", "mean"),
                                     med_days=("days_to_next", "median"))
    t["retained"] *= 100
    ct = pd.crosstab(df.impression, df.retained)
    chi2, p = chi2_contingency(ct)[:2]
    print(f"\n{label} — retention within {WIN}d of first decision (chi2 p={p:.2e}):")
    print(t.round(1).to_string())
    return t

t_all = report(first, "ALL first impressions")
t_clean = report(first[first.clean_first], "CLEAN first impressions (no other agent fix pending)")

# per first-agent breakdown (all firsts)
per = (first.groupby(["agent", "impression"])["retained"].agg(["size", "mean"]).unstack())
print("\n90-day retention by agent of the first fix (%):")
print((per["mean"] * 100).round(1).to_string())

# retention curve: % of repos with a next agent fix within X days, by impression
xs = [7, 14, 30, 60, 90, 120, 180]
curves = {}
for imp in ["merged", "rejected"]:
    s = first[(first.impression == imp) & (first.decision_dt <= END - pd.Timedelta(days=max(xs)))]
    curves[imp] = [100 * (s.days_to_next <= x).mean() for x in xs]

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(xs, curves["merged"], "o-", color="#2ca02c", label="first fix merged")
ax[0].plot(xs, curves["rejected"], "o-", color="#d62728", label="first fix rejected")
ax[0].set_xlabel("Days since first decision"); ax[0].set_ylabel("% repos with another agent fix")
ax[0].set_title("NC6: repo keeps using agents after its first agent fix\n(180d follow-up subset)")
ax[0].legend(); ax[0].grid(alpha=.3)
pm = (per["mean"] * 100)
pm.plot.bar(ax=ax[1], color=["#2ca02c", "#d62728"])
ax[1].set_ylabel(f"{WIN}-day retention (%)"); ax[1].set_xlabel("Agent of the first fix")
ax[1].set_title("NC6: first-impression effect by agent")
ax[1].legend(title="first impression"); ax[1].grid(axis="y", alpha=.3)
ax[1].tick_params(axis="x", rotation=15)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "nc6_retention.png"), dpi=130); plt.close()
print("\nsaved figures/nc6_retention.png")
