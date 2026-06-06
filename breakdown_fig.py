"""Make the repo-type 2x2 breakdown figure (justifies the matched comparison)."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DATA = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes\data"
FIG = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes\figures"

p = pd.read_parquet(os.path.join(DATA, "fix_prs_only.parquet"),
                    columns=["repo_id","state","merged_at","is_agent"])
p["closed"] = p["state"] == "closed"
mna = p["merged_at"].isna() | p["merged_at"].astype(str).isin(["","NaT","None"])
p["rejected"] = p["closed"] & mna
A = set(p.loc[p.is_agent,"repo_id"]); H = set(p.loc[~p.is_agent,"repo_id"])
shared = A & H
p["matched"] = p["repo_id"].isin(shared)

def rr(mask):
    d = p[mask & p.closed]; return 100*d["rejected"].mean(), len(d)

a_sh, na_sh = rr(p.is_agent & p.matched)
a_so, na_so = rr(p.is_agent & ~p.matched)
h_sh, nh_sh = rr(~p.is_agent & p.matched)
h_so, nh_so = rr(~p.is_agent & ~p.matched)
print(f"Agent  shared {a_sh:.1f}% (n={na_sh:,}) | single-type {a_so:.1f}% (n={na_so:,})")
print(f"Human  shared {h_sh:.1f}% (n={nh_sh:,}) | single-type {h_so:.1f}% (n={nh_so:,})")

x = np.arange(2)
plt.figure(figsize=(8,5))
b1 = plt.bar(x-0.2, [a_sh, a_so], width=0.4, label="Agent", color="#d62728")
b2 = plt.bar(x+0.2, [h_sh, h_so], width=0.4, label="Human", color="#1f77b4")
for bars, ns in [(b1,[na_sh,na_so]), (b2,[nh_sh,nh_so])]:
    for bar, n in zip(bars, ns):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                 f"{bar.get_height():.1f}%\n(n={n:,})", ha="center", va="bottom", fontsize=8)
plt.xticks(x, ["Shared repos\n(have both agent & human)", "Single-type repos\n(only one kind)"])
plt.ylabel("Rejection rate (% of closed)")
plt.title("Where the fix lives changes the rejection rate\n(so we compare agent vs human only in shared repos)")
plt.ylim(0, max(a_sh,a_so,h_sh,h_so)+5); plt.legend(); plt.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_repotype_breakdown.png"), dpi=130); plt.close()
print("saved fig_repotype_breakdown.png")
