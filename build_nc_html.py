"""Build NC_FINDINGS.html — plain-language write-up of NC1-NC9 with simplified one-message
figures (figures/simple/nc*.png), embedded as base64 so the file is self-contained.
Summary numbers for NC1-NC4 and NC7 are taken from the printed output of analysis_nc*.py;
trend figures (NC5, NC6, NC8, NC9) are recomputed from the data so they stay exact."""
import base64, os, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes"
D = os.path.join(BASE, "data"); SIMP = os.path.join(BASE, "figures", "simple")
os.makedirs(SIMP, exist_ok=True)
END = pd.Timestamp("2026-02-28", tz="UTC")
AGENT_C, HUMAN_C = "#d62728", "#1f77b4"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "figure.facecolor": "white"})

clean = pd.read_parquet(os.path.join(BASE, "analysis_clean.parquet"))
ts = pd.read_parquet(os.path.join(D, "fix_prs_only.parquet"), columns=["id", "merged_at", "closed_at"])
for c in ["merged_at", "closed_at"]:
    ts[c] = pd.to_datetime(ts[c], utc=True, errors="coerce")
cl = clean.merge(ts, on="id", how="left")

def save(name):
    plt.tight_layout(); plt.savefig(os.path.join(SIMP, name), dpi=130); plt.close()

# ---------- NC1: instruction-file adoption (numbers from analysis_nc1_did.py) ----------
plt.figure(figsize=(7.5, 4.6))
plt.bar(["Before adopting\nan instruction file", "After adopting"], [11.5, 13.4],
        color=["#8da8c6", "#4a6fa5"], width=.55)
for x, v in enumerate([11.5, 13.4]):
    plt.text(x, v + .2, f"{v:.1f}%", ha="center", fontsize=14, fontweight="bold")
plt.ylabel("Agent fixes rejected (%)"); plt.ylim(0, 16)
plt.title("NC1: rejection in the 351 projects that adopted\nan instruction file")
plt.grid(axis="y", alpha=.3); save("nc1.png")

# ---------- NC2: predictability (numbers from analysis_nc2_predict.py) ----------
imp = pd.Series({"Project's past rejection rate": .084, "Which agent wrote it": .045,
                 "Length of the PR description": .038, "Size of the fix": .016,
                 "Number of files touched": .013, "Length of the title": .007})
plt.figure(figsize=(8.5, 4.6))
imp[::-1].plot.barh(color="#4a6fa5")
plt.xlabel("How much the prediction worsens when this signal is removed")
plt.title("NC2: what predicts that an agent fix will be rejected\n(model score 0.76; 0.5 = coin toss, 1.0 = perfect)")
plt.grid(axis="x", alpha=.3); save("nc2.png")

# ---------- NC3: autonomy (numbers from analysis_nc3_autonomy.py) ----------
plt.figure(figsize=(7.5, 4.6))
plt.bar(["Copilot", "Devin"], [78.2, 90.2], color=["#ff7f0e", "#8c564b"], width=.5)
for x, v in enumerate([78.2, 90.2]):
    plt.text(x, v + 1, f"{v:.0f}%", ha="center", fontsize=14, fontweight="bold")
plt.ylabel("Merged fixes with zero human commits (%)"); plt.ylim(0, 100)
plt.title("NC3: how often a merged agent fix needed\nno human code at all (cloud agents)")
plt.grid(axis="y", alpha=.3); save("nc3.png")

# ---------- NC4: skill vs task mix (numbers from analysis_nc4_decomp.py) ----------
ags = ["Claude_Code", "Cursor", "Copilot", "Devin"]
rate = [-4.4, 5.4, 18.3, -10.9]; comp = [-2.0, 0.2, 0.1, -4.7]
x = np.arange(len(ags)); w = .38
plt.figure(figsize=(8.5, 4.8))
plt.bar(x - w/2, rate, w, label="Real change in skill", color="#d62728")
plt.bar(x + w/2, comp, w, label="Change in task difficulty", color="#1f77b4")
plt.axhline(0, color="k", lw=.8); plt.xticks(x, ags)
plt.ylabel("Change in rejection rate (pts)")
plt.title("NC4: each agent's change is mostly real, not a shift\nin what it was asked to fix (early 2025 → late 2025)")
plt.legend(); plt.grid(axis="y", alpha=.3); save("nc4.png")

# ---------- NC5: fast-merge trend (recomputed) ----------
d5 = cl[cl.closed & cl.mature].copy()
d5["decision_dt"] = pd.to_datetime(np.where(d5.merged, d5.merged_at, d5.closed_at), utc=True)
d5["lat_h"] = (d5.decision_dt - d5.created_dt).dt.total_seconds() / 3600
d5 = d5[d5.lat_h.notna() & (d5.lat_h >= 0)]
d5["fast"] = d5.merged & (d5.lat_h <= 10/60)
m5 = d5[d5.matched & d5.merged & d5.group.isin(["agent", "human"])]
plt.figure(figsize=(9, 4.8))
for g, col in [("agent", AGENT_C), ("human", HUMAN_C)]:
    s = m5[m5.group == g].groupby("month")["fast"].agg(["size", "mean"])
    s = s[s["size"] >= 100]["mean"] * 100
    plt.plot(s.index, s.values, "o-", color=col, label=f"{g} fixes")
plt.ylabel("Merges decided within 10 minutes (%)"); plt.ylim(0)
plt.title("NC5: instant merges of agent fixes are becoming less common")
plt.legend(); plt.grid(alpha=.3); plt.xticks(rotation=60); save("nc5.png")

# ---------- NC6: retention curve (recomputed) ----------
ag6 = cl[cl.group == "agent"].sort_values("created_dt").copy()
ag6["decision_dt"] = pd.to_datetime(np.where(ag6.merged, ag6.merged_at, ag6.closed_at), utc=True)
first = ag6.groupby("repo_id").first()
first = first[first.closed & first.decision_dt.notna()]
first = first[first.decision_dt <= END - pd.Timedelta(days=180)]
nxt = ag6.merge(first[["decision_dt"]].rename(columns={"decision_dt": "dec0"}),
                left_on="repo_id", right_index=True, how="inner")
nxt = nxt[nxt.created_dt > nxt.dec0]
first["days_to_next"] = (nxt.created_dt - nxt.dec0).dt.days.groupby(nxt.repo_id).min()
xs = [7, 14, 30, 60, 90, 120, 180]
plt.figure(figsize=(8.5, 4.8))
for imp_, col, lab in [(True, "#2ca02c", "first fix was accepted"), (False, AGENT_C, "first fix was rejected")]:
    s = first[first.merged == imp_]
    plt.plot(xs, [100 * (s.days_to_next <= x).mean() for x in xs], "o-", color=col, label=lab)
plt.xlabel("Days after the first fix was decided")
plt.ylabel("Projects that used an agent again (%)")
plt.title("NC6: a rejected first fix makes projects far less likely\nto try an agent again")
plt.legend(); plt.grid(alpha=.3); plt.ylim(0); save("nc6.png")

# ---------- NC7: durability (numbers from analysis_nc7_refix.py) ----------
x = np.arange(2); w = .38
plt.figure(figsize=(8, 4.8))
plt.bar(x - w/2, [77.1, 29.5], w, label="agent fixes", color=AGENT_C)
plt.bar(x + w/2, [77.1, 29.9], w, label="human fixes", color=HUMAN_C)
for xi, (a, h) in zip(x, [(77.1, 77.1), (29.5, 29.9)]):
    plt.text(xi - w/2, a + 1, f"{a:.0f}%", ha="center", fontweight="bold")
    plt.text(xi + w/2, h + 1, f"{h:.0f}%", ha="center", fontweight="bold")
plt.xticks(x, ["All files\n(busy files inflate this)", "Quiet files only\n(the honest comparison)"])
plt.ylabel("Fixed code touched again by a fix within 90 days (%)")
plt.title("NC7: agent fixes last just as long as human fixes")
plt.legend(); plt.grid(axis="y", alpha=.3); save("nc7.png")

# ---------- NC8: abandonment trend (recomputed) ----------
d8 = cl[cl.created_dt <= END - pd.Timedelta(days=60)].copy()
dec = d8[["merged_at", "closed_at"]].min(axis=1)
d8["undecided"] = ~(dec.notna() & ((dec - d8.created_dt).dt.days <= 60))
m8 = d8[d8.matched & d8.group.isin(["agent", "human"])]
plt.figure(figsize=(9, 4.8))
for g, col in [("agent", AGENT_C), ("human", HUMAN_C)]:
    s = m8[m8.group == g].groupby("month")["undecided"].agg(["size", "mean"])
    s = s[s["size"] >= 100]["mean"] * 100
    plt.plot(s.index, s.values, "o-", color=col, label=f"{g} fixes")
plt.ylabel("Still waiting for a decision after 60 days (%)"); plt.ylim(0)
plt.title("NC8: more and more fixes are simply left without an answer")
plt.legend(); plt.grid(alpha=.3); plt.xticks(rotation=60); save("nc8.png")

# ---------- NC9: head-to-head matrix (recomputed) ----------
AG = ["Claude_Code", "Cursor", "Copilot", "Devin"]
ag9 = clean[clean.group == "agent"].sort_values("created_dt")
g9 = ag9.groupby("repo_id").agg(n=("id", "size"), k=("agent", "nunique"))
d9 = ag9[ag9.repo_id.isin(g9[(g9.n >= 10) & (g9.k >= 2)].index)].copy()
d9["half"] = np.where(d9.groupby("repo_id").cumcount() < d9.groupby("repo_id")["id"].transform("size") // 2,
                      "early", "late")
dom = d9[d9.half == "late"].groupby("repo_id")["agent"].agg(lambda s: s.value_counts().idxmax())
eset = d9[d9.half == "early"].groupby("repo_id")["agent"].agg(set)
hh = pd.concat([eset.rename("early"), dom.rename("winner")], axis=1).dropna()
W = pd.DataFrame(0, index=AG, columns=AG)
for a, b in itertools.permutations(AG, 2):
    both = hh[hh.early.map(lambda s: a in s and b in s)]
    W.loc[a, b] = int((both.winner == a).sum())
# one row per matchup, split like an election result: winner's share left, loser's right
COL = {"Claude_Code": "#9467bd", "Cursor": "#2ca02c", "Copilot": "#ff7f0e", "Devin": "#8c564b"}
pairs = []
for a, b in itertools.combinations(AG, 2):
    wa, wb = W.loc[a, b], W.loc[b, a]
    if wa + wb == 0: continue
    if wb > wa: a, b, wa, wb = b, a, wb, wa          # winner on the left
    pairs.append((a, b, wa, wb))
pairs.sort(key=lambda p: p[2] + p[3], reverse=True)   # biggest samples on top
fig, ax = plt.subplots(figsize=(9.5, 5))
for i, (a, b, wa, wb) in enumerate(pairs):
    y = len(pairs) - 1 - i; n = wa + wb
    pa = 100 * wa / n
    ax.barh(y, pa, color=COL[a], height=.62)
    ax.barh(y, 100 - pa, left=pa, color=COL[b], height=.62, alpha=.45)
    ax.text(1.5, y, f"{a.replace('_', ' ')}  {pa:.0f}%", va="center", ha="left",
            color="white", fontsize=13, fontweight="bold")
    ax.text(98.5, y, f"{100 - pa:.0f}%  {b.replace('_', ' ')}", va="center", ha="right",
            color="#1a1a1a", fontsize=13)
    ax.text(101.5, y, f"{n} projects", va="center", ha="left", color="#555", fontsize=11)
ax.axvline(50, color="k", ls="--", lw=1, alpha=.6)
ax.text(50, len(pairs) - .25, "50–50", ha="center", fontsize=11, color="#444")
ax.set_xlim(0, 114); ax.set_ylim(-.6, len(pairs) - .2)
ax.set_yticks([]); ax.set_xticks([])
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("NC9: when two agents shared a project, which one\nthe project ended up keeping", pad=14)
save("nc9.png")

# ---------- assemble HTML ----------
def b64(name):
    with open(os.path.join(SIMP, name), "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def fig(name, caption):
    return (f'<figure><img src="data:image/png;base64,{b64(name)}" alt="{name}">'
            f'<figcaption>{caption}</figcaption></figure>')

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>How AI Coding Agents Fix Bugs Over Time — Nine New Findings</title>
<style>
  body {{ font-family: Georgia, 'Times New Roman', serif; max-width: 780px; margin: 40px auto;
          padding: 0 24px; color: #1a1a1a; line-height: 1.65; font-size: 17px; }}
  h1 {{ font-size: 25px; border-bottom: 2px solid #333; padding-bottom: 8px; line-height: 1.3; }}
  h2 {{ font-size: 19px; margin-top: 38px; color: #111; }}
  p {{ margin: 14px 0; }}
  figure {{ text-align: center; margin: 26px 0; }}
  figure img {{ max-width: 100%; border: 1px solid #ddd; }}
  figcaption {{ font-size: 14px; color: #555; text-align: left; margin-top: 10px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 15px; margin: 16px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 7px 11px; text-align: left; vertical-align: top; }}
  th {{ background: #f2f2f2; }}
  code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 14px;
          font-family: Consolas, 'Courier New', monospace; }}
  .keyidea {{ background: #f7f9fc; border-left: 4px solid #4a6fa5; padding: 14px 18px;
              margin: 18px 0; font-size: 16px; }}
  .keyidea strong {{ color: #2c4a73; }}
  ul {{ margin: 14px 0; }} li {{ margin: 7px 0; }}
  @media print {{ body {{ margin: 0; font-size: 12pt; }} }}
</style></head><body>

<h1>How AI Coding Agents Fix Bugs Over Time — Nine New Findings</h1>

<p><strong>What this page is.</strong> Beyond the eight research questions described in the study
approach, we ran nine further analyses (NC1&ndash;NC9, "new contributions") on the same dataset:
422,618 bug-fix pull requests (PRs) from December 2024 to February 2026 &mdash; 121,832 written by AI
agents (Claude Code, Cursor, Copilot, Devin) and 300,786 by human developers. As before, every
agent-vs-human comparison is made only <em>inside projects where both contribute</em>, so we compare
like with like.</p>

<h2>The nine findings at a glance</h2>
<table>
<tr><th>NC</th><th>Question</th><th>Answer in one line</th></tr>
<tr><td>NC1</td><td>Do instruction files (e.g. <code>CLAUDE.md</code>) make agent fixes better?</td>
    <td>No measurable effect on acceptance &mdash; but fixes get bigger and include more tests.</td></tr>
<tr><td>NC2</td><td>Can we tell in advance that a fix will be rejected?</td>
    <td>Fairly well (score 0.76 of 1.0); the project's history matters most.</td></tr>
<tr><td>NC3</td><td>How often does a merged agent fix need no human code at all?</td>
    <td>About 4 out of 5 times (cloud agents).</td></tr>
<tr><td>NC4</td><td>Did agents really change, or just get easier/harder tasks?</td>
    <td>The changes are real: Claude Code truly improved, Copilot truly declined.</td></tr>
<tr><td>NC5</td><td>Are agent fixes waved through without review?</td>
    <td>They are decided 4&times; faster than human fixes, but scrutiny is <em>increasing</em>.</td></tr>
<tr><td>NC6</td><td>Does a bad first experience put projects off agents?</td>
    <td>Yes: retention drops from 60% to 40% if the first fix is rejected.</td></tr>
<tr><td>NC7</td><td>Do agent fixes hold up as well as human fixes?</td>
    <td>Yes &mdash; on a fair measure they are re-fixed at the same rate (29.5% vs 29.9%).</td></tr>
<tr><td>NC8</td><td>How many fixes get no answer at all?</td>
    <td>4&ndash;5%, and rising fast for agents (2% &rarr; 8% over the window).</td></tr>
<tr><td>NC9</td><td>When two agents share a project, who wins?</td>
    <td>Projects settle on one agent, usually the one that performed better early.</td></tr>
</table>

<h2>NC1 — Do instruction files help agents?</h2>
<p>Developers can add an instruction file (such as <code>CLAUDE.md</code> or
<code>copilot-instructions.md</code>) that tells an agent how to work in their project. We compared
351 projects before and after they adopted one, against 500 similar projects that never did
&mdash; the standard "difference-in-differences" design, which removes the effect of time itself
and of each project's own habits.</p>
<p>The result is a null: after adoption, rejection moves by <strong>+1.4 points</strong>, with an
uncertainty range (&minus;1.2 to +4.0) that comfortably includes zero. What <em>does</em> change is
the work itself: after adoption, agent fixes become significantly <strong>larger</strong> and
<strong>include tests more often</strong>. Instruction files appear to change <em>how</em> agents
work, not how often their work is accepted.</p>
{fig("nc1.png", "<strong>NC1.</strong> Rejection of agent fixes in adopting projects, before vs "
 "after adoption. The small rise is not statistically distinguishable from zero once time and "
 "project effects are removed (+1.4 pts, range &minus;1.2 to +4.0). A placebo check &mdash; "
 "pretending adoption happened four months earlier &mdash; correctly finds nothing.")}

<h2>NC2 — Can a wasted fix be predicted in advance?</h2>
<p>Every rejected PR is wasted effort for the maintainer who reviews it. We trained a model on
fixes from the past and asked it to predict rejections in a <em>future</em> it had never seen
(training on data up to October 2025, testing on October&ndash;December 2025). It scores
<strong>0.76</strong> on a scale where 0.5 is a coin toss and 1.0 is perfect &mdash; good enough to
be useful for triage, e.g. flagging risky PRs for closer review.</p>
<p>The strongest signal is not the fix itself but the <strong>project's own history</strong>: how
often that project rejected agent fixes before. Next come <em>which agent</em> wrote the fix and how
long its description is. Whether the fix includes a test barely matters.</p>
{fig("nc2.png", "<strong>NC2.</strong> The six most useful signals for predicting rejection, "
 "measured by how much the model's score drops when each signal is hidden from it.")}

<h2>NC3 — How autonomous are agents really?</h2>
<p>A merged agent PR does not necessarily mean the agent did all the work: a human may have pushed
follow-up commits to finish the job. We counted, for every merged agent fix, whether any commit in
it was authored by a human. This is only measurable for the <em>cloud</em> agents (Copilot, Devin),
which commit under their own bot identity; Cursor and Claude Code run on the developer's machine and
commit under the developer's name, so their fixes cannot be separated this way.</p>
<p><strong>81%</strong> of merged cloud-agent fixes contain zero human commits (Copilot 78%, Devin
90%). Put the other way: about <strong>1 in 5</strong> "agent" fixes needed a human to step in
before it could be merged. Autonomy is lowest on network and concurrency bugs (~78%) and highest
on UI and authentication fixes (~86%).</p>
{fig("nc3.png", "<strong>NC3.</strong> Share of merged fixes with no human commit at all, for the "
 "two agents whose commits are reliably identifiable.")}

<h2>NC4 — Did agents change, or did their tasks change?</h2>
<p>Earlier we saw that Claude Code's rejection rate fell during 2025 while Copilot's rose. But that
could be an illusion: maybe Claude Code was simply handed easier bugs later on. We split each
agent's change into two parts: the part explained by a <strong>shift in task mix</strong> (bug type
and fix size) and the part that remains at a <strong>fixed task mix</strong> &mdash; the real change.</p>
<p>The changes are mostly real. Claude Code improved by 5.8 points, of which 4.4 is genuine; Copilot
worsened by 17.4 points, of which 18.3 is genuine (its task mix actually got slightly easier);
Devin's 12-point improvement is also mostly genuine.</p>
{fig("nc4.png", "<strong>NC4.</strong> Each agent's change in rejection rate from early 2025 to "
 "late 2025, split into the part that reflects real performance (red) and the part explained by "
 "a change in what it was asked to fix (blue).")}

<h2>NC5 — Are agent fixes rubber-stamped?</h2>
<p>Inside the same projects, agent fixes are decided much faster than human ones: the median agent
fix is merged in <strong>1.5 hours</strong>, against <strong>6.3 hours</strong> for a human fix.
One in five agent merges (<strong>21%</strong>) happens within <strong>10 minutes</strong> of the PR
being opened &mdash; too fast for any real review &mdash; versus 9% for human fixes.</p>
<p>The natural worry is that this is getting worse: that maintainers trust agents more and check
them less. The data says the opposite. The share of instant merges is <strong>falling</strong>
(21% &rarr; 16% over the window) and the median decision time is slowly rising. The sharpest
correction is Copilot's: its instant-merge share collapsed from 22% to 8%. Scrutiny of agent code
is increasing, not eroding.</p>
{fig("nc5.png", "<strong>NC5.</strong> Share of merged fixes that were accepted within 10 minutes, "
 "month by month, inside the shared projects. Note: we cannot tell a self-merge from a reviewer "
 "merging instantly &mdash; both count as a fast merge.")}

<h2>NC6 — First impressions decide adoption</h2>
<p>For each of 8,987 projects we found its <em>first ever</em> agent bug-fix PR and asked: did the
project use an agent again within 90 days? If the first fix was <strong>accepted</strong>,
<strong>60%</strong> of projects came back. If it was <strong>rejected</strong>, only
<strong>40%</strong> did.</p>
<p>The effect differs sharply by agent. Devin suffers most: after a rejected first fix, only 26% of
projects try again (versus 59% after an accepted one). Claude Code is nearly immune &mdash; 75%
return even after a rejection, barely below the 79% after a success &mdash; suggesting its users
are committed to the tool rather than evaluating it on a single trial.</p>
{fig("nc6.png", "<strong>NC6.</strong> Share of projects that submitted another agent fix, by days "
 "since the first fix was decided, split by how that first fix ended. The gap opens immediately "
 "and never closes. This is observational: projects whose first fix merges may differ in other "
 "ways from projects whose first fix is rejected.")}

<h2>NC7 — Do agent fixes hold up?</h2>
<p>A fix that gets merged but doesn't actually solve the problem will soon be followed by another
fix to the same code. So we measured: after a merged fix, how often does a <em>later</em> bug-fix PR
touch the same source file within 90 days?</p>
<p>Measured naively over all files, both agents and humans land at 77% &mdash; but that number
mostly reflects busy files that change all the time anyway. The honest comparison uses
<strong>quiet files</strong> (files rarely touched by fixes), where a re-touch genuinely suggests
the bug came back. There, agent fixes are re-fixed <strong>29.5%</strong> of the time and human
fixes <strong>29.9%</strong> &mdash; statistically identical. Agent fixes last just as long as
human fixes. This replaces our weaker revert-based measure (RQ4), where events were too rare
(~0.1%) to support a confident comparison.</p>
{fig("nc7.png", "<strong>NC7.</strong> How often fixed code is touched again by another fix within "
 "90 days. Left: all files (inflated by busy files). Right: quiet files only &mdash; the fair "
 "comparison &mdash; where agents and humans are indistinguishable.")}

<h2>NC8 — The fixes that get no answer</h2>
<p>Our rejection rate only counts PRs that received a decision. But some PRs are simply ignored
&mdash; never merged, never closed. Giving every PR exactly the same 60-day observation window, we
find <strong>4.4%</strong> of agent fixes and <strong>5.0%</strong> of human fixes still undecided
after 60 days. Modest &mdash; but the agent share <strong>quadrupled</strong> over the study window,
from about 2% to about 8%, with Copilot the most-ignored agent (11% overall).</p>
<p>This matters for reading the headline numbers: if ignored PRs were counted as rejections, agent
rejection would rise from 16.9% to 20.6%. The seemingly flat rejection trend hides a growing pile
of unanswered agent work.</p>
{fig("nc8.png", "<strong>NC8.</strong> Share of fixes still undecided 60 days after being opened, "
 "month by month, inside the shared projects. Every PR gets the same 60-day window, so older PRs "
 "have no advantage.")}

<h2>NC9 — When two agents share a project, who wins?</h2>
<p>In 526 projects, at least two different agents submitted bug fixes. Do such projects keep
experimenting, or settle on a favourite? They settle: among projects where two agents genuinely
competed early on, the agent mix becomes markedly more concentrated over time, and <strong>36%</strong>
end up using a single agent exclusively.</p>
<p>Head-to-head, <strong>Claude Code is the most common keeper</strong>: it wins 65% of projects it
shared with Copilot and 81% of those shared with Devin. Devin loses every matchup (12&ndash;21%).
And the choice is partly earned: in projects where two agents both had enough early fixes to
compare, the agent with the <em>lower early rejection rate</em> ended up as the keeper
<strong>59%</strong> of the time &mdash; more than chance.</p>
{fig("nc9.png", "<strong>NC9.</strong> Each bar is one head-to-head matchup: projects where both "
 "agents submitted fixes early on, split by which agent the project mostly used in the end. The "
 "agent that won the matchup is shown on the left of each bar. The bottom matchup rests on only "
 "8 projects and should not be over-read.")}

<h2>Read with care</h2>
<ul>
<li><strong>NC1</strong> finds no effect on <em>acceptance</em>; instruction files may still help in
ways we cannot see here (fewer retries, better code style, happier developers).</li>
<li><strong>NC3</strong> covers only the two cloud agents; local agents (Cursor, Claude Code) blend
into the developer's own commits by design.</li>
<li><strong>NC5</strong> cannot distinguish a self-merge from a reviewer merging instantly.</li>
<li><strong>NC6</strong> is observational, not a controlled experiment.</li>
<li><strong>NC7</strong>'s all-files number is dominated by file churn; rely on the quiet-files
comparison.</li>
<li><strong>NC9</strong>'s smallest match-ups rest on few projects.</li>
</ul>

</body></html>
"""

out = os.path.join(BASE, "NC_FINDINGS.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print("saved", out, f"({len(html)//1024} KB)")
