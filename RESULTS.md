# Results — Agentic Bug-Fixing Over Time

Temporal extension of the MSR '26 rejection paper, on the
[GitHub-Agentic-PR-Dataset](https://huggingface.co/datasets/mabujadallah/GitHub-Agentic-PR-Dataset):
422,618 bug-fix PRs, Dec 2024 – Feb 2026 (15 months); 121,832 by agents (Claude_Code, Cursor, Copilot, Devin),
300,786 by humans. Full analysis + figures in `analysis_trio.ipynb`.

**Fair comparison:** all agent-vs-human results use only the **1,218 repos that contain both** agent and human fixes
(within-project comparison). RQ2a/RQ2b/RQ3 are agent-only and use all data.

## Findings

- **RQ1a — rejection over time (matched).** Inside the same projects, agent fixes are rejected a bit more than human
  fixes (~**18%** vs ~**15%**). Per agent the trends diverge: Claude_Code improves over time while Copilot worsens.
  → `figures/rq1a_matched.png`
- **RQ1b — fix size over time (matched).** Agent fixes are slightly larger (median **44** vs **33** lines) and have been
  getting tighter recently. → `figures/rq1b_matched.png`
- **RQ2a — agent adoption over time.** The agent mix shifts fast: Claude_Code + Cursor dominated early; **Copilot jumped
  from ~2% to ~30%** after its May 2025 launch; **Devin faded** from ~27% to ~5%. → `figures/rq2a_agent_share_over_time.png`
- **RQ2b — switching agents (relay).** Of 653 issues whose first agent fix was rejected, ~70% were eventually fixed —
  but almost always by **retrying the same agent (45%)** or a **human takeover (23%)**; a **different agent succeeds only
  1.8%** of the time. "Just try a better model" is rare. → `figures/rq2b_recovery_breakdown.png`
- **RQ3 — instruction files over time.** Developers increasingly pair bug-fixes with agent-instruction files
  (`CLAUDE.md`, `copilot-instructions.md`, `AGENTS.md`, `.cursorrules`, …): the share of agent fix PRs touching one rose
  from ~**0.03%** (late 2024) to ~**1.9%** (mid 2025), then ~1.5% (1,817 PRs, 740 repos). Lower bound — only counts
  instruction edits made *inside* a fix PR. → `figures/rq3_instructions.png`
- **RQ4 — durability (matched).** Agent fixes are reverted **less** than human fixes (~**0.07%** vs ~**0.13%**); reverts
  are rare overall, so fixes are durable. Lower bound (PR-routed reverts only). → `figures/rq4_matched.png`
- **RQ5 — bug-type difficulty (matched).** Agents struggle most with **concurrency (+10 pts vs humans), type errors (+9),
  and security (+7)**; on easy/cosmetic fixes (typos/docs) they match or beat humans. → `figures/rq5_matched.png`
- **RQ6 — does a test help? (matched).** Counter-intuitively, fixes with a test are rejected **more** (agents +~5 pts);
  the effect **survives** controlling for fix size. → `figures/rq6_matched.png`, `figures/rq6c_matched.png`

## New contributions (NC5–NC9)

Scripts `analysis_nc5_*.py` … `analysis_nc9_*.py`; same cleaned table + timestamps/file diffs from the raw tables.

- **NC5 — rubber-stamping / time-to-decision (matched).** Agent fixes are decided dramatically faster than human
  fixes: median time-to-merge **1.5 h vs 6.3 h**, and **20.7% vs 9.1%** of merges land within **10 minutes**
  (effectively no review). But the trend goes *against* complacency: agent fast-merges are **declining**
  (21%→16%, ρ=−0.52) and time-to-merge is rising (ρ=+0.69, p=0.01) — scrutiny is *increasing*. Copilot's
  fast-merge share collapsed **22%→8%**. → `figures/nc5_latency.png`
- **NC6 — first impressions drive adoption (all agents).** Among 8,987 repos with a decided first agent fix,
  90-day retention (repo submits another agent fix) is **60% if the first fix merged vs 40% if rejected**
  (p≈10⁻⁵⁵; 59% vs 35% on "clean" firsts with nothing else pending). The effect is largest for **Devin
  (59% vs 26%)** and almost absent for **Claude_Code (79% vs 75%)** — a bad first impression costs an agent the
  repo. → `figures/nc6_retention.png`
- **NC7 — re-fix recurrence (matched, merged fixes).** A durability measure with signal (reverts are ~0.1%):
  a fix is "re-fixed" if a later fix PR touches the same non-test code file within 90 days. Overall agent =
  human (**77.1% = 77.1%**, dominated by hot-file churn); on **cold files** (≤3 fix PRs ever) it is **29.5% vs
  29.9%** (p=0.63) — *agent fixes are as durable as human fixes*. Gaps by bug type mirror RQ5: agents worse on
  performance/build/concurrency (+3–4 pts), better on ui/docs (−5–6 pts). Within-repo paired: agents +4 pts
  (p<0.001). → `figures/nc7_refix.png`
- **NC8 — silent abandonment (uniform 60-day window).** Fix PRs still undecided 60 days after creation: agents
  **4.4%** vs humans **5.0%** (matched). But the share is **rising fast for agents (2.1%→8.1%**, ρ=+0.79,
  p=0.001; humans 4.2%→7.3%), and **Copilot is worst (10.8%** overall**)**. Counting undecided as rejected lifts
  agent rejection 16.9%→20.6% — the flat RQ1a trend hides growing attention debt. → `figures/nc8_abandonment.png`
- **NC9 — within-repo agent competition (526 multi-agent repos).** Repos with genuine early competition
  **consolidate** (HHI 0.74→0.85, p<0.001; 36% end on a single agent). Head-to-head, **Claude_Code wins repos
  vs Copilot 65%** and vs Devin 81%; Devin loses every matchup (12–21%). And it is partly **merit**: the agent
  with the lower early rejection rate ends up dominant **59%** of the time (binomial p=0.037). →
  `figures/nc9_competition.png`

## Why our rejection rate (~16%) differs from the paper (46%)

Verified in `paper_comparison.ipynb`. It is **not** project popularity, time window, or collection timing:
- The paper's 46.4% reproduces exactly (`hao-li/AIDev` `pull_request.parquet` + `pr_task_type` fix labels → 3,225 closed
  fixes, Devin/Copilot-heavy).
- The **same PRs** (matched by URL) are **~48%** rejected in our data too — outcomes did not change.
- The gap is purely the **fix set**: our `type=fix` + fresh re-collection labels ~**7× more** fix PRs in the same
  repos+window, mostly easy-merging, diluting to ~16%. (Our own AIDev-style LLM classifier gives ~14% — confirming it is
  the population, not the classifier.)

**Implication:** the absolute rejection rate is not comparable to the paper; the contribution is the **relative** and
**temporal** findings above.

## Limitations

- Agent and human PRs come from different projects → comparisons use matched repos only.
- RQ2b sees only fixes that explicitly link an issue (~18%); RQ3 sees only instruction edits inside fix PRs (lower bound);
  RQ4 revert counts are a lower bound.
- Bug types (RQ5) come from a keyword heuristic on title/body — worth spot-checking before publication.
- NC5 fast-merges cannot distinguish self-merge from reviewer rubber-stamp (no reviewer data).
- NC6 is observational: repos whose first fix merges may differ from those whose first fix is rejected.
- NC7's 90-day file-touch measure is dominated by hot-file churn; the cold-file subset is the cleaner read.
