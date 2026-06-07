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
