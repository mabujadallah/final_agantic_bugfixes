# Methodology — How AI Coding Agents Fix Bugs Over Time

A plain-language write-up of the approach behind `analysis_trio.ipynb`.

## What the study is

It's a follow-up to our MSR'26 paper. The paper looked at one snapshot of *why AI-agent
pull-request (PR) fixes get rejected*. This notebook adds **time** — it tracks how things
change month by month over 15 months — and expands the work to 8 research questions.

**The data:** 422,618 bug-fix PRs from December 2024 to February 2026, from the
[GitHub-Agentic-PR-Dataset](https://huggingface.co/datasets/mabujadallah/GitHub-Agentic-PR-Dataset).

- 121,832 were written by AI agents (Claude Code, Cursor, Copilot, Devin)
- 300,786 were written by humans

---

## Data sources

Three parquet files under `data/`, each a different "layer" of the same PRs:

| File | What it holds | Used for |
|------|---------------|----------|
| `fix_prs_only.parquet` | one row per bug-fix PR (state, dates, author, repo) | rejection, timing, agent share, issue links, bug types |
| `fix_pr_commit_details.parquet` | one row per file changed in a PR | fix size (churn), tests, instruction files |
| `pr_commits.parquet` | one row per commit (sha, message) | detecting reverts |

---

## Preprocessing (step by step)

Before any question is answered, the raw tables are cleaned and enriched. Nothing is dropped
silently — every reduction is intentional and noted below.

### 1. PR-level table (from `fix_prs_only.parquet`)
1. Load only the needed columns (`id, state, created_at, merged_at, repo_id, is_agent, agent`).
2. **Month** = first 7 characters of `created_at` (i.e. `YYYY-MM`).
3. **Drop** rows with a missing or empty month (can't be placed on the timeline).
4. **`closed`** = `state == "closed"`.
5. **`merged`** = `merged_at` is actually present. Empty strings and the text values
   `"NaT"`, `"None"`, plus true nulls are all treated as *not merged* (a common parquet
   gotcha this guards against).
6. **`rejected`** = closed **and** not merged.
7. **`grp`** = `"Agent"` if `is_agent` else `"Human"`.
8. **Matched-repo flag** (the core fairness step, see below): find repos that contain agent
   fixes and repos that contain human fixes, take the intersection (1,218 repos), and mark
   each PR `matched` if its repo is in that set.

### 2. File-level table (from `fix_pr_commit_details.parquet`)
1. Load `pr_id, filename, additions, deletions`.
2. **`lines`** = `additions + deletions` (missing counts treated as 0).
3. **`churn`** per PR = sum of `lines` across its files. PRs with **no file rows** keep churn
   as missing (NaN) and are simply **left out of the size numbers** — they are not counted as 0.
4. **`is_test`** = filename matches a test-file pattern
   (`tests/`, `__tests__`, `spec`, `_test.`, `test_`, `.test.`, `.spec.`, …).
   **`has_test`** per PR = any of its files is a test file (missing → `False`).
5. **`instr`** = filename matches an agent-instruction-file pattern
   (`CLAUDE.md`, `copilot-instructions.md`, `AGENTS.md`, `GEMINI.md`, `.cursorrules`,
   `.cursor/rules`, `.windsurfrules`, `.github/instructions/`, `CONVENTIONS.md`, `.aider`) — used for RQ3.
6. **Merge** `churn` and `has_test` back onto the PR table with a left join, so every PR keeps
   its row even if it had no file detail.

### 3. Issue-link table (for RQ2b — "does switching agents help?")
1. Reload PRs with `title` and `body`.
2. **Extract issue numbers** from the text with a regex that matches
   `close[s/d] / fix[es/ed] / resolve[s/d] #<number>`.
3. **Explode** to one row per (PR, issue) pair and sort by creation time.
4. **Group** by `(repo, issue)` and keep only issues with **≥2 fix attempts**, then look at
   what (if anything) rescued a first agent fix that was rejected.
   *Coverage note: only ~18% of PRs reference an issue this way.*

### 4. Revert detection (for RQ4 — from `pr_commits.parquet`)
1. Load `sha, pr_id, message`.
2. Build a map from **first 12 chars of each fix commit's sha → its PR**.
3. Scan commit messages for `reverts commit <sha>`; map the referenced sha back to the
   original PR (ignoring a commit that reverts itself).
   *This only catches reverts that go through PRs, so the count is a lower bound.*

### 5. Bug-type labelling (for RQ5)
1. Reload `title` and `body`.
2. Assign each fix to the **first matching** keyword category, in priority order:
   security, crash, concurrency, memory, performance, auth, ui, build/ci, typo/doc, typing,
   data/parse, network — otherwise `other_logic`.
   *Labels are keyword-derived and should be spot-checked by hand before publishing.*

---

## Key terms (kept simple)

- **Month** — the month the PR was created.
- **Merged** — the fix was accepted into the project.
- **Rejected** — the PR was closed and never merged.
- **Rejection rate** — out of the PRs that were *decided* (merged or closed), the share that
  were rejected. Still-open PRs are excluded, because their outcome is unknown.
- **Code churn** — lines a fix adds + removes (a simple size measure).
- **Has a test** — the fix changed at least one test file.

---

## The one big idea: fair comparison through "matched repos"

Agents and humans don't work on the same projects, so comparing *all* agent fixes to *all*
human fixes partly compares different projects, not the agents themselves.

The notebook shows this matters first:

- In projects with **both** agent and human fixes: agents rejected **18.1%**, humans **15.1%**.
- In projects with only **one** kind (mostly small solo projects that merge almost everything):
  agents **14.7%**, humans **10.9%**.

So the project itself moves the numbers a lot. The fix: every agent-vs-human question uses
**only the 1,218 repositories that contain both kinds of fixes** (agent n=47,925, human
n=287,358). Any remaining difference is then about agent vs human — not about which project.

Questions that are only about agents (which agent is used, switching, instruction files) use
the **full** data, since there is no human baseline to keep fair.

---

## The 8 research questions

**Matched repos (fair agent vs human):**
- **RQ1a** — Does the rejection rate change over time? *(monthly trend line, agent vs human)*
- **RQ1b** — Do fixes get bigger or smaller over time? *(median churn per month)*
- **RQ4** — Once merged, do fixes get undone? *(revert rate)*
- **RQ5** — Which bug types are agents good/bad at? *(rejection by keyword category)*
- **RQ6** — Does adding a test make a fix more likely to be accepted? *(+ size-controlled re-check)*

**All agents:**
- **RQ2a** — Do people switch which agent they use? *(monthly market share of each agent)*
- **RQ2b** — When an agent fix is rejected, does switching to another agent rescue it?
- **RQ3** — Are people adding agent instruction files over time?

---

## How conclusions are checked

- Trends over time are tested with a **Spearman** correlation.
- The test-inclusion effect (RQ6) is tested with a **χ² test**, then re-checked by comparing
  only fixes of *similar size* (within churn bins) to rule out "it's just because those fixes
  are bigger."

---

## What to keep in mind (limits)

- Agent vs human is compared **only inside the 1,218 shared repos**, so the comparison is fair.
- The "switching agents" question (RQ2b) only sees the ~18% of fixes that mention an issue.
- Revert counts are a **lower bound** — only PR-routed reverts are visible.
- Bug types (RQ5) are keyword-derived and should be spot-checked before publishing.
- The overall ~16% rejection rate is **not** directly comparable to the paper's earlier
  snapshot — it's a much larger, more recent re-collection, not a popularity effect.
