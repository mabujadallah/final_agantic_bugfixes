# Bug-Fix Agentic-PRs — Temporal Findings (Quick Trio)

**Dataset:** `mabujadallah/GitHub-Agentic-PR-Dataset` -> `fix_prs_only.parquet` (422,618 bug-fix PRs, type=fix). **Span:** 2024-12 to 2026-02 (15 months).
**Groups:** Agent = is_agent==True (121,832 PRs: Claude_Code, Cursor, Copilot, Devin); Human = is_agent==False (300,786 PRs).

## How everything is calculated (shared definitions)
- **Month** = first 7 chars of `created_at` (`YYYY-MM`), i.e. the PR's creation month.
- **Rejected** = `state=='closed'` AND `merged_at` is null/empty (closed without merge).
- **Merged** = `merged_at` is present. **Decided/closed** = `state=='closed'` (merged + rejected).
- **Rejection rate** = rejected / closed (open PRs excluded, so undecided recent PRs don't deflate it).
- **Trend test** = Spearman correlation between month index (0..N) and the monthly value; reports rho and p (monthly points are the observations).
- *Caveat:* the most recent month(s) are right-censored (slow merges still pending), so read the last point loosely.

## RQ1a — Does the rejection rate change over time?
**Answer:** Agent rejection rate goes from **13.8%** (2024-12) to **15.8%** (2026-02); mean **16.3%**, range 13.8-23.0%. Trend Spearman rho=0.04 (p=0.889). Human baseline mean **14.8%** (rho=0.93, p=0.000).
- Interpretation: no significant monotonic time trend for agents (alpha=.05).
- Figure: `figures/rq1a_rejection_over_time.png`
- How: per month, rejected/closed within each group (defs above); trend = Spearman(month-index, monthly rate).

## RQ2a — Do people switch which agent they use to fix bugs?
**Answer:** Yes, the mix shifts a lot. Share of agent bug-fix PRs (start -> end month):
  - Claude_Code: 48% -> 38%  (peak 48% in 2024-12)
  - Cursor: 42% -> 30%  (peak 42% in 2024-12)
  - Copilot: 2% -> 28%  (peak 33% in 2025-07)
  - Devin: 8% -> 5%  (peak 27% in 2025-04)
- Copilot's volume is tiny until ~2025-05 then explodes (platform launch effect); Cursor & Copilot spike together in 2025-07; Devin declines after spring 2025.
- Figure: `figures/rq2a_agent_share_over_time.png`
- How: among agent PRs, monthly count per agent / total agent PRs that month = share%.

## RQ1b — Does code churn (fix size) change over time?
**Answer:** Agent median churn: 16 (2024-12) -> 10 (2026-02); overall agent median **33** lines, human median **6**. Trend Spearman rho=0.23 (p=0.404).
- Coverage: 68.8% of fix PRs have commit-file rows; the rest treated as 0 churn (no recorded file changes).
- Figure: `figures/rq1b_churn_over_time.png`
- How: churn(PR) = sum(additions+deletions) over all its commit-file rows; monthly **median** per group.

## RQ6 — Does shipping a test correlate with acceptance?
**Agent:** with test = **17.7%** rejected (n=24,289); without test = **15.6%** (n=91,722); diff +2.1 pts; chi-square p=4.83e-15 (significant).
**Human:** with test = **16.3%** rejected (n=57,750); without test = **14.6%** (n=230,939); diff +1.7 pts; chi-square p=6.62e-25 (significant).
- Test-inclusion (agent): 14% -> 21% over the period (mean 19%).
- Figures: `figures/rq6_test_effect.png`, `figures/rq6b_test_inclusion_over_time.png`
- How: a PR 'has_test' if ANY changed file path matches the test regex `(^|/)(tests?|__tests__|spec)(/|\.)|(_test\.|test_|\.test\.|\.spec\.|_spec\.|\.tests\.)`; rejection rate compared with/without test; chi-square 2x2 (has_test x rejected) on closed PRs.


---

## RQ2b — Model-substitution / relay (does switching agent rescue a failed fix?)
- Coverage: 17.9% of fix PRs reference an issue via a closing keyword; 6,311 issues received >=2 fix attempts (multi-attempt issues).
- Issues whose FIRST fix was an agent and got rejected: **653**.
- Of those, **69.7%** were later fixed by a merged PR (recovered):
    - different agent (**substitution**): 12 (1.8%)
    - same agent (retry): 293 (44.9%)
    - human takeover: 150 (23.0%)
    - never recovered: 198 (30.3%)
- Figures: `figures/rq2b_recovery_breakdown.png`, `figures/rq2b_substitution_matrix.png`
- How: parse GitHub closing keywords (`fixes/closes/resolves #N`) from title+body; group fix PRs by (repo, issue); for issues whose earliest fix attempt was an agent PR that was rejected, the first later **merged** attempt is the 'rescuer' (same agent = retry, other agent = substitution, human = takeover).

## RQ4 — Fix durability (do merged fixes get reverted?)
- Indexed 1,156,360 commits across 326,054 fix PRs.
- Found 599 fix PRs with at least one commit reverted (via 'reverts commit <sha>').
- **Agent merged fixes reverted: 51/97,374 = 0.052%** | Human: 303/245,530 = 0.123%.
    - Claude_Code: 25/37,891 = 0.066%
    - Cursor: 21/33,218 = 0.063%
    - Copilot: 4/20,664 = 0.019%
    - Devin: 1/5,601 = 0.018%
- Figure: `figures/rq4_revert_rate.png`
- How: collect commit SHAs of all fix PRs; scan ALL PR commit messages for `reverts commit <sha>`; a merged fix PR is 'reverted' if one of its commits is referenced by a revert in a different PR. Revert rate = reverted / merged, per group. (Lower bound: only reverts done via PRs are visible.)


---

## RQ5 — Which bug types are agents good vs bad at?
- Agent fix PRs by bug type (closed only): rejection rate, median churn, human rejection baseline:
    - security       n=  3890  reject= 26.9%  churn(med)=   64  (human 21.4%)
    - typing         n=   858  reject= 21.7%  churn(med)=   29  (human 21.4%)
    - concurrency    n=  1263  reject= 20.9%  churn(med)=   64  (human 16.5%)
    - memory         n=   335  reject= 19.1%  churn(med)=   51  (human 18.0%)
    - crash          n=  2754  reject= 18.4%  churn(med)=   34  (human 17.5%)
    - network        n= 19004  reject= 18.2%  churn(med)=   24  (human 14.3%)
    - data_parse     n=  4571  reject= 17.3%  churn(med)=   51  (human 16.4%)
    - performance    n=  2951  reject= 15.9%  churn(med)=   83  (human 16.7%)
    - build_ci       n= 22886  reject= 15.7%  churn(med)=   16  (human 15.9%)
    - security_auth  n=  4889  reject= 14.9%  churn(med)=   38  (human 16.9%)
    - ui             n= 12251  reject= 14.7%  churn(med)=   51  (human 13.7%)
    - other_logic    n= 35219  reject= 14.4%  churn(med)=   43  (human 14.4%)
    - typo_doc       n=  5140  reject= 13.2%  churn(med)=   13  (human 14.1%)
- Hardest (agent): security (26.9%). Easiest: typo_doc (13.2%).
- Figure: `figures/rq5_rejection_by_type.png`
- How: keyword classifier (first match wins) over title + first 300 chars of body into 13 buckets (+'other_logic'); per type, rejection = rejected/closed, churn = median. Heuristic labels (no issue labels in data).

## RQ6 (robustness) — Is 'tests -> more rejection' just a size confound?
- Confound check: median churn WITH test = 250 vs WITHOUT = 15 lines (test PRs are larger).
    - churn 0      : +test   0.0% (n=    5) | no-test  21.9% (n= 18741) | diff -21.9pts
    - churn 1-10   : +test  14.4% (n= 1035) | no-test  12.2% (n= 22059) | diff +2.1pts
    - churn 11-50  : +test  14.5% (n= 3105) | no-test  12.0% (n= 20219) | diff +2.4pts
    - churn 51-200 : +test  18.4% (n= 6770) | no-test  15.3% (n= 15096) | diff +3.2pts
    - churn 200+   : +test  18.3% (n=13374) | no-test  17.9% (n= 15607) | diff +0.5pts
- Verdict: across comparable-size bins (n>=200/side), the test gap is +2.0 pts (size-weighted) -> **robust to size: test PRs are still rejected more even within equal-size bins**. The large raw median-churn gap (250 vs 15) shows size IS a real co-factor, but it does not fully explain the effect.
- Figure: `figures/rq6c_churn_controlled.png`
- How: agent closed fix PRs binned by churn; within each bin compare rejection rate with vs without a test.
