"""Draws a clean, paper-style methodology diagram for analysis_trio.ipynb."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

# ---- palette (light fill / saturated edge) --------------------------------
C_DATA_F, C_DATA_E   = "#e8eef7", "#2f5e9e"   # dataset / agent / human
C_FAIR_F, C_FAIR_E   = "#fdebd0", "#cf8a2e"   # fairness check
C_MSET_F, C_MSET_E   = "#d5f0dd", "#1e8449"   # matched design
C_ASET_F, C_ASET_E   = "#fadbd8", "#c0392b"   # agent-only design
C_MRQ_F              = "#eafaf0"
C_ARQ_F              = "#fdeaea"
C_DL_F,   C_DL_E     = "#ece7f6", "#6c4ea3"   # data layers
C_MET_F,  C_MET_E    = "#eef0f1", "#7f8c8d"   # methods
INK   = "#1a1a1a"
GREY  = "#5b6168"

fig, ax = plt.subplots(figsize=(13, 9.6))
ax.set_xlim(0, 13); ax.set_ylim(0, 10); ax.axis("off")

# thin outer frame for a printed-figure feel
ax.add_patch(FancyBboxPatch((0.15, 0.12), 12.7, 9.76,
             boxstyle="round,pad=0,rounding_size=0.12",
             fc="white", ec="#cfd4da", lw=1.2, zorder=0))

def box(x, y, w, h, text, fc, ec, fs=10, weight="normal", align="center", lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.04",
                 fc=fc, ec=ec, lw=lw, zorder=2))
    if align == "center":
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, weight=weight, color=INK, zorder=3)
    else:
        ax.text(x + 0.22, y + h/2, text, ha="left", va="center",
                fontsize=fs, weight=weight, color=INK, zorder=3)

def rq_block(x, y, w, h, title, lines, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.04",
                 fc=fc, ec=ec, lw=1.6, zorder=2))
    ax.text(x + w/2, y + h - 0.27, title, ha="center", va="center",
            fontsize=9.5, weight="bold", color=ec, zorder=3)
    body = "\n".join(lines)
    ax.text(x + 0.28, y + h - 0.62, body, ha="left", va="top",
            fontsize=9, color=INK, zorder=3, linespacing=1.45)

def arrow(x1, y1, x2, y2, color=GREY, lw=1.8, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=15, color=color, lw=lw, ls=ls, zorder=1,
                 connectionstyle=f"arc3,rad={rad}"))

# ---- title ----------------------------------------------------------------
ax.text(6.5, 9.55, "How AI Coding Agents Fix Bugs Over Time — Study Design",
        ha="center", va="center", fontsize=16, weight="bold", color=INK)
ax.text(6.5, 9.18, "Longitudinal follow-up to the MSR'26 rejection paper   ·   GitHub-Agentic-PR-Dataset",
        ha="center", va="center", fontsize=10.5, color=GREY, style="italic")

# ---- 1. dataset -----------------------------------------------------------
box(3.95, 8.30, 5.1, 0.62, "422,618 bug-fix PRs   ·   Dec 2024 – Feb 2026  (15 months)",
    C_DATA_F, C_DATA_E, fs=10.5, weight="bold")

# ---- 2. split agent / human ----------------------------------------------
arrow(5.7, 8.30, 4.35, 7.92)
arrow(7.3, 8.30, 8.65, 7.92)
box(2.55, 7.30, 3.4, 0.62, "Agent fixes · 121,832",       C_DATA_F, C_DATA_E, fs=10)
ax.text(4.25, 7.16, "Claude Code · Cursor · Copilot · Devin",
        ha="center", va="top", fontsize=7.8, color=GREY, style="italic")
box(7.05, 7.30, 3.4, 0.62, "Human fixes · 300,786",       C_DATA_F, C_DATA_E, fs=10)

# ---- 3a. agent-only design (left lane) ------------------------------------
arrow(3.8, 7.30, 3.35, 5.88, color=C_ASET_E, lw=1.9)
box(0.70, 4.95, 5.20, 0.92,
    "AGENT-ONLY DESIGN\nfull agent data  ·  no human baseline",
    C_ASET_F, C_ASET_E, fs=10, weight="bold")

# ---- 3. fairness check (right, motivates matched design) ------------------
arrow(5.2, 7.30, 6.9, 6.97, color=GREY)
arrow(8.4, 7.30, 9.7, 6.97, color=GREY)
box(5.95, 6.18, 6.35, 0.80,
    "FAIRNESS CHECK — the project, not just the author, drives rejection\n"
    "shared repos 18.1% / 15.1%      vs      single-type repos 14.7% / 10.9%",
    C_FAIR_F, C_FAIR_E, fs=9.2)

# ---- 3b. matched design (right lane) --------------------------------------
arrow(9.12, 6.18, 9.12, 5.88, color=C_MSET_E, lw=1.9)
box(5.95, 4.95, 6.35, 0.92,
    "MATCHED DESIGN — fair agent vs human\n"
    "1,218 repos with BOTH kinds  ·  Agent 47,925 · Human 287,358",
    C_MSET_F, C_MSET_E, fs=10, weight="bold")

# ---- research questions ---------------------------------------------------
arrow(3.30, 4.95, 3.30, 4.47, color=C_ASET_E)
arrow(9.12, 4.95, 9.12, 4.47, color=C_MSET_E)

rq_block(0.70, 2.95, 5.20, 1.50, "AGENT-ONLY QUESTIONS",
    ["RQ2a   which agent is used over time",
     "RQ2b   does switching agents rescue",
     "             a rejected fix?",
     "RQ3     instruction-file adoption"],
    C_ARQ_F, C_ASET_E)

rq_block(5.95, 2.95, 6.35, 1.50, "MATCHED QUESTIONS  (agent vs human)",
    ["RQ1a  rejection rate over time          RQ5  rejection by bug type",
     "RQ1b  fix size (code churn)               RQ6  does adding a test help?",
     "RQ4    reverts after merge"],
    C_MRQ_F, C_MSET_E)

# ---- shared data layers (feeds every RQ) ----------------------------------
arrow(3.30, 2.78, 3.30, 2.93, color=C_DL_E, lw=1.4)
arrow(9.12, 2.78, 9.12, 2.93, color=C_DL_E, lw=1.4)
ax.add_patch(FancyBboxPatch((0.70, 2.00), 11.60, 0.78,
             boxstyle="round,pad=0.02,rounding_size=0.04",
             fc=C_DL_F, ec=C_DL_E, lw=1.6, zorder=2))
ax.text(6.5, 2.56, "SHARED DATA LAYERS  —  derived once, used by every RQ",
        ha="center", va="center", fontsize=9.4, weight="bold", color=C_DL_E, zorder=3)
ax.text(6.5, 2.23, "PR metadata    ·    file changes → churn / tests / instruction files    "
        "·    commit messages → reverts",
        ha="center", va="center", fontsize=9.0, color=INK, zorder=3)

# ---- methods --------------------------------------------------------------
box(0.70, 1.22, 11.60, 0.60,
    "METHODS:   monthly trend lines   ·   Spearman trend test   ·   "
    "χ² for test-inclusion (RQ6)   ·   churn-binned robustness re-check",
    C_MET_F, C_MET_E, fs=9.0, weight="bold")

# ---- definitions / limits footer -----------------------------------------
ax.text(6.5, 0.82,
        "Conservative definitions:  rejected = closed & never merged   ·   "
        "rate computed over DECIDED PRs only (open PRs dropped)",
        ha="center", va="center", fontsize=8.4, color=INK)
ax.text(6.5, 0.50,
        "Stated limits:  issue-linking covers ~18% of PRs (RQ2b)   ·   "
        "reverts are a lower bound   ·   RQ5 bug types are keyword-derived",
        ha="center", va="center", fontsize=8.0, color=GREY, style="italic")

out = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes\methodology_figure.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("saved", out)
