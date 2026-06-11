"""Draws an explanatory methodology diagram for analysis_trio.ipynb."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# palette
C_DATA   = "#dfe7f3"; C_DATA_E   = "#3b6db5"
C_SPLIT  = "#f6e2c9"; C_SPLIT_E  = "#cf8a2e"
C_MATCH  = "#d9efd9"; C_MATCH_E  = "#2e8b3d"
C_AGENT  = "#f6d6d6"; C_AGENT_E  = "#c0392b"
C_RQ_M   = "#eaf3ea"; C_RQ_M_E   = "#2e8b3d"
C_RQ_A   = "#fbeaea"; C_RQ_A_E   = "#c0392b"
GREY     = "#555555"

fig, ax = plt.subplots(figsize=(14, 9.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis("off")

def box(x, y, w, h, text, fc, ec, fs=10, weight="normal", align="center", round=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={round}",
                                fc=fc, ec=ec, lw=1.6, zorder=2))
    ha = {"center": "center", "left": "left"}[align]
    tx = x + w/2 if align == "center" else x + 0.18
    ax.text(tx, y + h/2, text, ha=ha, va="center", fontsize=fs, weight=weight, zorder=3, color="#1a1a1a")

def arrow(x1, y1, x2, y2, color=GREY, style="-|>", lw=1.8, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                                 color=color, lw=lw, ls=ls, zorder=1,
                                 connectionstyle="arc3,rad=0"))

# ---- Title ----
ax.text(7, 9.75, "Methodology — How AI Coding Agents Fix Bugs Over Time",
        ha="center", va="center", fontsize=16, weight="bold")
ax.text(7, 9.32, "Longitudinal follow-up to the MSR'26 rejection paper  ·  GitHub-Agentic-PR-Dataset",
        ha="center", va="center", fontsize=10.5, color=GREY, style="italic")

# ---- 1. Dataset ----
box(4.0, 8.25, 6.0, 0.75,
    "422,618 bug-fix PRs  ·  Dec 2024 – Feb 2026 (15 months)",
    C_DATA, C_DATA_E, fs=11, weight="bold")

# ---- 2. Split agent / human ----
arrow(6.0, 8.25, 4.3, 7.55)
arrow(8.0, 8.25, 9.7, 7.55)
box(2.0, 6.85, 4.5, 0.7, "Agent fixes  ·  121,832\n(Claude Code, Cursor, Copilot, Devin)",
    C_DATA, C_DATA_E, fs=9.5)
box(7.5, 6.85, 4.5, 0.7, "Human fixes  ·  300,786",
    C_DATA, C_DATA_E, fs=9.5)

# ---- 3. Fairness check + matched filter ----
box(0.4, 5.55, 6.1, 0.85,
    "FAIRNESS CHECK: project type drives rejection\n"
    "shared 18.1%/15.1%  vs  single-type 14.7%/10.9%",
    C_SPLIT, C_SPLIT_E, fs=9)
arrow(4.25, 6.85, 3.45, 6.42)

box(0.4, 4.25, 6.1, 0.95,
    "MATCHED SET  →  1,218 repos with BOTH kinds\n"
    "Agent 47,925  ·  Human 287,358\n"
    "(controls for project; isolates agent-vs-human)",
    C_MATCH, C_MATCH_E, fs=9.5, weight="bold")
arrow(3.45, 5.55, 3.45, 5.22)

# all-agent path  (branches from the AGENT box, not human)
box(7.5, 5.55, 4.5, 0.85,
    "ALL-AGENT SET  →  full agent data\n(agent-only questions, no human baseline)",
    C_AGENT, C_AGENT_E, fs=9.5, weight="bold")
ax.add_patch(FancyArrowPatch((6.5, 7.05), (9.75, 6.42), arrowstyle="-|>", mutation_scale=16,
                             color=C_AGENT_E, lw=1.8, zorder=1,
                             connectionstyle="arc3,rad=-0.25"))

# ---- 4. Data layers (left side, feeding analyses) ----
box(0.4, 2.55, 3.0, 1.25,
    "DATA LAYERS\n"
    "• PR meta (state, dates,\n   agent)\n"
    "• File changes → churn,\n   tests, instruction files\n"
    "• Commit msgs → reverts",
    "#f3f1f7", "#7b5ea7", fs=8.2, align="left")

# ---- 5a. Matched-repo RQs ----
box(3.8, 2.55, 4.0, 1.55,
    "MATCHED  (agent vs human)\n"
    "RQ1a  rejection rate over time\n"
    "RQ1b  fix size (code churn)\n"
    "RQ4   reverts after merge\n"
    "RQ5   rejection by bug type\n"
    "RQ6   does adding a test help?",
    C_RQ_M, C_RQ_M_E, fs=9, align="left")
arrow(3.45, 4.25, 5.8, 4.10)

# ---- 5b. All-agent RQs ----
box(8.3, 2.55, 3.9, 1.55,
    "ALL-AGENT\n"
    "RQ2a  which agent is used\n"
    "RQ2b  does switching agents\n         rescue a rejected fix?\n"
    "RQ3   instruction-file\n         adoption over time",
    C_RQ_A, C_RQ_A_E, fs=9, align="left")
arrow(9.75, 5.55, 10.25, 4.10)

# data layers feed both RQ blocks
arrow(2.4, 2.55, 4.2, 2.2, color="#7b5ea7", ls="--", lw=1.3)
arrow(3.4, 3.1, 3.78, 3.2, color="#7b5ea7", ls="--", lw=1.3)

# ---- 6. Methods footer ----
box(0.4, 1.35, 11.8, 0.8,
    "METHODS:  monthly trend lines   ·   Spearman trend test   ·   χ² for test-inclusion effect   ·   "
    "churn-binned robustness re-check (RQ6)",
    "#eeeeee", "#888888", fs=9.5, weight="bold")
arrow(5.8, 2.55, 6.3, 2.15, lw=1.3)
arrow(10.25, 2.55, 8.0, 2.15, lw=1.3)

# ---- 7. Limits footer ----
ax.text(7, 0.78,
        "Conservative definitions:  rejected = closed & never merged   ·   rate over DECIDED PRs only (open dropped)",
        ha="center", va="center", fontsize=9, color="#1a1a1a")
ax.text(7, 0.42,
        "Stated limits:  issue-linking covers ~18% of PRs (RQ2b)   ·   reverts are a lower bound   ·   RQ5 bug types are keyword-derived",
        ha="center", va="center", fontsize=8.5, color=GREY, style="italic")

# legend
legend = [
    Line2D([0],[0], marker="s", color="w", markerfacecolor=C_MATCH, markeredgecolor=C_MATCH_E,
           markersize=13, label="Matched-repo path (fair agent vs human)"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor=C_AGENT, markeredgecolor=C_AGENT_E,
           markersize=13, label="All-agent path (agent-only questions)"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor="#f3f1f7", markeredgecolor="#7b5ea7",
           markersize=13, label="Shared data layers"),
]
ax.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.985, 0.985),
          frameon=True, fontsize=8.5, ncol=1)

plt.tight_layout()
out = r"C:\Users\Mahmoudabujadallah\final_agantic_bugfixes\methodology_figure.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
