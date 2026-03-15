"""Protocol comparison diagram: SEQUENTIAL vs HYBRID vs SIMULTANEOUS.

Redesign: context-box approach. Each turn has a compact listing of what the
speaker sees. B's extra edges (the asymmetric ones A doesn't get) are
highlighted. No arrows — the listing IS the information.

Key message at a glance:
  SEQUENTIAL: B sees 2 extra items (A's proposal, A's critique)
  HYBRID:     B sees 1 extra item  (A's critique)
  SIMULTANEOUS: symmetric — no extras
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Colors (colorblind-safe)
SPEAKER_A = "#0072B2"
SPEAKER_B = "#E69F00"
ASYM_COLOR = "#D55E00"  # highlight for asymmetric context
GRAY = "#404040"
LIGHT_GRAY = "#888888"

turns = [
    ("A proposes", SPEAKER_A),
    ("B proposes", SPEAKER_B),
    ("A critiques", SPEAKER_A),
    ("B critiques", SPEAKER_B),
]

# What each turn sees in each protocol.
# Each entry: list of (label, is_asymmetric)
# is_asymmetric = True means B sees this but A doesn't get the equivalent.
protocols = [
    {
        "name": "SEQUENTIAL",
        "subtitle": "Both rounds sequential",
        "context": [
            # A proposes: question only (shown separately)
            [],
            # B proposes: sees A's proposal (extra)
            [("A's proposal", True)],
            # A critiques: sees both proposals
            [("both proposals", False)],
            # B critiques: sees both proposals + A's critique
            [("both proposals", False), ("A's critique", True)],
        ],
        "simul_bands": [],
        "blind_turns": [],  # which turns are blind
        "extra_count": 2,
    },
    {
        "name": "HYBRID",
        "subtitle": "Blind proposals, sequential critiques",
        "context": [
            [],
            [],  # B is blind
            [("both proposals", False)],
            [("both proposals", False), ("A's critique", True)],
        ],
        "simul_bands": [(0, 1)],
        "blind_turns": [1],
        "extra_count": 1,
    },
    {
        "name": "SIMULTANEOUS",
        "subtitle": "Both phases simultaneous",
        "context": [
            [],
            [],  # B is blind
            [("both proposals", False)],
            [("both proposals", False)],  # symmetric — no A's critique
        ],
        "simul_bands": [(0, 1), (2, 3)],
        "blind_turns": [1],  # only B proposes is blind; B critiques sees completed round
        "extra_count": 0,
    },
]

# Layout
y_positions = [5.0, 3.9, 2.0, 0.9]
box_w, box_h = 2.2, 0.55
ctx_x = box_w / 2 + 0.25  # context text left edge

fig, axes = plt.subplots(1, 3, figsize=(18, 9))
fig.subplots_adjust(wspace=0.04, top=0.84, bottom=0.10, left=0.02, right=0.98)


def draw_turn_box(ax, x, y, label, color):
    box = FancyBboxPatch(
        (x - box_w / 2, y - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.08", facecolor=color + "25",
        edgecolor=color, linewidth=2.0, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=12, fontweight="bold", color=color, zorder=4)


def draw_context(ax, x, y, items, is_blind=False):
    """Draw context items to the right of a turn box."""
    if is_blind:
        # Big X + "blind"
        ax.text(x + 0.15, y, "\u2716", fontsize=22, color=ASYM_COLOR,
                ha="center", va="center", fontweight="bold")
        ax.text(x + 0.55, y, "blind", fontsize=11, color=ASYM_COLOR,
                ha="left", va="center", style="italic", fontweight="bold")
        return

    if not items:
        return

    # "sees:" header
    ax.text(x, y + 0.15, "sees:", fontsize=8.5, color=LIGHT_GRAY,
            ha="left", va="bottom", fontweight="bold")

    for i, (label, is_asym) in enumerate(items):
        color = ASYM_COLOR if is_asym else GRAY
        weight = "bold" if is_asym else "normal"
        marker = "\u25B6 " if is_asym else "  "
        ax.text(x + 0.05, y - 0.20 * i - 0.08, marker + label,
                fontsize=10, color=color, ha="left", va="top",
                fontweight=weight)


def draw_simul_bracket(ax, y_top, y_bot, x_left):
    """Bracket + 'simul.' for simultaneous turns."""
    bx = x_left - 0.2
    ax.plot([bx, bx], [y_bot, y_top],
            color=LIGHT_GRAY, linewidth=1.8, solid_capstyle="round", zorder=1)
    tick = 0.1
    ax.plot([bx, bx + tick], [y_top, y_top],
            color=LIGHT_GRAY, linewidth=1.8, solid_capstyle="round", zorder=1)
    ax.plot([bx, bx + tick], [y_bot, y_bot],
            color=LIGHT_GRAY, linewidth=1.8, solid_capstyle="round", zorder=1)
    mid_y = (y_top + y_bot) / 2
    ax.text(bx - 0.1, mid_y, "simul.",
            fontsize=7, color=LIGHT_GRAY, ha="right", va="center",
            style="italic", rotation=90)


for col_idx, (ax, proto) in enumerate(zip(axes, protocols)):
    ax.set_xlim(-3.2, 4.2)
    ax.set_ylim(-0.6, 6.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(0.3, 6.2, proto["name"], ha="center", va="center",
            fontsize=18, fontweight="bold", color=GRAY)
    ax.text(0.3, 5.7, proto["subtitle"], ha="center", va="top",
            fontsize=11, color=GRAY, linespacing=1.3)

    # Phase separator
    sep_y = 2.95
    ax.plot([-box_w / 2 - 0.6, ctx_x + 2.5], [sep_y, sep_y],
            color=GRAY, alpha=0.12, linewidth=0.8, zorder=0)
    ax.text(ctx_x + 2.5, sep_y + 0.15, "propose", fontsize=6,
            color=LIGHT_GRAY, ha="right", style="italic")
    ax.text(ctx_x + 2.5, sep_y - 0.22, "critique", fontsize=6,
            color=LIGHT_GRAY, ha="right", style="italic")

    # Simultaneous brackets
    for bs, be in proto["simul_bands"]:
        draw_simul_bracket(ax, y_positions[bs], y_positions[be], -box_w / 2)

    # "sees: question" for first turn (same across all protocols)
    ax.text(ctx_x, y_positions[0], "sees: question",
            fontsize=10, color=LIGHT_GRAY, ha="left", va="center")

    # Flow arrows between turns
    for i in range(len(y_positions) - 1):
        y_from = y_positions[i] - box_h / 2 - 0.04
        y_to = y_positions[i + 1] + box_h / 2 + 0.04
        if y_to < y_from:
            ax.annotate("", xy=(0, y_to), xytext=(0, y_from),
                        arrowprops=dict(
                            arrowstyle="->,head_width=0.1,head_length=0.07",
                            color=LIGHT_GRAY + "77", lw=0.7), zorder=1)

    # Turn boxes + context
    for i, (label, color) in enumerate(turns):
        draw_turn_box(ax, 0, y_positions[i], label, color)
        is_blind = i in proto["blind_turns"]
        draw_context(ax, ctx_x, y_positions[i], proto["context"][i], is_blind)

    # Score badge
    badge_y = -0.1
    count = proto["extra_count"]
    if count > 0:
        badge_color = ASYM_COLOR
        badge_text = f"+{count} extra edge{'s' if count > 1 else ''} for B"
    else:
        badge_text = "symmetric"
        badge_color = GRAY
    bw, bh = 2.4, 0.40
    ax.add_patch(FancyBboxPatch(
        (-bw / 2 + 0.3, badge_y - bh / 2), bw, bh,
        boxstyle="round,pad=0.06",
        facecolor=badge_color + "12",
        edgecolor=badge_color + "44",
        linewidth=1.0, zorder=2,
    ))
    ax.text(0.3, badge_y, badge_text, ha="center", va="center",
            fontsize=12, fontweight="bold", color=badge_color, zorder=3)


# Title
fig.text(0.5, 0.94, "Protocol Comparison: Information Asymmetry",
         ha="center", fontsize=18, fontweight="bold", color=GRAY)

# Legend
fig.text(0.5, 0.03,
         "\u25B6  = B sees this, but A doesn't get the equivalent (asymmetric edge)"
         "        \u2716 = hidden (simultaneous phase, no cross-visibility)",
         ha="center", fontsize=11, color=GRAY)

plt.savefig("reports/figures/fig_protocol_comparison.png", dpi=600, bbox_inches="tight",
            facecolor="white")
print("Saved fig_protocol_comparison.png")
