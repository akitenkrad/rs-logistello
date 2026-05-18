#!/usr/bin/env python3
"""Generate the README hero banner (assets/banner.png).

Deterministic, dependency-light (matplotlib only — already a project dep).
Run from the repo root:

    uv run python scripts/make_banner.py

This is an eye-catching presentation asset, not a functional diagram.
Re-run to regenerate after tweaking the design constants below.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

# ---- palette -------------------------------------------------------------
FELT_TOP = np.array([0.039, 0.184, 0.137])  # deep Othello-felt green
FELT_BOT = np.array([0.012, 0.035, 0.047])  # near-black
GOLD = "#E8B23A"
WHITE = "#F4F4EF"
DISC_BLACK = "#0D0D0F"
DISC_WHITE = "#EFEFE6"
MUTED = "#8FB7A6"

W, H = 1600, 520  # px (dpi=100)


def _vertical_gradient(h: int, w: int, top: np.ndarray, bot: np.ndarray) -> np.ndarray:
    t = np.linspace(0.0, 1.0, h)[:, None, None]
    grad = top[None, None, :] * (1.0 - t) + bot[None, None, :] * t
    return np.repeat(grad, w, axis=1)


def _font(weight: str = "bold") -> fm.FontProperties:
    # DejaVu ships with matplotlib, so the banner renders identically anywhere.
    return fm.FontProperties(family="DejaVu Sans", weight=weight)


def main() -> None:
    out = pathlib.Path(__file__).resolve().parents[1] / "assets" / "banner.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # background felt gradient + soft vignette
    ax.imshow(
        _vertical_gradient(H, W, FELT_TOP, FELT_BOT),
        extent=(0, W, 0, H),
        origin="upper",
        aspect="auto",
        zorder=0,
    )
    yy, xx = np.mgrid[0:H, 0:W]
    vignette = 1.0 - 0.35 * (
        ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
    )
    ax.imshow(
        np.clip(vignette, 0, 1),
        extent=(0, W, 0, H),
        origin="upper",
        aspect="auto",
        cmap="Greys_r",
        alpha=0.18,
        zorder=1,
    )

    # faint 8x8 board grid on the right third (board motif)
    gx0, gy0, cell = 1066, 70, 47
    for i in range(9):
        ax.plot(
            [gx0 + i * cell, gx0 + i * cell],
            [gy0, gy0 + 8 * cell],
            color=MUTED,
            alpha=0.16,
            lw=1.0,
            zorder=2,
        )
        ax.plot(
            [gx0, gx0 + 8 * cell],
            [gy0 + i * cell, gy0 + i * cell],
            color=MUTED,
            alpha=0.16,
            lw=1.0,
            zorder=2,
        )

    # a dramatic diagonal of discs "flipping" black -> white across the grid
    rng = np.random.default_rng(1997)  # the Murakami-match year; deterministic
    r = cell * 0.40
    for k in range(8):
        cx = gx0 + (k + 0.5) * cell
        cy = gy0 + (7 - k + 0.5) * cell
        frac = k / 7.0  # 0 = black, 1 = white
        # soft drop shadow
        ax.add_patch(Circle((cx + 3, cy - 4), r, color="black", alpha=0.35, zorder=3))
        col = DISC_BLACK if frac < 0.45 else DISC_WHITE
        edge = "#2A2A2A" if frac < 0.45 else "#C9C9BD"
        ax.add_patch(
            Circle((cx, cy), r, facecolor=col, edgecolor=edge, lw=1.4, zorder=4)
        )
        if 0.35 <= frac <= 0.65:  # the in-flight "flip" highlight
            ax.add_patch(
                Circle((cx, cy), r, facecolor="none", edgecolor=GOLD, lw=2.0, zorder=5)
            )
    # scatter a few extra discs on the board for richness
    for (fx, fy, c) in [
        (1, 6, DISC_WHITE),
        (2, 5, DISC_BLACK),
        (5, 2, DISC_WHITE),
        (6, 1, DISC_BLACK),
        (3, 6, DISC_BLACK),
        (4, 1, DISC_WHITE),
    ]:
        cx = gx0 + (fx + 0.5) * cell
        cy = gy0 + (fy + 0.5) * cell
        ax.add_patch(Circle((cx + 2, cy - 3), r, color="black", alpha=0.3, zorder=3))
        ax.add_patch(
            Circle(
                (cx, cy),
                r,
                facecolor=c,
                edgecolor="#2A2A2A" if c == DISC_BLACK else "#C9C9BD",
                lw=1.2,
                zorder=4,
            )
        )

    # wordmark (baseline-anchored so the underline can sit clear of glyphs)
    ax.text(
        92,
        352,
        "LOGISTELLO",
        color=WHITE,
        fontsize=84,
        fontproperties=_font("bold"),
        va="baseline",
        zorder=7,
    )
    # gold underline, clearly below the wordmark baseline (no glyph overlap)
    ax.add_patch(
        FancyBboxPatch(
            (98, 322),
            512,
            8,
            boxstyle="round,pad=1,rounding_size=4",
            facecolor=GOLD,
            edgecolor="none",
            zorder=6,
        )
    )

    # tagline + the famous score as its own gold line (no overlay hack)
    ax.text(
        98,
        276,
        "The Othello program that beat the human world champion",
        color=MUTED,
        fontsize=22,
        fontproperties=_font("normal"),
        va="baseline",
        zorder=7,
    )
    ax.text(
        98,
        232,
        "6–0",
        color=GOLD,
        fontsize=30,
        fontproperties=_font("bold"),
        va="baseline",
        zorder=7,
    )
    ax.text(
        170,
        232,
        "— Takeshi Murakami vs. Logistello, NEC RI, August 1997",
        color=WHITE,
        fontsize=20,
        fontproperties=_font("normal"),
        va="baseline",
        zorder=7,
    )
    ax.text(
        98,
        150,
        "A faithful Rust + Python reproduction  ·  Buro, 1994–1999",
        color=WHITE,
        fontsize=20,
        fontproperties=_font("bold"),
        va="baseline",
        zorder=7,
    )
    ax.text(
        98,
        112,
        "learned pattern evaluation   ·   ProbCut / Multi-ProbCut   ·   "
        "GLEM   ·   self-play opening-book learning",
        color=MUTED,
        fontsize=15,
        fontproperties=_font("normal"),
        va="baseline",
        zorder=7,
    )

    fig.savefig(out, dpi=100, facecolor=tuple(FELT_BOT))
    plt.close(fig)
    size_kb = out.stat().st_size / 1024
    print(f"wrote {out} ({W}x{H}px, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
