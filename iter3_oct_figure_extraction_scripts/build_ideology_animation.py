"""
Build ideology_animation.gif: a 7-scene animated stickman-style diagram
illustrating the project's fundamental narrative arc.

Scenes:
  1. SETUP        — Dataset · Model · Attacker (the actors)
  2. ERM          — Uniform weights, flat gradient norms, attacker at chance
  3. DRO ACTIVATES — λ rises, minority samples upweighted
  4. MEMORIZATION — Minority pre-clip gradient norms spike (+18 to +30%)
  5. ATTACKER     — WB AUC jumps, concentrated in minority subgroups
  6. DP-SGD       — Noise + clipping annihilate magnitude channel (‖g‖ ≈ 27 vs C=1.0)
  7. VERDICT      — Relative DRO−ERM gap survives at ε=8; naive stacking is not enough
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D
from PIL import Image

OUT = "/Users/srimaan/Projects/privacy-audit/docs/figures/ideology_animation.gif"

W, H = 1100, 650
BG = "#0f1220"
INK = "#e8ecff"
ACCENT = "#ff7a59"      # orange - harm / leakage
GOOD   = "#6bd490"      # green  - fairness gain
BLUE   = "#5ab9ff"      # blue   - model / mechanism
YELLOW = "#ffd866"      # yellow - DP-SGD / noise
VIOLET = "#c299ff"      # violet - minority subgroup

SCENE_FRAMES = 6
FADE = 0.08             # per-frame fade-in alpha increment

def base_canvas(title):
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    ax.set_axis_off()
    # header
    ax.text(W/2, H - 30, title,
            ha="center", va="center", color=INK,
            fontsize=22, weight="bold", family="DejaVu Sans")
    # footer bar
    ax.add_patch(Rectangle((0, 0), W, 28, facecolor="#1a1f38", edgecolor="none"))
    ax.text(W/2, 14, "Privacy Audit · DRO × DP-SGD · Steinke One-Run Audit",
            ha="center", va="center", color="#7a84b0", fontsize=10, family="DejaVu Sans")
    return fig, ax

def box(ax, x, y, w, h, label, color=BLUE, alpha=1.0, text_color=INK, fontsize=12, bold=False):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=10",
                       linewidth=1.5, edgecolor=color,
                       facecolor=color, alpha=alpha*0.18)
    ax.add_patch(p)
    # border
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=10",
                       linewidth=2.0, edgecolor=color,
                       facecolor="none", alpha=alpha)
    ax.add_patch(b)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, label,
            ha="center", va="center", color=text_color,
            fontsize=fontsize, weight=weight, family="DejaVu Sans", alpha=alpha)

def stickman(ax, x, y, scale=1.0, color=INK, alpha=1.0, label=None, highlight=None):
    """Draw a stickman at (x,y). highlight can be a color to make the body that color."""
    s = scale
    head = Circle((x, y + 42*s), 10*s, fill=False, edgecolor=color, lw=2, alpha=alpha)
    ax.add_patch(head)
    # body
    body_col = highlight if highlight else color
    ax.plot([x, x], [y + 32*s, y + 10*s], color=body_col, lw=2.5, alpha=alpha, solid_capstyle="round")
    # arms
    ax.plot([x - 14*s, x + 14*s], [y + 26*s, y + 26*s], color=body_col, lw=2.5, alpha=alpha, solid_capstyle="round")
    # legs
    ax.plot([x, x - 12*s], [y + 10*s, y - 10*s], color=body_col, lw=2.5, alpha=alpha, solid_capstyle="round")
    ax.plot([x, x + 12*s], [y + 10*s, y - 10*s], color=body_col, lw=2.5, alpha=alpha, solid_capstyle="round")
    if label:
        ax.text(x, y - 22*s, label, ha="center", va="center",
                color=color, fontsize=10, family="DejaVu Sans", alpha=alpha)

def arrow(ax, x1, y1, x2, y2, color=INK, alpha=1.0, lw=2.0, style="->", text=None, text_color=INK):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=18,
                        color=color, lw=lw, alpha=alpha)
    ax.add_patch(a)
    if text:
        mx, my = (x1 + x2)/2, (y1 + y2)/2 + 10
        ax.text(mx, my, text, ha="center", va="bottom",
                color=text_color, fontsize=9, family="DejaVu Sans", alpha=alpha)

def bar(ax, x, y, w, h, label, color, alpha=1.0):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, alpha=alpha*0.85, edgecolor="none"))
    ax.text(x + w/2, y + h + 4, label, ha="center", va="bottom",
            color=INK, fontsize=9, family="DejaVu Sans", alpha=alpha)


# ---------- SCENE 1: SETUP ----------
def scene_setup(fade=1.0):
    fig, ax = base_canvas("①  SETUP — Three Actors")
    # Dataset
    box(ax, 80, 330, 200, 140, "Dataset\nAdult · Texas · UTKFace",
        color=BLUE, alpha=fade, fontsize=14, bold=True)
    # samples grid
    for i in range(3):
        for j in range(5):
            col = VIOLET if (i == 0 and j >= 3) else INK
            c = Circle((120 + j*22, 360 - i*22), 6, facecolor=col, alpha=fade*0.6)
            ax.add_patch(c)
    ax.text(180, 315, "majority     minority", ha="center", va="center",
            color=INK, fontsize=9, family="DejaVu Sans", alpha=fade)

    # Model
    box(ax, 450, 330, 200, 140, "Model θ\nERM  or  χ²-DRO",
        color=GOOD, alpha=fade, fontsize=14, bold=True)
    stickman(ax, 550, 380, scale=1.0, color=GOOD, alpha=fade)

    # Attacker
    box(ax, 820, 330, 200, 140, "Attacker\nMIA: BB + WB",
        color=ACCENT, alpha=fade, fontsize=14, bold=True)
    stickman(ax, 920, 380, scale=1.0, color=ACCENT, alpha=fade, highlight=ACCENT)

    # Question
    ax.text(W/2, 230, "What happens when we stack fairness on top of privacy?",
            ha="center", va="center", color=YELLOW,
            fontsize=16, weight="bold", family="DejaVu Sans", alpha=fade,
            bbox=dict(boxstyle="round,pad=0.5", edgecolor=YELLOW,
                      facecolor=YELLOW, alpha=fade*0.1))
    arrow(ax, 280, 400, 450, 400, color=INK, alpha=fade, text="train on canaries")
    arrow(ax, 650, 400, 820, 400, color=INK, alpha=fade, text="audit leakage")
    return fig


# ---------- SCENE 2: ERM ----------
def scene_erm(fade=1.0):
    fig, ax = base_canvas("②  ERM BASELINE — Uniform Weights · Attacker at Chance")
    # weights bar (uniform)
    ax.text(120, 550, "Sample weights (ERM)", color=INK, fontsize=12,
            family="DejaVu Sans", alpha=fade)
    for i in range(10):
        col = VIOLET if i >= 7 else BLUE
        bar(ax, 100 + i*26, 420, 22, 90, "", color=col, alpha=fade)

    # gradient norms (flat)
    ax.text(430, 550, "Pre-clip ‖g‖  (uniform)", color=INK, fontsize=12,
            family="DejaVu Sans", alpha=fade)
    for i in range(10):
        col = VIOLET if i >= 7 else BLUE
        bar(ax, 420 + i*26, 420, 22, 60 + 2*np.sin(i), "", color=col, alpha=fade)

    # attacker AUC
    ax.text(820, 550, "MIA AUC", color=INK, fontsize=12,
            family="DejaVu Sans", alpha=fade)
    ax.text(880, 470, "0.50", ha="center", va="center",
            color=GOOD, fontsize=44, weight="bold",
            family="DejaVu Sans", alpha=fade)
    ax.text(880, 420, "≈ chance", ha="center", va="center",
            color=GOOD, fontsize=14, family="DejaVu Sans", alpha=fade)

    # narrative
    ax.text(W/2, 230, "Equal treatment → no gradient signal to exploit.",
            ha="center", va="center", color=YELLOW,
            fontsize=15, weight="bold", family="DejaVu Sans", alpha=fade,
            bbox=dict(boxstyle="round,pad=0.5", edgecolor=YELLOW,
                      facecolor=YELLOW, alpha=fade*0.1))
    return fig


# ---------- SCENE 3: DRO activates ----------
def scene_dro(fade=1.0):
    fig, ax = base_canvas("③  χ²-DRO ACTIVATES — Minority Upweighting (λ ∈ [10%, 90%])")
    # weights - minority boosted
    ax.text(120, 550, "Sample weights (DRO)", color=INK, fontsize=12,
            family="DejaVu Sans", alpha=fade)
    heights = [70, 72, 68, 74, 70, 72, 74, 140, 150, 130]
    for i, h in enumerate(heights):
        col = VIOLET if i >= 7 else BLUE
        bar(ax, 100 + i*26, 420, 22, h, "", color=col, alpha=fade)
    # λ indicator
    box(ax, 90, 290, 240, 40, "λ/max = 24%  (Goldilocks)",
        color=YELLOW, alpha=fade, fontsize=11, bold=True)

    # gradient norms starting to rise
    ax.text(430, 550, "Pre-clip ‖g‖  (rising for minority)", color=INK,
            fontsize=12, family="DejaVu Sans", alpha=fade)
    gheights = [58, 60, 62, 59, 61, 60, 63, 85, 90, 92]
    for i, h in enumerate(gheights):
        col = VIOLET if i >= 7 else BLUE
        bar(ax, 420 + i*26, 420, 22, h, "", color=col, alpha=fade)

    # narrative
    ax.text(W/2, 230, "DRO optimises worst-case risk → minority samples pulled harder.",
            ha="center", va="center", color=YELLOW,
            fontsize=14, weight="bold", family="DejaVu Sans", alpha=fade,
            bbox=dict(boxstyle="round,pad=0.5", edgecolor=YELLOW,
                      facecolor=YELLOW, alpha=fade*0.1))
    return fig


# ---------- SCENE 4: MEMORIZATION ----------
def scene_mem(fade=1.0):
    fig, ax = base_canvas("④  MEMORIZATION — Minority Gradient Norms Spike")
    # gradient norms - big minority spike
    ax.text(120, 560, "Pre-clip ‖g‖ per subgroup (Adult, No-DP)", color=INK,
            fontsize=12, family="DejaVu Sans", alpha=fade)

    groups = [
        ("White\nn=860",     10.7, GOOD,  "+11%"),
        ("Black\nn=446",     12.9, VIOLET, "+21%"),
        ("Asian-Pac\nn=1039",12.3, VIOLET, "+15%"),
        ("Amer-Ind\nn=311",  13.8, VIOLET, "+29%"),
        ("Other\nn=271",     13.9, ACCENT, "+30%"),
    ]
    bw = 90
    for i, (name, h, col, delta) in enumerate(groups):
        x = 90 + i*140
        bar(ax, x, 320, bw, h*15, "", color=col, alpha=fade)
        ax.text(x + bw/2, 300, name, ha="center", va="top",
                color=INK, fontsize=9, family="DejaVu Sans", alpha=fade)
        ax.text(x + bw/2, 320 + h*15 + 8, delta, ha="center", va="bottom",
                color=col, fontsize=11, weight="bold",
                family="DejaVu Sans", alpha=fade)

    # narrative
    ax.text(W/2, 200, "Smaller subgroups → harder samples → bigger gradients.\n"
                       "The model is *memorising* exactly the people DRO was meant to help.",
            ha="center", va="center", color=ACCENT,
            fontsize=13, weight="bold", family="DejaVu Sans", alpha=fade,
            bbox=dict(boxstyle="round,pad=0.5", edgecolor=ACCENT,
                      facecolor=ACCENT, alpha=fade*0.12))
    return fig


# ---------- SCENE 5: ATTACKER wins ----------
def scene_attacker(fade=1.0):
    fig, ax = base_canvas("⑤  ATTACKER — Leakage Concentrated in Minorities")
    # subgroup AUC table
    rows = [
        ("White (majority)",  0.419, 0.476, "+0.057"),
        ("Black",             0.419, 0.488, "+0.069"),
        ("Asian-Pac-Islander",0.419, 0.491, "+0.072"),
        ("Amer-Indian-Eskimo",0.419, 0.530, "+0.111"),
        ("Other",             0.419, 0.565, "+0.146"),
    ]
    # header
    ax.text(140, 520, "Subgroup", color=INK, fontsize=11, weight="bold",
            family="DejaVu Sans", alpha=fade)
    ax.text(380, 520, "ERM WB AUC", color=INK, fontsize=11, weight="bold",
            family="DejaVu Sans", alpha=fade, ha="center")
    ax.text(540, 520, "DRO WB AUC", color=INK, fontsize=11, weight="bold",
            family="DejaVu Sans", alpha=fade, ha="center")
    ax.text(700, 520, "Δ (DRO−ERM)", color=INK, fontsize=11, weight="bold",
            family="DejaVu Sans", alpha=fade, ha="center")

    for i, (name, e, d, delta) in enumerate(rows):
        y = 480 - i*40
        ax.text(140, y, name, color=INK, fontsize=11,
                family="DejaVu Sans", alpha=fade, va="center")
        ax.text(380, y, f"{e:.3f}", color=GOOD, fontsize=12, weight="bold",
                family="DejaVu Sans Mono", alpha=fade, ha="center", va="center")
        ax.text(540, y, f"{d:.3f}", color=ACCENT, fontsize=12, weight="bold",
                family="DejaVu Sans Mono", alpha=fade, ha="center", va="center")
        # Δ grows with minority status
        intensity = min(1.0, 0.3 + i*0.175)
        ax.text(700, y, delta, color=ACCENT, fontsize=13, weight="bold",
                family="DejaVu Sans Mono", alpha=min(1.0, fade*intensity + fade*0.3),
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", edgecolor=ACCENT,
                          facecolor=ACCENT, alpha=min(1.0, fade*0.1*intensity)))

    # Pearson r
    box(ax, 780, 330, 240, 70, "Pearson r(subgroup n, ΔAUC)\n= −0.83  to  −0.91",
        color=ACCENT, alpha=fade, fontsize=12, bold=True)

    # narrative
    ax.text(W/2, 180,
            "The *smaller* the subgroup, the *bigger* the DRO-induced leakage.\n"
            "r(subgroup size, ΔAUC) ∈ [−0.91, −0.83] across all three datasets.",
            ha="center", va="center", color=ACCENT,
            fontsize=13, weight="bold", family="DejaVu Sans", alpha=fade,
            bbox=dict(boxstyle="round,pad=0.5", edgecolor=ACCENT,
                      facecolor=ACCENT, alpha=fade*0.12))
    return fig


# ---------- SCENE 6: DP-SGD ----------
def scene_dp(fade=1.0):
    fig, ax = base_canvas("⑥  DP-SGD — Noise + Clipping Annihilate the Magnitude Channel")
    # three channels as stacked boxes
    channels = [
        ("Magnitude  ‖g‖",
         "ANNIHILATED\n‖g‖ ≈ 27–34  >>  C=1.0\nclipping saturates every sample",
         ACCENT, True),
        ("Direction  ⟨Δw, g⟩",
         "PERSISTS  ← leakage lives here\nWB AUC survives clipping",
         YELLOW, False),
        ("Output  (loss drop)",
         "ATTENUATED\nBB AUC → 0.50 at ε ≤ 8",
         BLUE, True),
    ]
    for i, (name, desc, col, killed) in enumerate(channels):
        x = 60 + i*340
        box(ax, x, 360, 310, 180, name, color=col, alpha=fade,
            fontsize=14, bold=True)
        ax.text(x + 155, 400, desc, ha="center", va="top",
                color=INK, fontsize=10, family="DejaVu Sans", alpha=fade)
        if killed:
            # big red X
            ax.plot([x+30, x+280], [540, 380], color=ACCENT, lw=6, alpha=fade*0.9)
            ax.plot([x+30, x+280], [380, 540], color=ACCENT, lw=6, alpha=fade*0.9)

    # noise
    for _ in range(40):
        ax.add_patch(Circle((np.random.uniform(40, W-40), np.random.uniform(280, 340)),
                            2, facecolor=YELLOW, alpha=fade*0.3))

    ax.text(W/2, 230, "DP-SGD clips ‖g‖ to C=1.0 — magnitude channel is gone.\n"
                       "But the *direction* channel ⟨Δw, g⟩ survives.",
            ha="center", va="center", color=YELLOW,
            fontsize=14, weight="bold", family="DejaVu Sans", alpha=fade,
            bbox=dict(boxstyle="round,pad=0.5", edgecolor=YELLOW,
                      facecolor=YELLOW, alpha=fade*0.1))
    return fig


# ---------- SCENE 7: VERDICT ----------
def scene_verdict(fade=1.0):
    fig, ax = base_canvas("⑦  VERDICT — The Fairness–Privacy Trade-off")
    # ε-axis
    ax.plot([80, W-80], [400, 400], color=INK, lw=2, alpha=fade)
    eps_pts = [
        ("ε = ∞\n(no DP)",   130, 1.00, ACCENT),
        ("ε = 8",           360, 0.55, YELLOW),
        ("ε = 4",           620, 0.18, BLUE),
        ("ε = 2",           880, 0.05, GOOD),
    ]
    for name, x, v, col in eps_pts:
        ax.plot([x, x], [400, 400 + v*180], color=col, lw=8, alpha=fade,
                solid_capstyle="round")
        ax.text(x, 380, name, ha="center", va="top", color=INK,
                fontsize=10, family="DejaVu Sans", alpha=fade)
        ax.text(x, 400 + v*180 + 8,
                f"ΔWB = {v*0.058:+.3f}" if v > 0.08 else "masked",
                ha="center", va="bottom", color=col, fontsize=10, weight="bold",
                family="DejaVu Sans Mono", alpha=fade)
    ax.text(50, 400 + 0.5*180, "ΔWB\nAUC", ha="right", va="center",
            color=INK, fontsize=10, family="DejaVu Sans", alpha=fade)

    # verdict box
    box(ax, 200, 110, W-400, 140,
        "Naive stacking (DRO + DP-SGD) is NOT a principled solution.\n"
        "The relative DRO−ERM gap survives at ε = 8 (p = 4.3×10⁻⁷).\n"
        "We need jointly-designed fair+private algorithms.",
        color=ACCENT, alpha=fade, fontsize=13, bold=True)
    return fig


SCENES = [scene_setup, scene_erm, scene_dro, scene_mem,
          scene_attacker, scene_dp, scene_verdict]


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=BG, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((W, H), Image.LANCZOS)
    return img


def main():
    frames = []
    durations = []
    # build frames: for each scene, ramp alpha from FADE to 1.0 across SCENE_FRAMES
    for scene_fn in SCENES:
        for f in range(SCENE_FRAMES):
            alpha = min(1.0, FADE + (1.0 - FADE) * (f / (SCENE_FRAMES - 1)))
            fig = scene_fn(fade=alpha)
            frames.append(fig_to_pil(fig))
            durations.append(80)     # ms
        # hold last frame of scene longer
        durations[-1] = 900

    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=False)
    print(f"✅ wrote {OUT}  ({len(frames)} frames, ~{sum(durations)/1000:.1f}s loop)")


if __name__ == "__main__":
    main()
