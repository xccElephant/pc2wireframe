"""Render the (corrected) 2-stage pc2wireframe pipeline figure.

Run: python assets/make_pipeline.py  ->  assets/pipeline_v2.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# ----------------------------------------------------------------------
# palette
TRAIN = "#F2A35E"      # trainable (orange)
TRAIN_E = "#D9822B"
FROZEN = "#5B9BD5"     # frozen (blue)
FROZEN_E = "#2E6DA4"
DATA = "#E2E2E2"       # data / IO (gray)
DATA_E = "#9A9A9A"
BAND = "#F3F3F3"       # stage band
TXT_DARK = "#222222"
TXT_LIGHT = "#FFFFFF"

fig, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim(0, 15)
ax.set_ylim(0, 9)
ax.axis("off")


def box(cx, cy, w, h, text, fc, ec, tc=TXT_DARK, fs=12, weight="bold", r=0.12):
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={r}",
        linewidth=1.6, facecolor=fc, edgecolor=ec, zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight=weight, zorder=4, linespacing=1.15)
    return (cx, cy, w, h)


def band(y0, y1, label):
    ax.add_patch(Rectangle((0.15, y0), 14.7, y1 - y0, facecolor=BAND,
                           edgecolor="none", zorder=0))
    ax.text(0.45, y1 - 0.28, label, ha="left", va="top",
            fontsize=15, color="#333333", fontweight="bold", zorder=2)


def arrow(p1, p2, style="-|>", ls="-", color="#333333", lw=2.0, rad=0.0,
          shrink=4):
    a = FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=18, linewidth=lw,
        linestyle=ls, color=color, zorder=2,
        connectionstyle=f"arc3,rad={rad}", shrinkA=shrink, shrinkB=shrink)
    ax.add_patch(a)


def rside(b):  # right-middle
    return (b[0] + b[2] / 2, b[1])


def lside(b):
    return (b[0] - b[2] / 2, b[1])


def top(b):
    return (b[0], b[1] + b[3] / 2)


def bot(b):
    return (b[0], b[1] - b[3] / 2)


def hflow(a, b):
    arrow(rside(a), lside(b))


# ======================================================================
# Stage 1 : Curve VAE  (y ~ 7.0)
band(6.0, 8.4, "Stage 1  ·  Curve VAE")
y1 = 7.1
s1_in = box(1.6, y1, 1.3, 1.0, "curve\n(norm.)", DATA, DATA_E, fs=10.5)
s1_enc = box(4.0, y1, 2.1, 1.1, "Encoder", TRAIN, TRAIN_E, TXT_LIGHT, fs=13)
s1_lat = box(6.6, y1, 1.5, 1.0, "latent\n(12-d)", DATA, DATA_E, fs=10.5)
s1_dec = box(9.4, y1, 2.1, 1.1, "Decoder", TRAIN, TRAIN_E, TXT_LIGHT, fs=13)
s1_out = box(12.0, y1, 1.5, 1.0, "curve\n@ any t", DATA, DATA_E, fs=10.5)
for a, b in [(s1_in, s1_enc), (s1_enc, s1_lat), (s1_lat, s1_dec), (s1_dec, s1_out)]:
    hflow(a, b)
ax.text(13.6, y1, "trainable\n(both)", ha="center", va="center",
        fontsize=10, color=TRAIN_E, fontweight="bold")

# ======================================================================
# Stage 2 : PC2Wireframe (end-to-end)  (y ~ 4.0)
band(2.6, 5.7, "Stage 2  ·  PC2Wireframe   (end-to-end; Curve VAE frozen)")
y2 = 4.05
s2_in = box(1.45, y2, 1.35, 1.1, "point\ncloud", DATA, DATA_E, fs=10)
s2_enc = box(4.0, y2, 2.5, 1.2, "PTv3 Encoder\n+ Compressor", TRAIN, TRAIN_E, TXT_LIGHT, fs=11)
s2_lat = box(6.55, y2, 1.3, 1.0, "latent\n16x256", DATA, DATA_E, fs=10)
s2_dec = box(9.3, y2, 2.6, 1.3, "Transformer Decoder\nnode + edge queries", TRAIN, TRAIN_E, TXT_LIGHT, fs=10.5)
s2_cd = box(12.2, y2, 1.55, 1.05, "Curve VAE\nDecoder", FROZEN, FROZEN_E, TXT_LIGHT, fs=10.5)
s2_out = box(13.95, y2, 0.95, 1.0, "wire-\nframe", DATA, DATA_E, fs=9)
for a, b in [(s2_in, s2_enc), (s2_enc, s2_lat), (s2_lat, s2_dec),
             (s2_dec, s2_cd), (s2_cd, s2_out)]:
    hflow(a, b)
ax.text(9.3, y2 - 1.02,
        "node queries -> coord + existence\n"
        "edge queries -> existence + endpoint dist. + curve latent",
        ha="center", va="top", fontsize=8.3, color="#555555", style="italic")

# frozen-reuse arrow from Stage 1 decoder -> Stage 2 curve decoder
arrow(bot(s1_dec), top(s2_cd), style="-|>", ls=(0, (4, 3)),
      color=FROZEN_E, lw=1.8, rad=0.12)
ax.text(11.0, 6.0, "frozen Curve VAE\nDecoder (Stage 1)", ha="center", va="center",
        fontsize=8.5, color=FROZEN_E, fontweight="bold")

# training supervision annotation
ax.text(3.0, y2 - 1.02,
        "train: Hungarian node/edge matching;\n"
        "Focal BCE existence + L1 coords/curves",
        ha="center", va="top", fontsize=8.3, color="#555555", style="italic")

# ======================================================================
# Inference  (y ~ 1.0)
ax.text(7.5, 1.78, "Inference", ha="center", va="center",
        fontsize=14, color="#333333", fontweight="bold")
yi = 0.95
i_in = box(1.4, yi, 1.1, 0.85, "point\ncloud", DATA, DATA_E, fs=9)
i_enc = box(3.7, yi, 2.3, 0.95, "PTv3 Encoder\n+ Compressor", FROZEN, FROZEN_E, TXT_LIGHT, fs=9.5)
i_lat = box(6.0, yi, 1.2, 0.8, "latent", DATA, DATA_E, fs=9.5)
i_wf = box(8.5, yi, 2.4, 0.95, "Wireframe\nDecoder", FROZEN, FROZEN_E, TXT_LIGHT, fs=10)
i_cv = box(11.1, yi, 1.8, 0.95, "Curve VAE\nDecoder", FROZEN, FROZEN_E, TXT_LIGHT, fs=10)
i_out = box(13.1, yi, 1.2, 0.85, "wire-\nframe", DATA, DATA_E, fs=9.5)
for a, b in [(i_in, i_enc), (i_enc, i_lat), (i_lat, i_wf), (i_wf, i_cv), (i_cv, i_out)]:
    hflow(a, b)
ax.text(14.2, yi, "all\nfrozen", ha="center", va="center",
        fontsize=9, color=FROZEN_E, fontweight="bold")

# ======================================================================
# legend
lx, ly = 9.2, 8.05
ax.add_patch(FancyBboxPatch((lx, ly - 0.18), 0.45, 0.36,
             boxstyle="round,pad=0.01,rounding_size=0.06",
             facecolor=TRAIN, edgecolor=TRAIN_E, zorder=4))
ax.text(lx + 0.6, ly, "trainable", va="center", fontsize=11, fontweight="bold")
ax.add_patch(FancyBboxPatch((lx + 2.2, ly - 0.18), 0.45, 0.36,
             boxstyle="round,pad=0.01,rounding_size=0.06",
             facecolor=FROZEN, edgecolor=FROZEN_E, zorder=4))
ax.text(lx + 2.8, ly, "frozen", va="center", fontsize=11, fontweight="bold")
arrow((lx + 4.2, ly), (lx + 4.9, ly), ls=(0, (4, 3)), color=FROZEN_E, lw=1.8)
ax.text(lx + 5.0, ly, "reuse frozen\nweights", va="center", fontsize=9,
        color=FROZEN_E, fontweight="bold")

plt.tight_layout()
out = __file__.rsplit("/", 1)[0] + "/pipeline_v2.png"
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print("saved", out)
