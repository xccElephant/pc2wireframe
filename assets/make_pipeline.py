"""Render the (corrected) 3-stage pc2wireframe pipeline figure.

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

fig, ax = plt.subplots(figsize=(15, 11))
ax.set_xlim(0, 15)
ax.set_ylim(0, 11)
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
# Stage 1 : Curve VAE  (y ~ 9.0)
band(8.0, 10.4, "Stage 1  ·  Curve VAE")
y1 = 9.1
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
# Stage 2 : Wireframe VAE  (y ~ 6.0)
band(4.6, 7.7, "Stage 2  ·  Wireframe VAE   (Curve VAE frozen)")
y2 = 5.95
s2_in = box(1.5, y2, 1.4, 1.1, "wireframe\ngraph", DATA, DATA_E, fs=10)
s2_ce = box(3.7, y2, 1.6, 1.05, "Curve VAE\nEncoder", FROZEN, FROZEN_E, TXT_LIGHT, fs=10.5)
s2_enc = box(6.0, y2, 1.9, 1.15, "Graph\nEncoder", TRAIN, TRAIN_E, TXT_LIGHT, fs=12)
s2_lat = box(8.0, y2, 1.3, 1.0, "latent\n64x64", DATA, DATA_E, fs=10)
s2_dec = box(10.1, y2, 1.9, 1.15, "Graph\nDecoder", TRAIN, TRAIN_E, TXT_LIGHT, fs=12)
s2_cd = box(12.4, y2, 1.6, 1.05, "Curve VAE\nDecoder", FROZEN, FROZEN_E, TXT_LIGHT, fs=10.5)
s2_out = box(14.0, y2, 0.9, 1.0, "wire-\nframe", DATA, DATA_E, fs=9)
for a, b in [(s2_in, s2_ce), (s2_ce, s2_enc), (s2_enc, s2_lat),
             (s2_lat, s2_dec), (s2_dec, s2_cd), (s2_cd, s2_out)]:
    hflow(a, b)
ax.text(10.1, y2 - 0.95, "nodes + adjacency\n+ per-edge curve latent",
        ha="center", va="top", fontsize=8.5, color="#555555", style="italic")

# frozen-reuse arrows from Stage 1 -> Stage 2 (to the CORRECT blocks)
arrow(bot(s1_enc), top(s2_ce), style="-|>", ls=(0, (4, 3)),
      color=FROZEN_E, lw=1.8, rad=-0.15)
arrow(bot(s1_dec), top(s2_cd), style="-|>", ls=(0, (4, 3)),
      color=FROZEN_E, lw=1.8, rad=0.15)
ax.text(2.45, 7.0, "frozen weights\nfrom Stage 1", ha="center", va="center",
        fontsize=8.5, color=FROZEN_E, fontweight="bold")
ax.text(12.95, 7.45, "frozen weights\nfrom Stage 1", ha="center", va="center",
        fontsize=8.5, color=FROZEN_E, fontweight="bold")

# ======================================================================
# Stage 3 : PC2Wireframe  (y ~ 3.0)
band(1.9, 4.3, "Stage 3  ·  PC2Wireframe   (both VAEs frozen)")
y3 = 2.95
s3_in = box(1.5, y3, 1.4, 1.1, "point\ncloud", DATA, DATA_E, fs=10)
s3_enc = box(4.2, y3, 2.4, 1.2, "PTv3 Encoder\n+ Compressor", TRAIN, TRAIN_E, TXT_LIGHT, fs=11)
s3_lat = box(6.9, y3, 1.3, 1.0, "latent\n64x64", DATA, DATA_E, fs=10)
s3_dec = box(9.3, y3, 1.9, 1.15, "Graph\nDecoder", FROZEN, FROZEN_E, TXT_LIGHT, fs=12)
s3_cd = box(11.7, y3, 1.6, 1.05, "Curve VAE\nDecoder", FROZEN, FROZEN_E, TXT_LIGHT, fs=10.5)
s3_out = box(13.6, y3, 1.1, 1.0, "wire-\nframe", DATA, DATA_E, fs=9.5)
for a, b in [(s3_in, s3_enc), (s3_enc, s3_lat), (s3_lat, s3_dec),
             (s3_dec, s3_cd), (s3_cd, s3_out)]:
    hflow(a, b)

# frozen reuse from stage 2 blocks
arrow(bot(s2_dec), top(s3_dec), style="-|>", ls=(0, (4, 3)),
      color=FROZEN_E, lw=1.8, rad=0.12)
arrow(bot(s2_cd), top(s3_cd), style="-|>", ls=(0, (4, 3)),
      color=FROZEN_E, lw=1.8, rad=-0.1)
ax.text(8.5, 4.05, "frozen Graph Decoder (Stage 2)",
        ha="center", va="center", fontsize=8.5, color=FROZEN_E, fontweight="bold")

# teacher-posterior alignment annotation
ax.text(6.9, y3 - 0.85,
        "train: align predicted latent to Stage-2 teacher posterior\n"
        "+ decode-through supervision",
        ha="center", va="top", fontsize=8.5, color="#555555", style="italic")

# ======================================================================
# Inference  (y ~ 0.9)
ax.text(7.5, 1.62, "Inference", ha="center", va="center",
        fontsize=14, color="#333333", fontweight="bold")
yi = 0.85
i_in = box(1.4, yi, 1.1, 0.85, "point\ncloud", DATA, DATA_E, fs=9)
i_enc = box(3.6, yi, 2.1, 0.95, "PTv3 Encoder", FROZEN, FROZEN_E, TXT_LIGHT, fs=10.5)
i_lat = box(5.9, yi, 1.2, 0.8, "latent", DATA, DATA_E, fs=9.5)
i_wf = box(8.2, yi, 2.0, 0.95, "Wireframe VAE\nDecoder", FROZEN, FROZEN_E, TXT_LIGHT, fs=10)
i_cv = box(10.6, yi, 1.8, 0.95, "Curve VAE\nDecoder", FROZEN, FROZEN_E, TXT_LIGHT, fs=10)
i_out = box(12.7, yi, 1.2, 0.85, "wire-\nframe", DATA, DATA_E, fs=9.5)
for a, b in [(i_in, i_enc), (i_enc, i_lat), (i_lat, i_wf), (i_wf, i_cv), (i_cv, i_out)]:
    hflow(a, b)
ax.text(13.9, yi, "all\nfrozen", ha="center", va="center",
        fontsize=9, color=FROZEN_E, fontweight="bold")

# ======================================================================
# legend
lx, ly = 9.2, 10.05
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
