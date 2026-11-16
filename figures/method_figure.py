"""PhysICL yöntem şekli (IEEE çift sütun, 7.16 in). Gerçek simülatör çıktılarıyla çizilir.

Kullanım: PYTHONPATH=<proje kökü> python method_figure.py <çıktı klasörü>
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from scipy.signal import butter, hilbert, sosfiltfilt  # noqa: E402

from pilot2.simulator import FS, impulse_response, render, sample_machine, sample_unit  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
INK, MUTED, EDGE, ACC = "#1f1f1e", "#6b6a64", "#55544e", "#2a78d6"
CLS = {"N": "#76756f", "OR": "#2a78d6", "IR": "#eb6834", "B": "#1baf7a"}
W_IN, H_IN = 7.16, 4.25


def fig_box(fig, x, y, w, h, fc="white", ec=EDGE, lw=0.7, r=0.006, z=1, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", transform=fig.transFigure,
                       fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha)
    fig.patches.append(p)
    return p


def fig_arrow(fig, p, q, style="-|>", rad=0.0, color=EDGE, lw=0.7, ls="-"):
    a = FancyArrowPatch(p, q, transform=fig.transFigure, arrowstyle=style, mutation_scale=6.5, lw=lw,
                        color=color, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}", linestyle=ls,
                        zorder=6)
    fig.patches.append(a)


def txt(fig, x, y, s, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center"); kw.setdefault("color", INK)
    fig.text(x, y, s, **kw)


def thumb(fig, x, y, w, h, sig, color, lw=0.45, frame=True):
    ax = fig.add_axes([x, y, w, h])
    ax.plot(sig, color=color, lw=lw)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, len(sig) - 1)
    m = np.max(np.abs(sig)) * 1.1 + 1e-9
    ax.set_ylim(-m, m) if sig.min() < 0 else ax.set_ylim(sig.min() - 0.05 * m, m)
    for s in ax.spines.values():
        s.set_visible(frame); s.set_linewidth(0.4); s.set_color("#b9b8b1")
    ax.patch.set_alpha(0)
    return ax


def envelope(x):
    sos = butter(4, 500, btype="highpass", fs=FS, output="sos")
    e = np.abs(hilbert(sosfiltfilt(sos, x)))
    return (e - e.mean()) / (e.std() + 1e-9)


def main(out):
    rng = np.random.default_rng(11)
    m = sample_machine(rng)
    while m.fc < 5000 or not (20 < m.conditions[0][0] < 40) or len(m.conditions) < 2:
        m = sample_machine(rng)
    units = {}
    for c in ("N", "OR", "IR", "B"):
        u = sample_unit(m, c, rng)
        if c != "N":
            u.snr_db = 7.0
        units[c] = u
    L = 512

    def win(c, cond=0, u=None, n=L, seed=0):
        r = np.random.default_rng(seed)
        x = render(m, u or units[c], cond, 4096, r)[:n]
        return (x - x.mean()) / x.std()

    fig = plt.figure(figsize=(W_IN, H_IN))

    # ------------------------------------------------------------------ (a) physics prior
    txt(fig, 0.005, 0.985, "(a) Randomized physics prior: a new virtual machine per training episode",
        ha="left", fontsize=8, fontweight="bold")
    # nested hierarchy
    fig_box(fig, 0.005, 0.6, 0.285, 0.36, fc="#f6f5f1")
    txt(fig, 0.015, 0.94, r"Machine  $\theta_M$", ha="left", fontweight="bold")
    txt(fig, 0.015, 0.905, "bearing geometry (n, d/D, φ) → BPFO, BPFI, BSF", ha="left", fontsize=6.6,
        color=MUTED)
    txt(fig, 0.015, 0.877, "1–4 structural modes, sensor band, tones", ha="left", fontsize=6.6, color=MUTED)
    fig_box(fig, 0.02, 0.62, 0.255, 0.235, fc="#ecebe6")
    txt(fig, 0.03, 0.835, r"Part  $\theta_U$  (per class, per part)", ha="left", fontweight="bold")
    txt(fig, 0.03, 0.80, "defect SNR, impact scatter, slip, spall", ha="left", fontsize=6.6, color=MUTED)
    txt(fig, 0.03, 0.772, "mode shift ±7%, load zone, roughness", ha="left", fontsize=6.6, color=MUTED)
    fig_box(fig, 0.035, 0.638, 0.225, 0.107, fc="white", ec=ACC, lw=0.9)
    txt(fig, 0.045, 0.722, r"Condition  $\theta_C$  (per recording)", ha="left", fontweight="bold")
    txt(fig, 0.045, 0.688, "speed ×0.6–1.5 or load change", ha="left", fontsize=6.6, color=MUTED)
    txt(fig, 0.045, 0.66, "slow speed fluctuation ≤ 1%", ha="left", fontsize=6.6, color=MUTED)
    txt(fig, 0.0, 0.565, "8 health states: N, OR, IR, B, OR+IR, imbalance, misalignment, looseness", ha="left",
        fontsize=6.3, color=MUTED, style="italic")

    # synthesis chain with real signals (outer race)
    u = units["OR"]
    fr = m.conditions[0][0]
    n = 2400
    t = np.arange(n) / FS
    period = 1 / (m.orders()["BPFO"] * fr)
    imp = np.zeros(n)
    k = np.arange(0, n / FS, period)
    idx = (k * FS + rng.normal(0, u.slip * period * FS, len(k))).astype(int).clip(0, n - 1)
    imp[idx] = np.exp(rng.normal(0, u.amp_sigma, len(idx)))
    h = impulse_response(u.modes)[:220]
    x_or = render(m, u, 0, n, np.random.default_rng(3))
    from scipy.signal import fftconvolve
    fault = fftconvolve(imp, impulse_response(u.modes))[:n]
    bg = render(m, units["N"], 0, n, np.random.default_rng(4))
    cy, ch, cw = 0.66, 0.15, 0.105
    xs = [0.315, 0.455, 0.595, 0.735, 0.875]
    titles = ["Impacts at fault order", "Part impulse response", "Background", "Band-limited sum",
              "Training window"]
    subs = ["BPFO / BPFI / BSF, load zone", "damped structural modes", "shaft, gear, line, noise",
            r"sensor band $f_c$ ≤ 5.5 kHz", "z-scored, 1024 samples"]
    ax = fig.add_axes([xs[0], cy, cw, ch])
    ax.vlines(np.flatnonzero(imp), 0, imp[imp > 0], color=CLS["OR"], lw=0.7)
    ax.set_xlim(0, n); ax.set_ylim(0, imp.max() * 1.15)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color("#b9b8b1")
    thumb(fig, xs[1], cy, cw, ch, h / np.abs(h).max(), CLS["OR"])
    thumb(fig, xs[2], cy, cw, ch, bg[:n] / bg.std(), CLS["N"])
    thumb(fig, xs[3], cy, cw, ch, x_or / x_or.std(), INK)
    thumb(fig, xs[4], cy, cw, ch, win("OR", n=1024), CLS["OR"])
    for x0, tt, ss in zip(xs, titles, subs):
        txt(fig, x0 + cw / 2, cy + ch + 0.075, tt, fontsize=7, fontweight="bold")
        txt(fig, x0 + cw / 2, cy + ch + 0.04, ss, fontsize=6.3, color=MUTED)
    ops = [r"$\ast$", "+", "\u2192", "\u2192"]
    for i, op in enumerate(ops):
        txt(fig, xs[i] + cw + (xs[i + 1] - xs[i] - cw) / 2, cy + ch / 2, op, fontsize=11, color=EDGE)
    txt(fig, 0.315 + (0.98 - 0.315) / 2, cy - 0.04,
        "Outer-race example from one virtual machine (simulator output); ranges are redrawn per machine, part, condition",
        fontsize=6.3, color=MUTED, style="italic")
    fig_arrow(fig, (0.29, 0.73), (0.312, 0.73))

    # divider
    fig.add_artist(plt.Line2D([0.005, 0.995], [0.55, 0.55], transform=fig.transFigure, color="#d6d5cf", lw=0.6))

    # ------------------------------------------------------------------ (b) episodic fine-tuning
    txt(fig, 0.005, 0.525, "(b) Episodic fine-tuning on simulated episodes only", ha="left", fontsize=8,
        fontweight="bold")
    classes = ["N", "OR", "IR"]
    tw, th = 0.075, 0.07
    sx, qx = 0.035, 0.235
    ytop = 0.405
    txt(fig, sx + tw + 0.004, 0.483, "Support: part 1, condition A", fontsize=6.8)
    txt(fig, qx + tw + 0.004, 0.483, "Queries: other part / condition*", fontsize=6.8)
    other = {c: sample_unit(m, c, np.random.default_rng(40 + i)) for i, c in enumerate(classes)}
    for c in classes:
        if c != "N":
            other[c].snr_db = 2.0
    for i, c in enumerate(classes):
        y0 = ytop - i * (th + 0.03)
        txt(fig, 0.02, y0 + th / 2, c, fontsize=7, fontweight="bold", color=CLS[c])
        for j in range(2):
            thumb(fig, sx + j * (tw + 0.008), y0, tw, th, win(c, 0, seed=10 * i + j, n=384), CLS[c])
        for j in range(2):
            thumb(fig, qx + j * (tw + 0.008), y0, tw, th, win(c, 1, u=other[c], seed=100 + 10 * i + j, n=384),
                  CLS[c] if j == 0 else "#9b9a93")
        txt(fig, qx + 2 * (tw + 0.008) + 0.01, y0 + th / 2, "?", fontsize=9, color=MUTED)
    fig_box(fig, 0.008, 0.155, 0.42, 0.345, fc="none", ec="#c9c8c1", lw=0.6)
    txt(fig, 0.218, 0.175, "2\u20138 classes, 1\u201310 shots, labels permuted; *p = 0.7 / 0.8 (shown: both differ)", fontsize=6.3, color=MUTED, style="italic")
    # model
    bx, by, bw, bh = 0.445, 0.2, 0.105, 0.2
    fig_box(fig, bx, by, bw, bh, fc="#eef4fc", ec=ACC, lw=1.0)
    txt(fig, bx + bw / 2, by + bh - 0.035, "TimEE", fontweight="bold", fontsize=8)
    txt(fig, bx + bw / 2, by + bh - 0.075, "in-context", fontsize=6.6)
    txt(fig, bx + bw / 2, by + bh - 0.105, "transformer", fontsize=6.6)
    for r_ in range(3):
        fig_box(fig, bx + 0.015, by + 0.02 + r_ * 0.022, bw - 0.03, 0.014, fc="white", ec=ACC, lw=0.5)
    fig_arrow(fig, (0.428, 0.3), (bx, 0.3))

    fig_box(fig, 0.47, 0.06, 0.095, 0.075, fc="white")
    txt(fig, 0.5175, 0.107, "Cross-entropy", fontsize=6.8, fontweight="bold")
    txt(fig, 0.5175, 0.08, "on query logits", fontsize=6.3, color=MUTED)
    fig_arrow(fig, (bx + bw / 2, by), (0.5175, 0.135))
    fig_arrow(fig, (0.47, 0.098), (bx + 0.012, by), rad=-0.45, ls="--")
    txt(fig, 0.455, 0.075, "backprop,\n2000 steps", fontsize=6.2, color=MUTED, ha="right")


    fig.add_artist(plt.Line2D([0.6, 0.6], [0.04, 0.5], transform=fig.transFigure, color="#d6d5cf", lw=0.6))

    # ------------------------------------------------------------------ (c) inference
    txt(fig, 0.615, 0.525, "(c) Diagnosis on a new machine (no weight update)", ha="left", fontsize=8,
        fontweight="bold")
    real = win("IR", 0, u=sample_unit(m, "IR", np.random.default_rng(77)), seed=5, n=1024)
    thumb(fig, 0.62, 0.3, 0.085, 0.08, real, INK)
    txt(fig, 0.6625, 0.42, "k labeled windows", fontsize=6.6)
    txt(fig, 0.6625, 0.395, "+ queries", fontsize=6.6)
    thumb(fig, 0.735, 0.37, 0.085, 0.07, real, INK)
    thumb(fig, 0.735, 0.205, 0.085, 0.07, envelope(real), ACC)
    txt(fig, 0.7775, 0.455, "raw view", fontsize=6.6)
    txt(fig, 0.7775, 0.29, "envelope view", fontsize=6.6)
    txt(fig, 0.7775, 0.185, ">500 Hz, Hilbert", fontsize=6.0, color=MUTED)
    fig_arrow(fig, (0.705, 0.34), (0.735, 0.405))
    fig_arrow(fig, (0.705, 0.34), (0.735, 0.24))
    fig_box(fig, 0.845, 0.25, 0.07, 0.16, fc="#eef4fc", ec=ACC, lw=1.0)
    txt(fig, 0.88, 0.355, "PhysICL", fontweight="bold", fontsize=7.5)
    txt(fig, 0.88, 0.325, "fixed", fontsize=6.4)
    txt(fig, 0.88, 0.302, "weights", fontsize=6.4)
    txt(fig, 0.88, 0.275, "×2 passes", fontsize=6.4, color=MUTED)
    fig_arrow(fig, (0.82, 0.405), (0.845, 0.36))
    fig_arrow(fig, (0.82, 0.24), (0.845, 0.29))
    fig_box(fig, 0.928, 0.27, 0.065, 0.12, fc="white")
    txt(fig, 0.9605, 0.355, r"$\bar p=\frac{1}{2}(p_{\mathrm{raw}}$", fontsize=6.6)
    txt(fig, 0.9605, 0.325, r"$+\,p_{\mathrm{env}})$", fontsize=6.6)
    txt(fig, 0.9605, 0.29, "argmax", fontsize=6.4, color=MUTED)
    fig_arrow(fig, (0.915, 0.33), (0.928, 0.33))
    txt(fig, 0.8, 0.1, "fine-tuned weights from (b) are used unchanged on every target machine",
        fontsize=6.3, color=MUTED, style="italic")

    os.makedirs(out, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(os.path.join(out, f"fig_method.{ext}"), **kw)
    print("fig_method kaydedildi")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figures")
