"""Şekil: simülatör–gerçek ölçüm uyumu (IEEE çift sütun).

Kullanım: python fidelity_figure.py <uyum klasörü> <çıktı klasörü>
<uyum klasörü>: run_pilot2.py adim=uyum çıktısı (uyum_ozellikler.csv, uyum_sim_ornek.csv).
Fizik benzerliği bu dosyalardan yeniden hesaplanır (kontrol önseli pencereleri sabit tohumla yeniden üretilir);
proje kökü (pilot2/) PYTHONPATH'te olmalı.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import COL2, MUTED, plt, save  # noqa: E402

DS = [("cwru", "CWRU"), ("seu", "SEU"), ("pu", "PU"), ("jnu", "JNU"), ("hit_mfs", "HIT-MFS"), ("hit_sb", "HIT-SB")]
TICKS = {"log_kurtosis": [3, 10, 30], "log_env_peak": [2, 5, 10]}
BLUE, LBLUE, GREY, DGREY = "#2a78d6", "#a9c9f0", "#c9c8c2", "#76756f"


def physics_like(feats, sim, seed=0):
    from pilot2.fidelity import FEATS, PhysicsLike, control_windows, features

    Fc = features(control_windows(2000, seed=4242))
    pl = PhysicsLike(sim[FEATS].values, Fc, np.random.default_rng(seed))
    ref_ctrl = features(control_windows(600, seed=999))
    feats = feats.assign(phys=pl.score(feats[FEATS].values) > 0.5, fault=feats.sinif != "N")
    tab = feats.groupby(["kaynak", "fault"]).phys.mean().unstack() * 100
    tab["all"] = feats.groupby("kaynak").phys.mean() * 100
    return tab, 100 * float((pl.score(ref_ctrl) > 0.5).mean())


def main(src, out):
    os.makedirs(out, exist_ok=True)
    feats = pd.read_csv(os.path.join(src, "uyum_ozellikler.csv"))
    sim = pd.read_csv(os.path.join(src, "uyum_sim_ornek.csv"))
    tab, ctrl_ref = physics_like(feats, sim)
    tab.round(1).to_csv(os.path.join(out, "fig8_physics_like.csv"))
    print(tab.round(1), "\ncontrol reference:", round(ctrl_ref, 1))

    panels = [("log_kurtosis", "(a) Kurtosis", lambda v: 10 ** v, "log", None),
              ("centroid_khz", "(b) Spectral centroid (kHz)", lambda v: v, "linear", (0, 4.5)),
              ("e_3_6k", "(c) Energy fraction, 3–6 kHz", lambda v: np.maximum(v, 1e-6), "log", (5e-7, 1.5)),
              ("log_env_peak", "(d) Envelope peak ratio", lambda v: 10 ** v, "log", None)]
    fig, axes = plt.subplots(1, 5, figsize=(COL2, 1.95), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25]})
    xs = np.arange(len(DS))
    for ax, (col, title, tf, scale, lim) in zip(axes[:4], panels):
        for x0, healthy, c in ((-1.15, True, GREY), (-0.8, False, DGREY)):
            v = tf(sim[col][(sim.sinif == "N") == healthy].values)
            q = np.quantile(v, [0.01, 0.25, 0.5, 0.75, 0.99])
            ax.plot([x0, x0], [q[0], q[4]], color=c, lw=0.9, solid_capstyle="butt")
            ax.plot([x0, x0], [q[1], q[3]], color=c, lw=3.6, solid_capstyle="butt")
            ax.plot([x0 - 0.12, x0 + 0.12], [q[2], q[2]], color="white", lw=0.9)
        for i, (k, _) in enumerate(DS):
            d = feats[feats.kaynak == k]
            for healthy, dx, mfc in ((True, -0.17, "white"), (False, 0.17, BLUE)):
                v = tf(d[col][(d.sinif == "N") == healthy].values)
                if len(v) == 0:
                    continue
                q = np.quantile(v, [0.25, 0.5, 0.75])
                ax.errorbar(xs[i] + dx, q[1], yerr=[[q[1] - q[0]], [q[2] - q[1]]], fmt="o", ms=2.8, mfc=mfc,
                            mec=BLUE, mew=0.7, ecolor=BLUE, elinewidth=0.7, capsize=0)
        ax.axvline(2.5, color="#d6d5cf", lw=0.6, zorder=0)
        ax.axvline(-0.45, color="#d6d5cf", lw=0.6, zorder=0)
        ax.set_yscale(scale)
        if lim:
            ax.set_ylim(*lim)
        tk = plt.matplotlib.ticker
        if col in TICKS:
            ax.yaxis.set_major_locator(tk.FixedLocator(TICKS[col]))
            ax.yaxis.set_major_formatter(tk.FormatStrFormatter("%g"))
            ax.yaxis.set_minor_formatter(tk.NullFormatter())
        if col == "e_3_6k":
            ax.yaxis.set_major_formatter(tk.LogFormatterMathtext())
            ax.yaxis.set_minor_formatter(tk.NullFormatter())
            ax.text(1, 2.2e-6, r"$<\!10^{-6}$", ha="center", va="bottom", fontsize=5.8, color=BLUE)
        ax.set_xticks([-0.975] + list(xs))
        ax.set_xticklabels(["Sim."] + [n for _, n in DS], rotation=90)
        ax.set_xlim(-1.45, len(DS) - 0.45)
        ax.set_title(title, loc="left", fontsize=7.5)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="both", labelsize=6.5)

    ax = axes[4]
    w = 0.38
    h = [tab.loc[k, False] if k in tab.index else np.nan for k, _ in DS]
    f = [tab.loc[k, True] if k in tab.index else np.nan for k, _ in DS]
    ax.bar(xs - w / 2, h, w, color="white", edgecolor=BLUE, lw=0.7, label="healthy")
    ax.bar(xs + w / 2, f, w, color=BLUE, edgecolor=BLUE, lw=0.7, label="faulty")
    simf = tab.loc["sim", True]
    ax.axhline(simf, color=DGREY, ls="--", lw=0.8)
    ax.axhline(ctrl_ref, color="#eb6834", ls=":", lw=0.9)
    ax.axvline(2.5, color="#d6d5cf", lw=0.6, zorder=0)
    ax.set_ylim(0, 108)
    ax.set_xticks(xs)
    ax.set_xticklabels([n for _, n in DS], rotation=90)
    ax.set_title("(e) Closer to physics prior (%)", loc="left", fontsize=7.5)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="both", labelsize=6.5)
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "fig8_fidelity")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "figures")
