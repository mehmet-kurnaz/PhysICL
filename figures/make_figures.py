"""PhysICL makale şekilleri (IEEE biçimi: tek sütun 3.5 in, çift sütun 7.16 in, Times uyumlu 8 pt yazı).

Kullanım:  python make_figures.py <sonuclar klasörü> <çıktı klasörü>
  <sonuclar klasörü> = içinde results/ ve results_pilot2/ olan klasör (sonuclar.zip açılmış hali)
Şekil 3 (simülatör örnekleri) için proje kökü (pilot2/ klasörü) PYTHONPATH'te olmalı.
Her şekil PDF (vektör, makaleye bu girer) ve PNG (600 dpi, önizleme) olarak kaydedilir.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from scipy import stats  # noqa: E402

COL1, COL2 = 3.5, 7.16
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.2, "lines.markersize": 4, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3de", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "axes.edgecolor": "#3a3a38", "text.color": "#1f1f1e",
    "axes.labelcolor": "#1f1f1e", "xtick.color": "#3a3a38", "ytick.color": "#3a3a38",
})
INK, MUTED = "#1f1f1e", "#76756f"

# Sabit yöntem kimlikleri: renk + işaret + çizgi (gri baskıda da ayırt edilsin diye ikincil kodlama)
M = {
    "physicl": dict(label="PhysICL (ours)", color="#2a78d6", marker="o", ls="-", lw=1.6),
    "control": dict(label="Control prior (non-physical)", color="#eb6834", marker="s", ls="--", lw=1.1),
    "timee":   dict(label="TimEE, off the shelf", color="#1baf7a", marker="^", ls="-.", lw=1.1),
    "mantis":  dict(label="MantisV2 + LR", color="#6250d6", marker="D", ls=":", lw=1.1),
    "env":     dict(label="Envelope spectrum", color="#e87ba4", marker="v", ls=(0, (5, 1.5, 1, 1.5)), lw=1.1),
}
DS_NAME = {"cwru": "CWRU", "seu": "SEU", "pu": "PU", "jnu": "JNU", "hit_mfs": "HIT-MFS", "hit_sb": "HIT-SB"}
CHANCE = {"cwru": 25, "seu": 20, "pu": 100 / 3, "jnu": 25, "hit_mfs": 100 / 3, "hit_sb": 100 / 3}
PHYS = ["fizik_v1|ham+zarf", "fizik_v1_s1|ham+zarf", "fizik_v1_s2|ham+zarf"]
CTRL = ["genel_v1|ham+zarf", "genel_v1_s1|ham+zarf", "genel_v1_s2|ham+zarf"]


def save(fig, out, name):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), **kw)
    plt.close(fig)
    print("  ", name)


def ci(v):
    v = np.asarray(v, float)
    return stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def load(root, *rel):
    dfs = [pd.read_csv(os.path.join(root, r, "episodes.csv")) for r in rel]
    return pd.concat(dfs).drop_duplicates(["dataset", "episode", "shot", "method"], keep="last")


def pivot(df, d, k):
    g = df[(df.dataset == d) & (df.shot == k)].pivot_table(index="episode", columns="method", values="acc") * 100
    out = {"timee": g.get("timee_raw"), "mantis": g.get("mantisv2_lr"), "env": g.get("env_proto")}
    if all(c in g for c in PHYS):
        out["physicl"] = g[PHYS].mean(1)
    if all(c in g for c in CTRL):
        out["control"] = g[CTRL].mean(1)
    return {k2: v for k2, v in out.items() if v is not None}


# ----------------------------------------------------------------------------- Fig. 1

def fig1(out):
    """PU, 5-shot: aynı rulman vs farklı rulman (pu_sanity.py, 10 epizot)."""
    rows = [("timee", 97.3, 40.9), ("mantis", 98.5, 40.3), ("env", 77.8, 50.5)]
    fig, ax = plt.subplots(figsize=(COL1, 1.95))
    x = np.arange(len(rows))
    w = 0.36
    for i, (m, same, diff) in enumerate(rows):
        c = M[m]["color"]
        ax.bar(i - w / 2, same, w, color=c, alpha=0.3, edgecolor=c, linewidth=0.6)
        ax.bar(i + w / 2, diff, w, color=c, edgecolor=c, linewidth=0.6)
        ax.text(i - w / 2, same + 1.5, f"{same:.1f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w / 2, diff + 1.5, f"{diff:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.axhline(100 / 3, color=MUTED, lw=0.7, ls=":")
    ax.text(-0.62, 100 / 3 + 1.2, "chance", ha="left", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xticks(x, [M[m]["label"].replace(", off the shelf", "") for m, *_ in rows])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.65, len(rows) - 0.35)
    ax.grid(axis="x", visible=False)
    leg = [plt.Rectangle((0, 0), 1, 1, facecolor="#76756f", alpha=0.3, edgecolor="#76756f", lw=0.6),
           plt.Rectangle((0, 0), 1, 1, facecolor="#76756f", edgecolor="#76756f", lw=0.6)]
    ax.legend(leg, ["Same bearing, other files", "Different bearing, same class"], loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)
    save(fig, out, "fig1_same_vs_different_bearing")


# ----------------------------------------------------------------------------- Fig. 2

def fig2(out):
    """Yöntem akışı (vektör)."""
    fig, ax = plt.subplots(figsize=(COL2, 2.25))
    ax.set_xlim(0, 100); ax.set_ylim(0, 31); ax.axis("off")
    edge, acc = "#5f5e57", "#2a78d6"

    def box(cx, cy, w, h, title, lines, main=False):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                                    fc=(acc if main else "white"), alpha=1, ec=(acc if main else edge),
                                    lw=(1.3 if main else 0.7)))
        if main:
            ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                                        fc="white", alpha=0.86, ec="none"))
        n = len(lines)
        ax.text(cx, cy + (1.3 if n else 0) + (0.6 if n == 2 else 0), title, ha="center", va="center",
                fontsize=8, fontweight="bold", color=INK)
        for j, ln in enumerate(lines):
            ax.text(cx, cy - 0.9 - 1.9 * j + (0.6 if n == 2 else 0), ln, ha="center", va="center",
                    fontsize=6.8, color=INK if main else "#3a3a38")

    def arrow(p, q, rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.7, color=edge,
                                     shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))

    ax.text(0.5, 30.2, "(a) Fine-tuning on the randomized simulator — no real data", fontsize=7.5,
            style="italic", color=MUTED, va="center")
    ax.text(0.5, 14.2, "(b) Diagnosis on a new machine — no weight update", fontsize=7.5,
            style="italic", color=MUTED, va="center")
    W, H, yA = 21, 8.6, 23.5
    xs = [11.5, 37.5, 63.5, 89.0]
    box(xs[0], yA, W, H, "Virtual machine", ["geometry, resonances", "sensor band, tones"])
    box(xs[1], yA, W, H, "Parts and conditions", ["defect size, mounting", "speed or load change"])
    box(xs[2], yA, W, H, "Training episodes", ["support from one part", "queries from other parts"])
    box(xs[3], yA, W, H, "Fine-tune TimEE", ["2000 steps, 8 episodes/step", "cross-entropy on queries"])
    for a, b in zip(xs[:-1], xs[1:]):
        arrow((a + W / 2, yA), (b - W / 2, yA))
    yB = 6.0
    box(xs[0], yB, W, H, "New machine", ["k labeled windows", "+ unlabeled queries"])
    box(xs[1], yB + 2.6, W, 4.2, "Raw window", [])
    box(xs[1], yB - 2.6, W, 4.2, "Envelope (>500 Hz)", [])
    box(xs[2], yB, W, H, "PhysICL", ["one model, two views", "one forward pass each"], main=True)
    box(xs[3], yB, W, H, "Diagnosis", ["mean of the two", "probability vectors"])
    arrow((xs[0] + W / 2, yB + 0.8), (xs[1] - W / 2, yB + 2.6))
    arrow((xs[0] + W / 2, yB - 0.8), (xs[1] - W / 2, yB - 2.6))
    arrow((xs[1] + W / 2, yB + 2.6), (xs[2] - W / 2, yB + 0.8))
    arrow((xs[1] + W / 2, yB - 2.6), (xs[2] - W / 2, yB - 0.8))
    arrow((xs[2] + W / 2, yB), (xs[3] - W / 2, yB))
    ym = 15.4                                     # iki şeridin arasındaki boşluk
    ax.plot([xs[3], xs[3], xs[2]], [yA - H / 2, ym, ym], color=edge, lw=0.7, solid_joinstyle="miter")
    arrow((xs[2], ym), (xs[2], yB + H / 2))
    ax.text((xs[2] + xs[3]) / 2, ym + 0.6, "fine-tuned weights", fontsize=6.8, color=MUTED, ha="center",
            va="bottom")
    save(fig, out, "fig2_pipeline")


# ----------------------------------------------------------------------------- Fig. 3

def fig3(out):
    """Simülatörden örnekler: aynı sanal makinede N/OR/IR/B, ham pencere + zarf spektrumu."""
    from scipy.signal import hilbert
    from pilot2.simulator import FS, render, sample_machine, sample_unit

    rng = np.random.default_rng(7)
    m = sample_machine(rng)
    while m.fc < 5000 or m.conditions[0][0] < 20:
        m = sample_machine(rng)
    fr = m.conditions[0][0]
    o = m.orders()
    classes = [("N", "Healthy"), ("OR", "Outer race"), ("IR", "Inner race"), ("B", "Ball")]
    fig, axes = plt.subplots(len(classes), 2, figsize=(COL2, 3.3), gridspec_kw={"width_ratios": [1, 1.35]})
    for i, (c, name) in enumerate(classes):
        u = sample_unit(m, c, rng)
        if c != "N":
            u.snr_db = 6.0                      # gösterim için okunur bir şiddet
        x = render(m, u, 0, 16384, rng)
        t = np.arange(1024) / FS * 1e3
        a0 = axes[i, 0]
        a0.plot(t, x[:1024] / x.std(), color="#2a78d6", lw=0.5)
        a0.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=7.5)
        a0.set_yticks([])
        a0.grid(False)
        xx = x - x.mean()
        env = np.abs(hilbert(xx)); env -= env.mean()
        spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) / len(env)
        f = np.fft.rfftfreq(len(env), 1 / FS)
        xmax = float(np.ceil(1.15 * max(o["BPFI"], 2 * o["BSF"])))
        keep = f <= xmax * fr
        a1 = axes[i, 1]
        a1.plot(f[keep] / fr, spec[keep] / spec[keep].max(), color="#2a78d6", lw=0.6)
        for lab, val in (("BSF", o["BSF"]), ("2BSF", 2 * o["BSF"]), ("BPFO", o["BPFO"]), ("BPFI", o["BPFI"])):
            a1.axvline(val, color=MUTED, lw=0.5, ls="--")
            if i == 0:
                a1.text(val, 1.03, lab, ha="center", va="bottom", fontsize=6.5, color=MUTED,
                        transform=a1.get_xaxis_transform())
        a1.set_yticks([])
        a1.set_xlim(0, xmax)
        a1.grid(False)
        if i < len(classes) - 1:
            a0.set_xticklabels([]); a1.set_xticklabels([])
    axes[-1, 0].set_xlabel("Time (ms)")
    axes[-1, 1].set_xlabel(r"Envelope-spectrum frequency / shaft frequency $f_r$")
    axes[0, 0].set_title("Raw window (1024 samples, z-scored)", pad=14)
    axes[0, 1].set_title(f"Envelope spectrum ($f_r$ = {fr:.1f} Hz); dashed: fault orders", pad=14)
    fig.tight_layout(h_pad=0.3, w_pad=1.0)
    save(fig, out, "fig3_simulator_examples")


# ----------------------------------------------------------------------------- Fig. 4

def fig4(root, out):
    """Altı veri setinde doğruluk–shot. Geliştirme: tohum-1 doğrulama epizotları; dış: dışarıda tutulan test."""
    dev = load(root, "dogrulama_taban", "A_tohum")
    held = load(root, "A_taban", "A_dis_test")
    panels = [("cwru", dev), ("seu", dev), ("pu", dev), ("jnu", held), ("hit_mfs", held), ("hit_sb", held)]
    order = ["env", "timee", "mantis", "control", "physicl"]
    fig, axes = plt.subplots(2, 3, figsize=(COL2, 3.6), sharex=True)
    shots = [1, 5, 10]
    for ax, (d, df) in zip(axes.ravel(), panels):
        for m in order:
            ys, es = [], []
            for k in shots:
                p = pivot(df, d, k)
                if m not in p:
                    ys.append(np.nan); es.append(0); continue
                ys.append(p[m].mean()); es.append(ci(p[m]))
            st = M[m]
            ax.errorbar(range(3), ys, yerr=es, color=st["color"], marker=st["marker"], ls=st["ls"],
                        lw=st["lw"], capsize=1.8, elinewidth=0.6, markersize=3.6 if m != "physicl" else 4.4,
                        zorder=5 if m == "physicl" else 3)
        ax.axhline(CHANCE[d], color=MUTED, lw=0.6, ls=":")
        ax.set_title(DS_NAME[d] + (" (held out)" if d in ("jnu", "hit_mfs", "hit_sb") else ""), pad=3)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(3), ["1", "5", "10"])
        ax.set_xlim(-0.25, 2.25)
    for a in axes[1]:
        a.set_xlabel("Shots per class")
    for a in axes[:, 0]:
        a.set_ylabel("Accuracy (%)")
    handles = [Line2D([], [], color=M[m]["color"], marker=M[m]["marker"], ls=M[m]["ls"], lw=M[m]["lw"],
                      markersize=4) for m in reversed(order)]
    handles.append(Line2D([], [], color=MUTED, ls=":", lw=0.6))
    fig.legend(handles, [M[m]["label"] for m in reversed(order)] + ["Chance"], loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.04), handlelength=2.4, columnspacing=1.2)
    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    save(fig, out, "fig4_accuracy_vs_shots")


# ----------------------------------------------------------------------------- Fig. 5

def fig5(root, out):
    """Fizik − kontrol (3 tohum ortalaması), eşleştirilmiş fark ve %95 GA; üç bağımsız epizot kümesinde."""
    sets = [("Confirmation\n(seed 1)", load(root, "dogrulama_taban", "A_tohum"), ["cwru", "seu", "pu"]),
            ("Replication\n(seed 2)", load(root, "C_taban"), ["cwru", "seu", "pu"]),
            ("Held out", load(root, "A_taban", "A_dis_test"), ["jnu", "hit_mfs", "hit_sb"])]
    rows = []
    for sname, df, dss in sets:
        for d in dss:
            for k in (5, 10):
                p = pivot(df, d, k)
                diff = (p["physicl"] - p["control"]).dropna()
                rows.append((sname, d, k, diff.mean(), ci(diff), stats.wilcoxon(diff).pvalue))
    fig, ax = plt.subplots(figsize=(COL1, 3.1))
    y, ticks, labels = 0, [], []
    for sname in [s[0] for s in sets]:
        y_first = y
        for d in dict.fromkeys(r[1] for r in rows if r[0] == sname):
            for k, mk, off in ((5, "o", 0.16), (10, "s", -0.16)):
                r = next(r for r in rows if r[0] == sname and r[1] == d and r[2] == k)
                sig = r[5] < 0.05
                ax.errorbar(r[3], y + off, xerr=r[4], fmt=mk, color="#2a78d6",
                            mfc=("#2a78d6" if sig else "white"), mec="#2a78d6", ms=3.6, elinewidth=0.7, capsize=1.5)
            ticks.append(y); labels.append(DS_NAME[d])
            y -= 1
        ax.text(1.02, (y_first + y + 1) / 2, sname, transform=ax.get_yaxis_transform(), fontsize=7,
                va="center", ha="left", color=MUTED)
        y -= 0.6
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(ticks, labels)
    ax.set_xlabel("PhysICL − control prior (accuracy points)")
    ax.grid(axis="y", visible=False)
    handles = [Line2D([], [], color="#2a78d6", marker="o", ls="", ms=3.6),
               Line2D([], [], color="#2a78d6", marker="s", ls="", ms=3.6),
               Line2D([], [], color="#2a78d6", marker="o", mfc="white", ls="", ms=3.6)]
    ax.legend(handles, ["5-shot", "10-shot", "open: p ≥ 0.05"], loc="lower right", handletextpad=0.2)
    save(fig, out, "fig5_physics_vs_control")
    pd.DataFrame(rows, columns=["set", "dataset", "shot", "mean_diff", "ci95", "p_wilcoxon"]).to_csv(
        os.path.join(out, "fig5_values.csv"), index=False)


# ----------------------------------------------------------------------------- Fig. 6

ABL = [("abl_birim_yok", "No part-level variation"), ("abl_kosul_yok", "No condition change"),
       ("abl_bant_yok", "Fixed sensor bandwidth"), ("abl_golge_yok", "No gear/line tones"),
       ("abl_mil_yok", "No shaft-fault classes"), ("uzunluk_500", "500 steps"),
       ("uzunluk_1000", "1000 steps"), ("uzunluk_4000", "4000 steps")]


def fig6(root, out):
    """Ablasyon: tam modele göre fark (10-shot), iki bağımsız epizot kümesinde."""
    sets = [("seed 1", load(root, "B_ablasyon"), "o", INK), ("seed 2", load(root, "C_taban"), "s", "#8a897f")]
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.35), sharey=True)
    for ax, d in zip(axes, ["cwru", "seu", "pu"]):
        for j, (sname, df, mk, c) in enumerate(sets):
            g = df[(df.dataset == d) & (df.shot == 10)].pivot_table(index="episode", columns="method",
                                                                    values="acc") * 100
            ref = g["fizik_v1|ham+zarf"]
            for i, (tag, _) in enumerate(ABL):
                col = f"{tag}|ham+zarf"
                if col not in g:
                    continue
                diff = (g[col] - ref).dropna()
                ax.errorbar(diff.mean(), -i + (0.15 if j == 0 else -0.15), xerr=ci(diff), fmt=mk, color=c, ms=3.4,
                            elinewidth=0.7, capsize=1.4)
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_title(f"{DS_NAME[d]}, 10-shot", pad=3)
        ax.set_xlabel("Change vs. full model (points)")
        ax.grid(axis="y", visible=False)
        ax.axhspan(-4.5, -7.5, color="#f2f1ed", zorder=0, lw=0)
    axes[0].set_yticks([-i for i in range(len(ABL))], [a[1] for a in ABL])
    handles = [Line2D([], [], color=c, marker=mk, ls="", ms=3.4) for _, _, mk, c in sets]
    fig.legend(handles, ["Episode set 1 (seed 1)", "Episode set 2 (seed 2), independent replication"],
               loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.07), handletextpad=0.2)
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "fig6_ablations")


# ----------------------------------------------------------------------------- Fig. 7

def fig7(root, out):
    """Sentetik doğrulama doğruluğu (fizik önseli doğrulama kümesi) — eğitim adımı; fizik ve kontrol, 3 tohum."""
    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    for tags, m in ((["fizik_v1", "fizik_v1_s1", "fizik_v1_s2"], "physicl"),
                    (["genel_v1", "genel_v1_s1", "genel_v1_s2"], "control")):
        curves = []
        for t in tags:
            f = os.path.join(root, f"ince_ayar_{t}", "log.csv")
            if os.path.exists(f):
                lg = pd.read_csv(f)
                curves.append(lg.set_index("step")["val_acc_fizik"] * 100)
                ax.plot(lg.step, lg.val_acc_fizik * 100, color=M[m]["color"], lw=0.5, alpha=0.45, ls=M[m]["ls"])
        if curves:
            mean = pd.concat(curves, axis=1).mean(1)
            ax.plot(mean.index, mean.values, color=M[m]["color"], marker=M[m]["marker"], ls=M[m]["ls"],
                    lw=1.4, ms=3.6, label=M[m]["label"].replace(" (ours)", " prior (ours)") if m == "physicl"
                    else M[m]["label"])
    ax.set_xlabel("Fine-tuning step")
    ax.set_ylabel("Synthetic val. accuracy (%)")
    ax.set_ylim(45, 72)
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.legend(loc="lower right")
    save(fig, out, "fig7_training_curves")


if __name__ == "__main__":
    root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "sonuclar")
    out = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else "figures")
    os.makedirs(out, exist_ok=True)
    r2 = os.path.join(root, "results_pilot2")
    print("Şekiller:", out)
    fig1(out)
    fig2(out)
    try:
        fig3(out)
    except ImportError as e:
        print("   fig3 atlandı (pilot2 bulunamadı):", e)
    fig4(r2, out)
    fig5(r2, out)
    fig6(r2, out)
    fig7(r2, out)
