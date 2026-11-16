"""Simülatör v0 önizlemesi: ince ayara geçmeden önce sentetik veri makul mü?

Üç kontrol:
  1. Görsel   : her sınıftan bir pencere + zarf spektrumu (arıza frekansı çizgileriyle);
                aynı sınıfın farklı makinelerdeki görünümü (çeşitlilik).
  2. Gerçekçilik: pencere öznitelikleri (basıklık, tepe faktörü, spektral ağırlık merkezi,
                <500 Hz enerji oranı) sentetik ve gerçek veri setlerinde karşılaştırılır.
                "Kapsama": gerçek pencerelerin yüzde kaçı sentetik dağılımın %1-%99 aralığında.
  3. Zorluk   : pilot 1 yöntemleri sentetik makinelerde, pilot 1 ile aynı bölmelerle denenir.
                Çok kolay (her şey %100) ya da imkânsız (her şey şans) olmamalı.
"""

from __future__ import annotations

import os
import time

import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis

from .simulator import CLASSES, FS, make_dataset, render, sample_episode, sample_machine, sample_unit

# Sabit kategorik renkler (her varlık her grafikte aynı renk)
COL = {"sentetik": "#2a78d6", "cwru": "#eb6834", "seu": "#1baf7a", "pu": "#eda100"}
INK, MUTED, GRID = "#2b2b2a", "#76756f", "#e4e3de"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlesize": 9, "axes.titlecolor": INK,
                         "figure.facecolor": "white", "axes.facecolor": "white", "lines.linewidth": 1.0})
    return plt


def envelope_spectrum(x, fs=FS):
    x = x - x.mean()
    env = np.abs(hilbert(x))
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) / len(env)
    return np.fft.rfftfreq(len(env), 1 / fs), spec


# ----------------------------------------------------------------------------- 1. görsel

def plot_classes(rng, path, n_env=8192):
    """Bir makinede her sınıf: zaman penceresi (1024) + zarf spektrumu (8192 örnekle, çözünürlük için)."""
    plt = _style()
    m = sample_machine(rng)
    while m.fc < 5000:                       # gösterim için geniş bantlı bir makine
        m = sample_machine(rng)
    fr = m.conditions[0][0]
    o = m.orders()
    fig, ax = plt.subplots(len(CLASSES), 2, figsize=(11, 1.55 * len(CLASSES)),
                           gridspec_kw={"width_ratios": [1, 1.4]})
    for i, cls in enumerate(CLASSES):
        u = sample_unit(m, cls, rng)
        x = render(m, u, 0, n_env, rng)
        t = np.arange(1024) / FS * 1000
        ax[i, 0].plot(t, x[:1024] / x.std(), color=COL["sentetik"], lw=0.7)
        ax[i, 0].set_ylabel(cls, rotation=0, ha="right", va="center", fontsize=10, color=INK)
        f, s = envelope_spectrum(x)
        keep = f <= 600
        ax[i, 1].plot(f[keep], s[keep], color=COL["sentetik"], lw=0.9)
        marks = {"1x": fr, "BPFO": o["BPFO"] * fr, "BPFI": o["BPFI"] * fr, "BSF": o["BSF"] * fr}
        for j, (name, fm) in enumerate(sorted(marks.items(), key=lambda kv: kv[1])):
            if fm <= 600:
                ax[i, 1].axvline(fm, color=MUTED, lw=0.7, ls="--")
                if i == 0:   # yakın etiketler çakışmasın diye iki sıraya dağıt
                    ax[i, 1].text(fm, 1.03 + 0.16 * (j % 2), name, color=MUTED, fontsize=7, ha="center",
                                  va="bottom", transform=ax[i, 1].get_xaxis_transform())
        ax[i, 1].set_yticks([])
        if i < len(CLASSES) - 1:
            ax[i, 0].set_xticklabels([]); ax[i, 1].set_xticklabels([])
    ax[-1, 0].set_xlabel("zaman (ms)")
    ax[-1, 1].set_xlabel("zarf spektrumu frekansı (Hz)")
    ax[0, 0].set_title("Ham pencere (1024 örnek, z-skorlu)", pad=26)
    ax[0, 1].set_title(f"Zarf spektrumu  (mil {fr:.1f} Hz, bant sınırı {m.fc:.0f} Hz; kesikli: arıza frekansları)", pad=26)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_diversity(rng, path, cls="OR", n_machines=6, n=8192):
    """Aynı sınıf (OR) farklı makinelerde: güç spektrumu ve zarf spektrumu (arıza düzeyi eksende)."""
    plt = _style()
    fig, ax = plt.subplots(n_machines, 2, figsize=(11, 1.5 * n_machines))
    for i in range(n_machines):
        m = sample_machine(rng)
        u = sample_unit(m, cls, rng)
        x = render(m, u, 0, n, rng)
        X = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
        f = np.fft.rfftfreq(n, 1 / FS)
        ax[i, 0].semilogy(f, X + 1e-12, color=COL["sentetik"], lw=0.6)
        ax[i, 0].set_xlim(0, 6000)
        ax[i, 0].set_yticks([])
        fe, se = envelope_spectrum(x)
        fr, bpfo = m.conditions[0][0], m.orders()["BPFO"]
        order = fe / fr
        keep = order <= 12
        ax[i, 1].plot(order[keep], se[keep], color=COL["sentetik"], lw=0.9)
        for h in (1, 2, 3):
            if h * bpfo <= 12:
                ax[i, 1].axvline(h * bpfo, color=MUTED, lw=0.7, ls="--")
        ax[i, 1].set_yticks([])
        ax[i, 0].set_ylabel(f"makine {i + 1}", rotation=0, ha="right", va="center", color=INK)
        ax[i, 1].text(0.99, 0.85, f"mil {fr:.0f} Hz · bant {m.fc:.0f} Hz · SNR {u.snr_db:+.0f} dB",
                      transform=ax[i, 1].transAxes, ha="right", fontsize=7, color=MUTED,
                      bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
        if i < n_machines - 1:
            ax[i, 0].set_xticklabels([]); ax[i, 1].set_xticklabels([])
    ax[-1, 0].set_xlabel("frekans (Hz)")
    ax[-1, 1].set_xlabel("zarf spektrumu, mil devrine oranlı (order)")
    ax[0, 0].set_title(f"{cls}: güç spektrumu (log) — rezonanslar makineden makineye değişir")
    ax[0, 1].set_title("Zarf spektrumu — kesikli: 1×, 2×, 3× BPFO")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- 2. gerçekçilik

FEATURES = ("basıklık", "tepe faktörü", "spektral merkez (Hz)", "<500 Hz enerji oranı")


def window_features(W: np.ndarray) -> np.ndarray:
    W = W - W.mean(1, keepdims=True)
    k = kurtosis(W, axis=1)
    crest = np.abs(W).max(1) / (W.std(1) + 1e-12)
    P = np.abs(np.fft.rfft(W * np.hanning(W.shape[1]), axis=1)) ** 2
    f = np.fft.rfftfreq(W.shape[1], 1 / FS)
    cent = (P * f).sum(1) / (P.sum(1) + 1e-12)
    low = P[:, f < 500].sum(1) / (P.sum(1) + 1e-12)
    return np.stack([k, crest, cent, low], 1)


def synthetic_features(rng, n_machines=200, per=8, classes=("N", "OR", "IR", "B")):
    out = {c: [] for c in classes}
    for _ in range(n_machines):
        m = sample_machine(rng)
        for c in classes:
            u = sample_unit(m, c, rng)
            cond = int(rng.integers(len(m.conditions)))
            W = render(m, u, cond, per * 1024, rng).reshape(per, 1024)
            out[c].append(window_features(W))
    return {c: np.concatenate(v) for c, v in out.items()}


def real_features(recs, per_rec=40):
    out = {}
    for r in recs:
        n = len(r.signal) // 1024
        if n == 0:
            continue
        W = r.signal[: n * 1024].reshape(n, 1024)
        W = W[np.linspace(0, n - 1, min(n, per_rec)).astype(int)]
        out.setdefault(r.label, []).append(window_features(W))
    return {c: np.concatenate(v) for c, v in out.items()}


def plot_realism(syn, reals, path):
    plt = _style()
    classes = list(syn)
    fig, ax = plt.subplots(len(FEATURES), len(classes), figsize=(2.9 * len(classes), 2.0 * len(FEATURES)))
    clip = {0: (-2, 30), 1: (1, 15), 2: (0, 6000), 3: (0, 1)}
    for j, c in enumerate(classes):
        for i, name in enumerate(FEATURES):
            a = ax[i, j]
            lo, hi = clip[i]
            bins = np.linspace(lo, hi, 40)
            a.hist(np.clip(syn[c][:, i], lo, hi), bins=bins, density=True, color=COL["sentetik"],
                   alpha=0.35, label="sentetik")
            for ds, F in reals.items():
                if c in F:
                    a.hist(np.clip(F[c][:, i], lo, hi), bins=bins, density=True, histtype="step",
                           color=COL[ds], lw=1.6, label=ds.upper())
            a.set_yticks([])
            if j == 0:
                a.set_ylabel(name, color=INK)
            if i == 0:
                a.set_title(c, fontsize=10)
    h, l = ax[0, 0].get_legend_handles_labels()
    for a in ax[0, 1:]:
        hh, ll = a.get_legend_handles_labels()
        for x, y in zip(hh, ll):
            if y not in l:
                h.append(x); l.append(y)
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def coverage(syn, reals):
    """Gerçek pencerelerin, aynı sınıfın sentetik %1-%99 aralığına düşme oranı (öznitelik başına)."""
    rows = []
    for ds, F in reals.items():
        for c, R in F.items():
            if c not in syn:
                continue
            lo, hi = np.percentile(syn[c], 1, axis=0), np.percentile(syn[c], 99, axis=0)
            inside = ((R >= lo) & (R <= hi)).mean(0) * 100
            rows.append((ds, c, len(R), *inside))
    return rows


# ----------------------------------------------------------------------------- 3. zorluk

def difficulty(rng, methods: dict, n_machines=10, shot=5, q=30, classes=("N", "OR", "IR", "B")):
    """Her makinede pilot 1 bölmeleriyle bir epizot: aynı koşul/birim, farklı koşul, farklı birim."""
    from pilot1.episodes import EpisodeSampler

    res = {}
    for mi in range(n_machines):
        m = sample_machine(rng)
        for split, mode in (("farkli_kosul", "condition"), ("farkli_birim", "unit")):
            _, recs = make_dataset(rng, classes, split=mode, n_units=3, n_windows=40, machine=m)
            smp = EpisodeSampler(recs, 1024, 400, "global_group" if mode == "condition" else "per_class_group")
            ep = smp.sample(rng, shot, q)
            Xs, ys = ep.support(shot)
            for n, meth in methods.items():
                pred = meth.fit_predict(Xs, ys, ep.Xq)[0]
                res.setdefault((split, n), []).append(float((pred == ep.yq).mean()))
        # üst sınır: aynı birim, aynı koşul, farklı pencereler (sızıntılı ama ayırt edilebilirlik ölçüsü)
        _, recs = make_dataset(rng, classes, split="condition", n_units=1, n_windows=40, machine=m)
        recs = [r for r in recs if r.group == "c0"]
        Xs, ys, Xq, yq = [], [], [], []
        for r in recs:
            W = r.signal.reshape(-1, 1024)
            W = ((W - W.mean(1, keepdims=True)) / (W.std(1, keepdims=True) + 1e-8)).astype(np.float32)
            Xs.append(W[:shot]); ys += [r.label] * shot
            Xq.append(W[shot:shot + q]); yq += [r.label] * len(W[shot:shot + q])
        Xs, Xq, ys, yq = np.concatenate(Xs), np.concatenate(Xq), np.asarray(ys), np.asarray(yq)
        for n, meth in methods.items():
            pred = meth.fit_predict(Xs, ys, Xq)[0]
            res.setdefault(("ayni_kayit", n), []).append(float((pred == yq).mean()))
        print(f"  zorluk: makine {mi + 1}/{n_machines}")
    return res


def speed(rng, n=20):
    t = time.time()
    sizes = []
    for _ in range(n):
        X, *_ = sample_episode(rng)
        sizes.append(len(X))
    dt = (time.time() - t) / n
    return dt, float(np.mean(sizes))


# ----------------------------------------------------------------------------- ana

def run(s: dict):
    out = os.path.join(s["out"], "onizleme")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(s["seed"])
    lines = ["# Simülatör v0 önizleme\n"]

    print("1/4 görsel kontrol grafikleri...")
    plot_classes(rng, os.path.join(out, "1_siniflar.png"))
    plot_diversity(rng, os.path.join(out, "2_makine_cesitliligi.png"))

    print("2/4 gerçekçilik: öznitelik dağılımları...")
    syn = synthetic_features(rng, n_machines=40 if s["test"] else 200)
    reals = {}
    if not s["test"]:
        from pilot1.config import Config
        from pilot1.data import load_dataset
        cfg = Config(data_root=s["data_root"])
        for ds in [d.strip().lower() for d in s["gercek_veri"].split(",") if d.strip()]:
            try:
                recs = load_dataset(ds, cfg)
                F = real_features(recs)
                reals[ds] = {c: v for c, v in F.items() if c in syn}
                print(f"  {ds}: " + ", ".join(f"{c}:{len(v)}" for c, v in reals[ds].items()))
            except Exception as e:
                print(f"  uyarı: {ds} yüklenemedi ({e}); atlanıyor")
    plot_realism(syn, reals, os.path.join(out, "3_gercek_vs_sentetik.png"))
    rows = coverage(syn, reals)
    lines += ["## Gerçekçilik: kapsama (%)\n",
              "Gerçek pencerelerin, aynı sınıfın sentetik %1–%99 aralığına düşme oranı. "
              "Düşük değer, simülatörün o veri setinin o özelliğini üretemediğini gösterir.\n",
              "| veri | sınıf | pencere | " + " | ".join(FEATURES) + " |",
              "|---|---|---|" + "---|" * len(FEATURES)]
    lines += [f"| {d.upper()} | {c} | {n} | " + " | ".join(f"{v:.0f}" for v in vals) + " |"
              for d, c, n, *vals in rows]
    if not rows:
        lines.append("| (gerçek veri yüklenmedi) |")

    print("3/4 zorluk: pilot 1 yöntemleri sentetik makinelerde...")
    names = [m.strip() for m in s["zorluk_yontemleri"].split(",") if m.strip()]
    if {"timee", "timee_raw", "mantisv2_lr"} & set(names):
        import torch  # noqa: F401  (Windows: TF'den önce)
    from pilot1.config import Config
    from pilot1.run import build_methods
    methods = build_methods(Config(random_init_fm=s["test"]), names)
    res = difficulty(rng, methods, n_machines=3 if s["test"] else s["zorluk_makine"])
    splits = ("ayni_kayit", "farkli_kosul", "farkli_birim")
    lines += ["\n## Zorluk: 5-shot doğruluk (%), N/OR/IR/B, şans = 25\n",
              "Her satır ayrı sentetik makinelerin ortalaması. `ayni_kayit` bir üst sınırdır "
              "(aynı kaydın farklı pencereleri); pilot 1'deki gerçek bölmeler `farkli_kosul` (CWRU/SEU) "
              "ve `farkli_birim` (PU) sütunlarıdır.\n",
              "| yöntem | " + " | ".join(splits) + " |", "|---|" + "---|" * len(splits)]
    for n in names:
        lines.append(f"| {n} | " + " | ".join(f"{100 * np.mean(res[(sp, n)]):.1f}" for sp in splits) + " |")

    print("4/4 hız...")
    dt, size = speed(rng, 5 if s["test"] else 20)
    lines += [f"\n## Hız\n\nBir eğitim epizodu ({size:.0f} pencere ortalama) {dt * 1000:.0f} ms'de üretiliyor "
              f"(tek çekirdek). 10 000 adımlık ince ayar için yalnızca üretim ≈ {dt * 1e4 / 60:.0f} dk.\n"]

    md = "\n".join(lines)
    with open(os.path.join(out, "onizleme.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)
    print(f"\nGrafikler ve özet: {os.path.abspath(out)}")
