"""Simülatör–gerçek ölçüm uyumu (kapsama analizi).

Soru: gerçek ölçüm pencereleri, fizik önselinin ürettiği pencerelerin kapsadığı dağılımın içinde mi?
İddia "simülatör makine X'i birebir üretir" DEĞİLDİR; yalnızca kapsama ölçülür.

Her pencere (12 kHz, 1024 örnek; modellerin gördüğü temsil) için 8 yorumlanabilir büyüklük:
  basıklık, tepe faktörü, spektral merkez, 4 bantta enerji payı (0-0,5 / 0,5-1,5 / 1,5-3 / 3-6 kHz),
  zarf spektrumu tepe oranı (500 Hz yüksek geçiren + Hilbert; 5-500 Hz'te en büyük / ortanca).
Hepsi genliğe duyarsızdır (z-skor etkisizdir).

Kapsama (sınıf bazında; gerçek N -> simülatör N, IR -> IR, ...):
  - Ortak (8 boyut): simülatör özellikleriyle standartlaştırılmış uzayda k=10 en yakın simülatör
    penceresine ortalama uzaklık. Eşik = ayrı tutulmuş simülatör pencerelerinde bu istatistiğin %99'luk
    değeri. Gerçek pencere eşiğin altındaysa "kapsanmış" sayılır (simülatör kendisi için tanım gereği ~%99).
  - Marjinal: her büyüklük için gerçek değerin simülatörün %0,5-%99,5 aralığında kalma oranı.
Kapsama tek başına geniş bir önselde gevşek bir ölçüttür (kontrol önselinin pencerelerinin de çoğu kapsanır).
Bu yüzden ikinci ölçüt: "fizik benzerliği" = aynı 8 boyutta, eşit sayıda fizik önseli ve kontrol önseli
penceresinden oluşan havuzda k=25 en yakın komşunun çoğunluğu fizik önselindense pencere "fizik benzeri".
Referans satırları: bağımsız yeni simülatör pencereleri ve bağımsız kontrol önseli pencereleri
(ölçütlerin bu iki uçta ne verdiğini gösterir).

Çıktılar (results_pilot2/uyum/): uyum_tablo.md, uyum_tablo.csv, uyum_marjinal.csv,
uyum_ozellikler.csv (pencere başına özellikler), uyum.png/.pdf (hızlı ön izleme şekli).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.spatial import cKDTree
from scipy.stats import kurtosis

FS = 12_000
L = 1024
SIM_CLASSES = ("N", "OR", "IR", "B", "OR+IR")
REAL_TO_SIM = {"N": "N", "OR": "OR", "IR": "IR", "B": "B", "IR+OR": "OR+IR", "OR+IR": "OR+IR"}
FEATS = ["log_kurtosis", "log_crest", "centroid_khz", "e_0_0.5k", "e_0.5_1.5k", "e_1.5_3k", "e_3_6k", "log_env_peak"]
BANDS = [(0, 500), (500, 1500), (1500, 3000), (3000, 6000)]
_HP = butter(4, 500, btype="highpass", fs=FS, output="sos")
_FREQ = np.fft.rfftfreq(L, 1 / FS)
_WIN = np.hanning(L)
_ENV_BAND = (_FREQ >= 5) & (_FREQ <= 500)


def features(W: np.ndarray) -> np.ndarray:
    """W: (n, L) pencereler -> (n, 8) özellikler."""
    W = np.asarray(W, dtype=np.float64)
    W = W - W.mean(1, keepdims=True)
    rms = np.sqrt((W ** 2).mean(1)) + 1e-12
    kurt = kurtosis(W, axis=1, fisher=False)                        # Gauss = 3
    crest = np.abs(W).max(1) / rms
    P = np.abs(np.fft.rfft(W * _WIN, axis=1)) ** 2
    P[:, 0] = 0.0
    tot = P.sum(1) + 1e-20
    cent = (P * _FREQ).sum(1) / tot / 1000.0
    bands = [P[:, (_FREQ >= lo) & (_FREQ < hi)].sum(1) / tot for lo, hi in BANDS]
    env = np.abs(hilbert(sosfiltfilt(_HP, W, axis=1), axis=1))
    env = env - env.mean(1, keepdims=True)
    E = np.abs(np.fft.rfft(env * _WIN, axis=1))[:, _ENV_BAND]
    peak = E.max(1) / (np.median(E, 1) + 1e-12)
    return np.column_stack([np.log10(kurt), np.log10(crest), cent, *bands, np.log10(peak)])


# ----------------------------------------------------------------------------- simülatör ve kontrol önseli

def simulate(n_machines: int, seed: int, opts: dict | None = None, per_unit: int = 2):
    from .simulator import render, sample_machine, sample_unit

    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_machines):
        m = sample_machine(rng, opts)
        cond = int(rng.integers(len(m.conditions)))
        for cls in SIM_CLASSES:
            u = sample_unit(m, cls, rng, opts)
            sig = render(m, u, cond, per_unit * L, rng)
            X.append(sig[: per_unit * L].reshape(per_unit, L))
            y += [cls] * per_unit
    return np.concatenate(X), np.asarray(y)


def control_windows(n_episodes: int, seed: int) -> np.ndarray:
    from .generic_prior import sample_episode

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_episodes):
        X, _, _ = sample_episode(rng, L=L)[:3]
        out.append(np.asarray(X)[rng.permutation(len(X))[:10]])
    return np.concatenate(out)


# ----------------------------------------------------------------------------- gerçek veri

def real_windows(name: str, cfg, n_max: int, rng) -> tuple[np.ndarray, np.ndarray]:
    from pilot1.data import load_dataset
    from pilot1.episodes import windows_of

    recs = load_dataset(name, cfg)
    X, y = [], []
    for r in recs:
        W = windows_of(r.signal, L, cfg.max_windows_per_recording)
        X.append(W)
        y += [r.label] * len(W)
    X, y = np.concatenate(X), np.asarray(y)
    if len(X) > n_max:                                    # sınıflar arası dengeli alt örnekleme
        per = n_max // len(set(y))
        idx = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:per] for c in sorted(set(y))])
        X, y = X[idx], y[idx]
    return X, y


# ----------------------------------------------------------------------------- kapsama

class PhysicsLike:
    """Fizik önseli ile kontrol önselini ayıran k-NN (eşit havuz). score > 0,5: fizik benzeri."""

    def __init__(self, F_phys: np.ndarray, F_ctrl: np.ndarray, rng, k: int = 25):
        n = min(len(F_phys), len(F_ctrl))
        A = F_phys[rng.permutation(len(F_phys))[:n]]
        B = F_ctrl[rng.permutation(len(F_ctrl))[:n]]
        P = np.vstack([A, B])
        self.lab = np.r_[np.ones(n), np.zeros(n)]
        self.mu, self.sd = P.mean(0), P.std(0) + 1e-9
        self.tree, self.k = cKDTree((P - self.mu) / self.sd), k

    def score(self, F: np.ndarray) -> np.ndarray:
        _, i = self.tree.query((F - self.mu) / self.sd, k=self.k)
        return self.lab[i].mean(1)


class Coverage:
    def __init__(self, F_sim: np.ndarray, y_sim: np.ndarray, rng, k: int = 10, level: float = 0.99):
        self.mu, self.sd = F_sim.mean(0), F_sim.std(0) + 1e-9
        self.k, self.level = k, level
        self.trees, self.thr, self.lo, self.hi = {}, {}, {}, {}
        for c in SIM_CLASSES:
            Z = self._z(F_sim[y_sim == c])
            idx = rng.permutation(len(Z))
            n_cal = max(50, len(Z) // 5)
            cal, ref = Z[idx[:n_cal]], Z[idx[n_cal:]]
            self.trees[c] = cKDTree(ref)
            self.thr[c] = np.quantile(self._stat(c, cal), level)
            raw = F_sim[y_sim == c]
            self.lo[c], self.hi[c] = np.quantile(raw, 0.005, 0), np.quantile(raw, 0.995, 0)

    def _z(self, F):
        return (F - self.mu) / self.sd

    def _stat(self, c, Z):
        d, _ = self.trees[c].query(Z, k=self.k)
        return d.mean(1)

    def covered(self, F: np.ndarray, sim_cls: np.ndarray) -> np.ndarray:
        out = np.zeros(len(F), bool)
        for c in set(sim_cls.tolist()):
            i = sim_cls == c
            out[i] = self._stat(c, self._z(F[i])) <= self.thr[c]
        return out

    def marginal(self, F: np.ndarray, sim_cls: np.ndarray) -> np.ndarray:
        out = np.zeros_like(F, dtype=bool)
        for c in set(sim_cls.tolist()):
            i = sim_cls == c
            out[i] = (F[i] >= self.lo[c]) & (F[i] <= self.hi[c])
        return out


def _row(name, cov, marg, y, phys):
    fault = y != "N"
    r = {"kaynak": name, "pencere": int(len(cov)), "fizik_benzeri": 100 * (phys > 0.5).mean(),
         "kapsama": 100 * cov.mean(),
         "kapsama_N": 100 * cov[~fault].mean() if (~fault).any() else np.nan,
         "kapsama_ariza": 100 * cov[fault].mean() if fault.any() else np.nan}
    m = 100 * marg.mean(0)
    r["en_dusuk_marjinal"] = f"{FEATS[int(np.argmin(m))]} {m.min():.0f}%"
    return r, m


def run(s: dict):
    import pandas as pd
    from pilot1.config import Config

    test = s.get("test", False)
    out = os.path.join(s["out"], "uyum")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(s.get("seed", 0) + 7)
    try:
        from .finetune import sim_opts
        opts = sim_opts(s) or None
    except Exception:
        opts = None
    n_mach = 40 if test else int(s.get("uyum_makine", 2000))
    n_real = 300 if test else int(s.get("uyum_pencere", 3000))
    datasets = [d.strip().lower() for d in s.get("uyum_veri", "cwru,seu,pu,jnu,hit_mfs,hit_sb").split(",") if d.strip()]
    cfg = Config(data_root=s["data_root"], synthetic=test)

    t0 = time.time()
    print(f"simülatör: {n_mach} makine x {len(SIM_CLASSES)} sınıf ...")
    Xs, ys = simulate(n_mach, seed=12345, opts=opts)
    Fs = features(Xs)
    cov = Coverage(Fs, ys, rng)
    print(f"  {len(Xs)} pencere, {time.time() - t0:.0f} s")
    print("kontrol önseli havuzu ...")
    Fpool = features(control_windows(max(30, len(Xs) // 10), seed=4242))
    pl = PhysicsLike(Fs, Fpool, rng)

    rows, margs, feat_rows = [], [], []
    Xh, yh = simulate(max(20, n_mach // 4), seed=54321, opts=opts)     # bağımsız simülatör (referans)
    Fh = features(Xh)
    r, m = _row("Simülatör (bağımsız, referans)", cov.covered(Fh, yh), cov.marginal(Fh, yh), yh, pl.score(Fh))
    rows.append(r); margs.append(m)
    feat_rows.append(pd.DataFrame(Fh, columns=FEATS).assign(kaynak="sim", sinif=yh))

    Xc = control_windows(60 if test else 600, seed=999)                  # fiziksiz kontrol önseli
    Fc = features(Xc)
    best = np.zeros(len(Fc), bool)
    for c in SIM_CLASSES:                                                 # herhangi bir sınıfça kapsanıyor mu
        best |= cov.covered(Fc, np.full(len(Fc), c))
    rows.append({"kaynak": "Kontrol önseli (fiziksiz, referans)", "pencere": len(Fc),
                 "fizik_benzeri": 100 * (pl.score(Fc) > 0.5).mean(), "kapsama": 100 * best.mean(),
                 "kapsama_N": np.nan, "kapsama_ariza": np.nan, "en_dusuk_marjinal": "-"})
    margs.append(np.full(len(FEATS), np.nan))

    for name in datasets:
        t1 = time.time()
        try:
            Xr, yr = real_windows(name, cfg, n_real, rng)
        except Exception as e:                                            # veri yoksa atla, devam et
            print(f"  {name}: atlandı ({e})")
            continue
        keep = np.array([c in REAL_TO_SIM for c in yr])
        Xr, yr = Xr[keep], yr[keep]
        Fr = features(Xr)
        sc = np.array([REAL_TO_SIM[c] for c in yr])
        r, m = _row(name.upper(), cov.covered(Fr, sc), cov.marginal(Fr, sc), yr, pl.score(Fr))
        rows.append(r); margs.append(m)
        feat_rows.append(pd.DataFrame(Fr, columns=FEATS).assign(kaynak=name, sinif=yr))
        print(f"  {name}: {len(Xr)} pencere, fizik benzeri %{r['fizik_benzeri']:.1f}, "
              f"kapsama %{r['kapsama']:.1f} ({time.time() - t1:.0f} s)")

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "uyum_tablo.csv"), index=False)
    pd.DataFrame(margs, columns=FEATS, index=tab.kaynak).round(1).to_csv(os.path.join(out, "uyum_marjinal.csv"))
    feats = pd.concat(feat_rows, ignore_index=True)
    feats.to_csv(os.path.join(out, "uyum_ozellikler.csv"), index=False)
    Fs_df = pd.DataFrame(Fs, columns=FEATS).assign(kaynak="sim_ref", sinif=ys)
    Fs_df.sample(min(len(Fs_df), 6000), random_state=0).to_csv(os.path.join(out, "uyum_sim_ornek.csv"), index=False)

    with open(os.path.join(out, "uyum_tablo.md"), "w", encoding="utf-8") as f:
        f.write("# Simülatör–gerçek uyumu (kapsama)\n\n")
        f.write(f"Simülatör: {len(Xs)} pencere ({n_mach} makine), seçenekler={json.dumps(opts or {})}. "
                f"Fizik benzeri: fizik/kontrol eşit havuzunda k=25 komşu çoğunluğu fizik. "
                f"Ortak kapsama: k=10 en yakın komşu, %99 eşik (sınıf bazında). Değerler %.\n\n")
        f.write(tab.round(1).to_markdown(index=False))
        f.write("\n\n## Marjinal kapsama (% , simülatörün %0,5–%99,5 aralığı)\n\n")
        f.write(pd.DataFrame(margs, columns=FEATS, index=tab.kaynak).round(0).to_markdown())
        f.write("\n")
    _plot(feats, Fs_df, out)
    print(f"\nbitti ({time.time() - t0:.0f} s) -> {out}")
    print(tab.round(1).to_string(index=False))


def _plot(feats, sim, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    show = [("log_kurtosis", "Kurtosis", True), ("log_crest", "Crest factor", True),
            ("centroid_khz", "Spectral centroid (kHz)", False), ("e_0_0.5k", "Energy < 0.5 kHz", False),
            ("e_3_6k", "Energy 3–6 kHz", False), ("log_env_peak", "Envelope peak ratio", True)]
    real = feats[feats.kaynak != "sim"]
    names = list(dict.fromkeys(real.kaynak))
    fig, axes = plt.subplots(1, len(show), figsize=(7.16, 2.3), constrained_layout=True)
    for ax, (col, lab, lg) in zip(axes, show):
        tf = (lambda v: 10 ** v) if lg else (lambda v: v)
        for grp, x0, colr in (("N", -0.8, "#9b9a93"), ("F", -0.4, "#555")):
            v = tf(sim[col][(sim.sinif == "N") == (grp == "N")])
            q = np.quantile(v, [0.01, 0.25, 0.5, 0.75, 0.99])
            ax.plot([x0, x0], [q[0], q[4]], color=colr, lw=1)
            ax.plot([x0, x0], [q[1], q[3]], color=colr, lw=4, solid_capstyle="butt")
            ax.plot(x0, q[2], "_", color="white", ms=5)
        for i, n in enumerate(names):
            d = real[real.kaynak == n]
            for grp, dx, mk in (("N", -0.12, "o"), ("F", 0.12, "s")):
                v = tf(d[col][(d.sinif == "N") == (grp == "N")])
                if len(v) == 0:
                    continue
                q = np.quantile(v, [0.25, 0.5, 0.75])
                ax.errorbar(i + 0.5 + dx, q[1], yerr=[[q[1] - q[0]], [q[2] - q[1]]], fmt=mk, ms=3,
                            mfc="white" if grp == "N" else "#2a78d6", mec="#2a78d6", ecolor="#2a78d6", lw=0.8)
        ax.set_xticks([-0.6] + [i + 0.5 for i in range(len(names))])
        ax.set_xticklabels(["Sim."] + [n.upper().replace("HIT_", "HIT-") for n in names], rotation=90, fontsize=6)
        ax.set_title(lab, fontsize=7)
        ax.tick_params(labelsize=6)
        if lg:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.suptitle("grey: simulator (light = healthy, dark = faulty; 1/25/50/75/99%); "
                 "blue: real median and IQR (open = healthy, filled = faulty)", fontsize=6)
    fig.savefig(os.path.join(out, "uyum.png"), dpi=200)
    fig.savefig(os.path.join(out, "uyum.pdf"))
    plt.close(fig)
