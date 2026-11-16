"""Kontrol önseli: fizik İÇERMEYEN genel sentetik zaman serileri.

Amaç: fiziksel önselle elde edilen kazancın gerçekten FİZİKTEN mi, yoksa yalnızca
"1024 uzunlukta, z-skorlu, epizodik ek eğitimden" mi geldiğini ayırmak (makalenin kilit ablasyonu).

Epizot yapısı simülatörle birebir aynıdır:
  makine -> ortak bir temel tarif (tüm sınıflarda var)
  sınıf  -> temel tarife eklenen rastgele bir üretim bileşeni (sinüs toplamı, AR süreci, Poisson darbe, genlik modülasyonu,
            rastgele yürüyüş, cıvıltı karışımı); tarifler rulman fiziğine dayanmaz
  birim  -> tarif parametrelerinde ±%20 kayma (aynı sınıfın başka örneği)
  koşul  -> tüm frekanslarda ortak ölçek (devir değişimine karşılık gelen genel zaman ölçeği)
Böylece iki önsel arasındaki tek fark, sinyallerin fiziksel içeriğidir.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .simulator import FS, _windows, zscore

COMPONENTS = ("sines", "ar", "spikes", "am", "walk", "chirp")
DELTA_W = (0.6, 6.0)   # sınıfa özgü bileşenin ağırlık aralığı (zorluk ayarı)


def _recipe(rng):
    """Bir sınıf tarifi: 1-3 bileşenin rastgele parametreli karışımı."""
    comps = rng.choice(COMPONENTS, int(rng.integers(1, 4)), replace=False)
    r = []
    for c in comps:
        p = {"kind": str(c), "w": float(np.exp(rng.uniform(np.log(0.2), np.log(2.0))))}
        if c == "sines":
            p["f"] = np.exp(rng.uniform(np.log(5), np.log(5000), int(rng.integers(1, 4))))
            p["a"] = rng.uniform(0.2, 1, len(p["f"]))
        elif c == "ar":
            p["f0"], p["r"] = float(np.exp(rng.uniform(np.log(20), np.log(5000)))), float(rng.uniform(0.8, 0.995))
        elif c == "spikes":
            p["rate"], p["decay"] = float(np.exp(rng.uniform(np.log(5), np.log(400)))), float(rng.uniform(1e-4, 5e-3))
        elif c == "am":
            p["fc"], p["fm"] = float(np.exp(rng.uniform(np.log(100), np.log(5000)))), float(np.exp(rng.uniform(np.log(1), np.log(200))))
            p["depth"] = float(rng.uniform(0.2, 1.0))
        elif c == "walk":
            p["smooth"] = float(rng.uniform(0.9, 0.999))
        elif c == "chirp":
            p["f0"], p["f1"] = np.exp(rng.uniform(np.log(10), np.log(5000), 2))
        r.append(p)
    return r


def _jitter(recipe, rng, s=0.2):
    out = []
    for p in recipe:
        q = dict(p)
        for k, v in p.items():
            if k in ("kind",):
                continue
            if k == "r" or k == "smooth":
                q[k] = float(np.clip(v + rng.normal(0, 0.005), 0.5, 0.999))
            elif k == "depth":
                q[k] = float(np.clip(v * rng.uniform(1 - s, 1 + s), 0, 1))
            else:
                q[k] = v * rng.uniform(1 - s, 1 + s, np.shape(v)) if np.ndim(v) else v * rng.uniform(1 - s, 1 + s)
        out.append(q)
    return out


def _render(recipe, scale, n, rng, fs=FS):
    t = np.arange(n) / fs
    nyq = 0.45 * fs
    x = np.zeros(n)
    for p in recipe:
        k = p["kind"]
        if k == "sines":
            s = sum(a * np.sin(2 * np.pi * min(f * scale, nyq) * t + rng.uniform(0, 2 * np.pi))
                    for f, a in zip(p["f"], p["a"]))
        elif k == "ar":
            th = 2 * np.pi * min(p["f0"] * scale, nyq) / fs
            s = lfilter([1.0], [1, -2 * p["r"] * np.cos(th), p["r"] ** 2], rng.normal(size=n))
        elif k == "spikes":
            cnt = rng.poisson(p["rate"] * scale * n / fs)
            d = np.zeros(n)
            np.add.at(d, rng.integers(0, n, cnt), rng.normal(0, 1, cnt))
            h = np.exp(-np.arange(int(5 * p["decay"] * fs) + 1) / (p["decay"] * fs))
            s = np.convolve(d, h)[:n]
        elif k == "am":
            s = (1 + p["depth"] * np.sin(2 * np.pi * p["fm"] * scale * t)) * \
                np.sin(2 * np.pi * min(p["fc"] * scale, nyq) * t)
        elif k == "walk":
            s = lfilter([1 - p["smooth"]], [1, -p["smooth"]], rng.normal(size=n)) * 30
        else:  # chirp
            f0, f1 = min(p["f0"] * scale, nyq), min(p["f1"] * scale, nyq)
            ph = 2 * np.pi * np.cumsum(np.linspace(f0, f1, n)) / fs
            s = np.sin(ph + rng.uniform(0, 2 * np.pi))
        s = s / (np.std(s) + 1e-9)
        x += p["w"] * s
    x += rng.uniform(0.05, 0.5) * rng.normal(size=n)
    return x.astype(np.float32)


def sample_episode(rng: np.random.Generator, L: int = 1024, n_way=(2, 8), k_shot=(1, 10),
                   q_per_class=(5, 15), p_cross_cond: float = 0.8, p_cross_unit: float = 0.7, **_):
    """simulator.sample_episode ile aynı arayüz ve aynı epizot yapısı."""
    C = int(rng.integers(n_way[0], n_way[1] + 1))
    k = int(rng.integers(k_shot[0], k_shot[1] + 1))
    q = int(rng.integers(q_per_class[0], q_per_class[1] + 1))
    n_cond = int(rng.integers(2, 5))
    scales = [1.0] + list(np.exp(rng.uniform(np.log(0.6), np.log(1.5), n_cond - 1)))
    cond_s = int(rng.integers(n_cond))
    cross_cond = rng.random() < p_cross_cond
    cond_q = int(rng.choice([i for i in range(n_cond) if i != cond_s])) if cross_cond else cond_s
    cross_unit = rng.random() < p_cross_unit
    perm = rng.permutation(C)

    # Fizik önseliyle eşit zorluk için: sınıflar ortak bir "makine" tarifini paylaşır, her sınıf buna
    # kendine özgü bir bileşen ekler (bir sınıf eklemesiz kalabilir: "sağlam" benzeri). Rulman fiziği yok.
    base = _recipe(rng)
    Xs, ys, Xq, yq = [], [], [], []
    for ci in range(C):
        if ci == 0 and rng.random() < 0.85:
            rec = [dict(p) for p in base]
        else:
            extra = _recipe(rng)[:1]
            extra[0]["w"] = float(np.exp(rng.uniform(np.log(DELTA_W[0]), np.log(DELTA_W[1]))))
            rec = [dict(p) for p in base] + extra
        u_s = _jitter(rec, rng)
        Xs.append(_windows(_render(u_s, scales[cond_s], k * L, rng), L, k)); ys += [perm[ci]] * k
        if cross_unit:
            n_u = int(rng.integers(1, 4))
            sizes = np.diff(np.linspace(0, q, n_u + 1).astype(int))
            for sz in sizes[sizes > 0]:
                Xq.append(_windows(_render(_jitter(rec, rng), scales[cond_q], sz * L, rng), L, sz))
        else:
            Xq.append(_windows(_render(u_s, scales[cond_q], q * L, rng), L, q))
        yq += [perm[ci]] * q
    X = zscore(np.concatenate(Xs + Xq))
    y = np.asarray(ys + yq, dtype=np.int64)
    return X, y, len(ys), {"k": k, "q": q, "C": C, "cross_cond": cross_cond, "cross_unit": cross_unit}
