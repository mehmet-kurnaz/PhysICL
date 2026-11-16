"""Simülatör v0: rastgeleleştirilmiş fiziksel titreşim önseli.

Her çağrıda yeni bir "sanal makine" üretilir. Makine; rulman geometrisi, yapısal rezonanslar,
sensör bant genişliği, mil harmonikleri, dişli/şebeke tonları ve gürültü rengini belirler.
Aynı makinenin içinde:
  - birim (unit)  : aynı sınıftan FARKLI bir parça örneği (ör. başka bir hasarlı rulman).
                    Hasar şiddeti, darbe dağınıklığı, montaj kaynaklı rezonans kayması birim başına değişir.
                    Pilot 1'deki PU bulgusu (aynı rulman %98, farklı rulman %40) bu katmanın nedenidir.
  - koşul (cond)  : çalışma noktası (devir ve yük). CWRU yük, SEU devir değişimine karşılık gelir.
Model, bu rahatsız edici değişkenlere rağmen arıza imzasını ayırt etmeyi öğrenmelidir.

Sınıflar:
  N      sağlam (bazı birimlerde periyodik olmayan hafif pürüzlülük darbeleri)
  OR     dış bilezik: BPFO'da darbeler, yük bölgesinde sabit genlik
  IR     iç bilezik: BPFI'da darbeler, mil devriyle yük bölgesi modülasyonu
  B      bilye: BSF veya 2xBSF'de darbeler, kafes (FTF) modülasyonu, aralıklı temas
  OR+IR  birleşik bilezik hasarı
  UNB    dengesizlik: 1x harmonik baskın
  MIS    eksen kaçıklığı: 2x (ve 3x, 4x) harmonikler yükselir
  LOOSE  mekanik gevşeklik: çok sayıda harmonik + alt harmonikler, kırpılmış dalga, devir başına darbe

Tüm çıktı cfg.fs = 12 kHz'tedir (pilot 1 ile aynı ortak temsil).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

FS = 12_000
CLASSES = ("N", "OR", "IR", "B", "OR+IR", "UNB", "MIS", "LOOSE")
BEARING_PARTS = {"OR": ("OR",), "IR": ("IR",), "B": ("B",), "OR+IR": ("OR", "IR")}


def _logu(rng, lo, hi, size=None):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size))


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


# ----------------------------------------------------------------------------- makine

@dataclass
class Machine:
    # rulman geometrisi -> devire oranlı arıza frekansları (order)
    n_balls: int
    d_ratio: float           # bilye çapı / hatve çapı
    cos_phi: float           # temas açısının kosinüsü
    # yapısal yol
    modes: np.ndarray        # (M, 3): frekans Hz, sönüm oranı, kazanç
    fc: float                # sensör/kayıt bant sınırı (Hz)
    # çalışma koşulları: (devir Hz, yük çarpanı)
    conditions: list
    fluct_a: float           # devir dalgalanma genliği (oran)
    fluct_f: float           # devir dalgalanma frekansı (Hz)
    # arka plan
    shaft_amps: np.ndarray   # sağlam makinede 1x..10x harmonik genlikleri
    gear_z: int | None       # dişli diş sayısı (yoksa None)
    gear_side: np.ndarray    # yan bant oranları (±1, ±2)
    line_f: float | None     # şebeke tonu (50/60 Hz) varsa
    noise_alpha: float       # renkli gürültü üssü (1/f^alpha)
    w: dict = field(default_factory=dict)   # bileşen ağırlıkları (RMS)
    resonant: bool = False   # v1: rezonans baskın, yüksek SNR'li rejim (CWRU tipi)

    def orders(self) -> dict:
        r, n, c = self.d_ratio, self.n_balls, self.cos_phi
        return {"BPFO": n / 2 * (1 - r * c), "BPFI": n / 2 * (1 + r * c),
                "BSF": (1 - (r * c) ** 2) / (2 * r), "FTF": 0.5 * (1 - r * c)}


def sample_machine(rng: np.random.Generator, opts: dict | None = None) -> Machine:
    """opts (ablasyon/sürüm; None = v0, tüm katmanlar açık):
      bant=False  : sensör bant sınırı rastgeleleştirmesi kapalı (hep 5,5 kHz)
      golge=False : dişli ve şebeke tonları kapalı
      surum="v1"  : makinelerin ~%30'u rezonans baskın, yüksek SNR'li rejimde (CWRU tipi)
    opts verilmediğinde rastgele sayı dizisi v0 ile birebir aynıdır."""
    o = opts or {}
    # Sensör bant genişliği: özgün örnekleme hızı rastgele (SEU 5,12 kHz -> ~2,3 kHz sınır)
    fs_orig = rng.choice([5120, 8192, 12000, 25600, 48000, 64000], p=[.12, .05, .25, .2, .18, .2])
    fc = min(0.45 * fs_orig, 5500.0) if o.get("bant", True) else 5500.0
    n_modes = rng.integers(1, 5)
    f = np.sort(_logu(rng, 1200, 5800, n_modes))
    if f[0] > 0.9 * fc:                       # en az bir rezonans bant içinde kalsın
        f[0] = rng.uniform(min(800, 0.5 * fc), 0.9 * fc)
    modes = np.stack([f, rng.uniform(0.02, 0.12, n_modes), _logu(rng, 0.3, 1.0, n_modes)], 1)

    base = rng.uniform(10, 50)                # mil devri 600-3000 rpm
    n_cond = rng.integers(2, 5)
    if rng.random() < 0.5:                    # yük değişimi: devir az değişir (CWRU benzeri)
        conds = [(base * (1 - 0.01 * i * rng.uniform(0.5, 1.5)), 1 + 0.3 * i * rng.uniform(0.5, 1.5))
                 for i in range(n_cond)]
    else:                                     # devir değişimi (SEU benzeri)
        conds = [(base * (rng.uniform(0.6, 1.5) if i else 1.0), _logu(rng, 0.7, 1.5)) for i in range(n_cond)]

    shaft = np.zeros(10)
    shaft[0] = 1.0
    decay = rng.uniform(0.1, 0.6)
    for k in range(1, 5):
        shaft[k] = decay ** k * _logu(rng, 0.5, 2.0)

    gear = int(rng.integers(12, 61)) if rng.random() < 0.6 else None
    line = float(rng.choice([50.0, 60.0])) if rng.random() < 0.3 else None
    if not o.get("golge", True):
        gear, line = None, None
    w = {"shaft": _logu(rng, 0.02, 3.0), "gear": _logu(rng, 0.02, 2.0) if gear else 0.0,
         "line": _logu(rng, 0.02, 0.5) if line else 0.0, "struct": _logu(rng, 0.2, 2.0),
         "colored": _logu(rng, 0.02, 1.0), "white": _logu(rng, 0.02, 0.5)}
    m = Machine(n_balls=int(rng.integers(7, 17)), d_ratio=rng.uniform(0.15, 0.35),
                cos_phi=np.cos(np.deg2rad(rng.uniform(0, 20))), modes=modes, fc=fc,
                conditions=conds, fluct_a=rng.uniform(0, 0.01), fluct_f=rng.uniform(0.2, 2.0),
                shaft_amps=shaft, gear_z=gear, gear_side=rng.uniform(0, 0.4, 2), line_f=line,
                noise_alpha=rng.uniform(0, 1.2), w=w)
    if o.get("surum", "v0") == "v1" and rng.random() < 0.3:
        # CWRU tipi: geniş bant, yüksek frekanslı baskın rezonans, zayıf düşük frekans içeriği
        m.resonant = True
        m.fc = 5500.0
        k = int(rng.integers(1, 3))
        m.modes = np.stack([np.sort(_logu(rng, 2500, 5400, k)), rng.uniform(0.03, 0.1, k),
                            _logu(rng, 0.5, 1.0, k)], 1)
        for key in ("shaft", "gear", "colored", "line"):
            m.w[key] *= _logu(rng, 0.02, 0.3)
    return m


# ----------------------------------------------------------------------------- birim

@dataclass
class Unit:
    cls: str
    modes: np.ndarray        # montaj kaynaklı kaymış rezonanslar
    shaft_amps: np.ndarray   # sınıfa göre değiştirilmiş harmonikler (1x..10x)
    subharm: np.ndarray      # 0.5x, 1.5x, 2.5x genlikleri (gevşeklik)
    clip_q: float | None     # gevşeklikte kırpma yüzdesi
    snr_db: float            # arıza bileşeninin arka plana oranı
    amp_sigma: float         # darbe genliği log-normal dağınıklığı
    slip: float              # darbe zamanlama sapması (periyodun oranı)
    spall: tuple | None      # (gecikme s, oran): giriş/çıkış çift darbesi
    load_eps: float          # yük bölgesi genişliği
    psi: float               # hasarın açısal konumu
    ball_mult: int           # B: 1 veya 2 x BSF
    ball_dropout: float      # B: aralıklı temas olasılığı
    rough_rate: float        # N: pürüzlülük darbe hızı (0 = yok)
    rough_db: float | None = None   # v1: pürüzlülük seviyesi birim başına sabit


def sample_unit(m: Machine, cls: str, rng: np.random.Generator, opts: dict | None = None) -> Unit:
    o = opts or {}
    modes = m.modes.copy()
    modes[:, 0] *= rng.uniform(0.93, 1.07, len(modes))
    modes[:, 2] *= _logu(rng, 0.7, 1.4, len(modes))

    amps = m.shaft_amps.copy()
    amps[0] *= _logu(rng, 0.7, 1.4)           # kalıntı dengesizlik birimden birime değişir
    sub, clip_q = np.zeros(3), None
    if cls == "UNB":
        amps[0] *= rng.uniform(2.5, 8.0)
        amps[1] *= rng.uniform(1.0, 1.5)
    elif cls == "MIS":
        amps[1] = amps[0] * rng.uniform(0.8, 3.0)
        amps[2] = amps[0] * rng.uniform(0.3, 1.5)
        amps[3] = max(amps[3], amps[0] * rng.uniform(0.1, 0.6))
    elif cls == "LOOSE":
        p = rng.uniform(0.3, 1.0)
        amps = amps[0] * rng.uniform(0.3, 1.0, 10) * np.arange(1, 11) ** -p
        amps[0] = m.shaft_amps[0] * rng.uniform(1.0, 2.0)
        sub = amps[0] * rng.uniform(0.1, 0.5, 3)
        clip_q = rng.uniform(70, 95)

    bearing = cls in BEARING_PARTS
    snr = (rng.uniform(-15, 5) if cls == "B" else rng.uniform(-10, 15)) if bearing else (rng.uniform(-12, 0) if cls == "LOOSE" else -np.inf)
    u = Unit(cls=cls, modes=modes, shaft_amps=amps, subharm=sub, clip_q=clip_q, snr_db=snr,
                amp_sigma=rng.uniform(0.1, 0.6), slip=rng.uniform(0.005, 0.02),
                spall=(rng.uniform(3e-4, 1.5e-3), rng.uniform(0.3, 1.0)) if rng.random() < 0.4 else None,
                load_eps=rng.uniform(0.3, 1.2), psi=rng.uniform(0, 1), ball_mult=int(rng.choice([1, 2])),
                ball_dropout=rng.uniform(0, 0.5),
                rough_rate=_logu(rng, 100, 1000) if (cls == "N" and rng.random() < 0.35) else 0.0)
    if o.get("surum", "v0") == "v1":
        if m.resonant and cls in BEARING_PARTS:          # CWRU tipi: hasar sinyali baskın
            u.snr_db = rng.uniform(0, 15) if cls == "B" else rng.uniform(5, 25)
        if cls == "N":                                     # sağlam sınıfta daha fazla çeşitlilik (PU)
            if rng.random() < 0.6:
                u.rough_rate = _logu(rng, 50, 2000)
                u.rough_db = rng.uniform(-20, 0)
    return u


# ----------------------------------------------------------------------------- sinyal

def impulse_response(modes: np.ndarray, fs: int = FS) -> np.ndarray:
    f, z, g = modes.T
    t_end = min(0.04, float(np.max(6 / (z * 2 * np.pi * f))))
    t = np.arange(int(t_end * fs)) / fs
    h = (g[:, None] * np.exp(-z[:, None] * 2 * np.pi * f[:, None] * t)
         * np.sin(2 * np.pi * f[:, None] * np.sqrt(1 - z[:, None] ** 2) * t)).sum(0)
    return h / (np.linalg.norm(h) + 1e-12)


def _impacts(cyc_fault, amps_fn, u: Unit, n: int, rng, fs=FS) -> np.ndarray:
    """cyc_fault: arıza frekansında birikmiş devir sayısı (monoton). Her tam sayıda bir darbe."""
    k = np.arange(np.ceil(cyc_fault[0]), np.floor(cyc_fault[-1]))
    if len(k) == 0:
        return np.zeros(n)
    k = k + rng.normal(0, u.slip, len(k))
    idx = np.searchsorted(cyc_fault, k).clip(0, n - 1)
    a = amps_fn(idx) * np.exp(rng.normal(0, u.amp_sigma, len(idx)))
    d = np.zeros(n)
    np.add.at(d, idx, a)
    if u.spall is not None:                   # hasar girişi/çıkışı: ikinci darbe
        idx2 = (idx + int(u.spall[0] * fs)).clip(0, n - 1)
        np.add.at(d, idx2, u.spall[1] * a)
    return d


def _colored(n, alpha, rng):
    X = np.fft.rfft(rng.normal(size=n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    return np.fft.irfft(X / f ** (alpha / 2), n)


def render(m: Machine, u: Unit, cond: int, n: int, rng: np.random.Generator, fs: int = FS) -> np.ndarray:
    """Bir birimin bir koşuldaki n örneklik kaydı."""
    pad = int(0.1 * fs)
    N = n + pad
    t = np.arange(N) / fs
    fr0, load = m.conditions[cond]
    walk = np.cumsum(rng.normal(0, 1, N)); walk = 0.002 * (walk - walk.mean()) / (np.abs(walk).max() + 1e-9)
    fr = fr0 * (1 + m.fluct_a * np.sin(2 * np.pi * m.fluct_f * t + rng.uniform(0, 2 * np.pi)) + walk)
    cyc = np.cumsum(fr) / fs + rng.uniform()  # mil devri sayısı
    h = impulse_response(u.modes, fs)

    # --- arka plan
    healthy_rms = np.sqrt(np.sum(m.shaft_amps ** 2) / 2)
    ph = rng.uniform(0, 2 * np.pi, 13)
    shaft = sum(a * np.sin(2 * np.pi * (k + 1) * cyc + ph[k]) for k, a in enumerate(u.shaft_amps) if a > 0)
    for j, (o, a) in enumerate(zip((0.5, 1.5, 2.5), u.subharm)):
        if a > 0:
            shaft = shaft + a * np.sin(2 * np.pi * o * cyc + ph[10 + j])
    if u.clip_q is not None:
        c = np.percentile(np.abs(shaft), u.clip_q)
        shaft = np.clip(shaft, -c, c)
    bg = m.w["shaft"] * load ** 0.5 * shaft / healthy_rms
    if m.gear_z:
        sb = [(0, 1.0), (1, m.gear_side[0]), (-1, m.gear_side[0]), (2, m.gear_side[1]), (-2, m.gear_side[1])]
        g = sum(a * np.sin(2 * np.pi * (m.gear_z + s) * cyc + rng.uniform(0, 2 * np.pi)) for s, a in sb)
        bg = bg + m.w["gear"] * load ** 0.5 * g / _rms(g)
    if m.line_f:
        bg = bg + m.w["line"] * np.sqrt(2) * np.sin(2 * np.pi * m.line_f * t + rng.uniform(0, 2 * np.pi))
    s = fftconvolve(rng.normal(size=N), h)[:N]
    bg = bg + m.w["struct"] * s / _rms(s)
    c_ = _colored(N, m.noise_alpha, rng)
    bg = bg + m.w["colored"] * c_ / _rms(c_) + m.w["white"] * rng.normal(size=N)
    bg = bg / _rms(bg)

    # --- arıza bileşeni
    fault = np.zeros(N)
    orders = m.orders()
    shaft_ang = 2 * np.pi * (cyc + u.psi)
    if u.cls in BEARING_PARTS:
        for part in BEARING_PARTS[u.cls]:
            if part == "OR":
                o, mod = orders["BPFO"], lambda i: np.ones(len(i))
            elif part == "IR":
                o = orders["BPFI"]
                mod = lambda i: 0.1 + 0.9 * np.clip(1 - (1 - np.cos(shaft_ang[i])) / (2 * u.load_eps), 0, 1) ** 1.5
            else:
                o = orders["BSF"] * u.ball_mult
                cage = 2 * np.pi * (orders["FTF"] * cyc + u.psi)
                mod = lambda i: ((0.1 + 0.9 * np.clip(1 - (1 - np.cos(cage[i])) / (2 * u.load_eps), 0, 1) ** 1.5)
                                 * (rng.random(len(i)) > u.ball_dropout))
            d = _impacts(o * cyc, mod, u, N, rng, fs)
            part_sig = fftconvolve(d, h)[:N]
            fault = fault + part_sig / _rms(part_sig) / np.sqrt(len(BEARING_PARTS[u.cls]))
        # devir ve yük arttıkça darbe sayısı ve enerjisi artar
        fault *= 10 ** (u.snr_db / 20) * np.sqrt(fr0 / m.conditions[0][0]) * load ** 0.5
    elif u.cls == "LOOSE":                    # devir başına, dağınık zamanlı darbeler
        uu = Unit(**{**u.__dict__, "slip": 0.05, "spall": None})
        d = _impacts(cyc, lambda i: np.ones(len(i)), uu, N, rng, fs)
        sig = fftconvolve(d, h)[:N]
        fault = 10 ** (u.snr_db / 20) * sig / _rms(sig)
    elif u.rough_rate > 0:                    # sağlam ama pürüzlü: Poisson zamanlı küçük darbeler
        k = rng.poisson(u.rough_rate * N / fs)
        d = np.zeros(N)
        np.add.at(d, rng.integers(0, N, k), np.exp(rng.normal(0, 0.5, k)))
        sig = fftconvolve(d, h)[:N]
        lvl = u.rough_db if u.rough_db is not None else rng.uniform(-15, -3)
        fault = 10 ** (lvl / 20) * sig / _rms(sig)

    x = bg + fault
    sos = butter(6, [5, m.fc], btype="bandpass", fs=fs, output="sos")
    x = sosfilt(sos, x)
    return x[pad:].astype(np.float32)


# ----------------------------------------------------------------------------- epizot (ince ayar için)

def zscore(X):
    return ((X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-8)).astype(np.float32)


def _windows(sig, L, k):
    return sig[: k * L].reshape(k, L)


def sample_episode(rng: np.random.Generator, L: int = 1024, n_way=(2, 8), k_shot=(1, 10),
                   q_per_class=(5, 15), p_cross_cond: float = 0.8, p_cross_unit: float = 0.7,
                   classes=CLASSES, opts: dict | None = None):
    """TimEE biçiminde bir epizot: önce destek, sonra sorgu pencereleri.

    Dönüş: X (N, L) z-skorlu, y (N,) 0..C-1 etiketleri (rastgele permütasyon), n_support, bilgi.
    """
    o = opts or {}
    if not o.get("birim", True):            # ablasyon: sorgu hep destekle aynı birimden
        p_cross_unit = 0.0
    if not o.get("kosul", True):            # ablasyon: sorgu hep destekle aynı koşulda
        p_cross_cond = 0.0
    if not o.get("mil", True):              # ablasyon: yalnız rulman sınıfları
        classes = tuple(c for c in classes if c in ("N",) + tuple(BEARING_PARTS))
    m = sample_machine(rng, o)
    C = int(rng.integers(n_way[0], min(n_way[1], len(classes)) + 1))
    others = [c for c in classes if c != "N"]
    if "N" in classes and (C > len(others) or rng.random() < 0.85):   # C sabit kalsın (yığın şekli)
        chosen = ["N"] + [str(c) for c in rng.choice(others, C - 1, replace=False)]
    else:
        chosen = [str(c) for c in rng.choice(others, C, replace=False)]
    k = int(rng.integers(k_shot[0], k_shot[1] + 1))
    q = int(rng.integers(q_per_class[0], q_per_class[1] + 1))
    cond_s = int(rng.integers(len(m.conditions)))
    cross_cond = rng.random() < p_cross_cond
    cond_q = int(rng.choice([i for i in range(len(m.conditions)) if i != cond_s])) if cross_cond else cond_s
    cross_unit = rng.random() < p_cross_unit
    perm = rng.permutation(C)

    Xs, ys, Xq, yq = [], [], [], []
    for ci, cls in enumerate(chosen):
        u_s = sample_unit(m, cls, rng, o)
        Xs.append(_windows(render(m, u_s, cond_s, k * L, rng), L, k)); ys += [perm[ci]] * k
        if cross_unit:
            n_u = int(rng.integers(1, 4))
            sizes = np.diff(np.linspace(0, q, n_u + 1).astype(int))
            for sz in sizes[sizes > 0]:
                u_q = sample_unit(m, cls, rng, o)
                Xq.append(_windows(render(m, u_q, cond_q, sz * L, rng), L, sz))
        else:                                 # aynı birim, ayrı kayıt
            Xq.append(_windows(render(m, u_s, cond_q, q * L, rng), L, q))
        yq += [perm[ci]] * q
    X = zscore(np.concatenate(Xs + Xq))
    y = np.asarray(ys + yq, dtype=np.int64)
    info = {"classes": chosen, "k": k, "q": q, "cross_cond": cross_cond, "cross_unit": cross_unit,
            "fc": m.fc, "speeds": [c[0] for c in m.conditions]}
    return X, y, len(ys), info


# ----------------------------------------------------------------------------- pilot 1 uyumlu veri seti

def make_dataset(rng: np.random.Generator, classes=("N", "OR", "IR", "B"), split: str = "condition",
                 n_units: int = 3, n_windows: int = 60, L: int = 1024, machine: Machine | None = None):
    """Pilot 1'in EpisodeSampler'ına verilebilecek Recording listesi.

    split="condition": her birim her koşulda kaydedilir; grup = koşul (CWRU/SEU benzeri)
    split="unit"     : tek koşul, her sınıfın birden çok birimi; grup = birim (PU benzeri)
    """
    from pilot1.data import Recording

    m = machine or sample_machine(rng)
    recs = []
    for cls in classes:
        for j in range(n_units):
            u = sample_unit(m, cls, rng)
            conds = range(len(m.conditions)) if split == "condition" else [0]
            for c in conds:
                grp = f"c{c}" if split == "condition" else f"{cls}-u{j}"
                recs.append(Recording("sim", cls, grp, f"{cls}_u{j}_c{c}",
                                      render(m, u, c, n_windows * L, rng)))
    return m, recs
