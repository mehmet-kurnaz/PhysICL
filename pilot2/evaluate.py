"""Pilot 2 / v1 değerlendirme: pilot 1'in BİREBİR AYNI epizotlarında.

Pilot 1'in ana döngüsü (pilot1.run.run) aynen kullanılır; yalnızca yöntem listesine TimEE
varyantları eklenir. Epizot tohumu veri setine bağlı olduğundan destek/sorgu pencereleri pilot 1 ile
aynıdır; taban çizgileri pilot 1'in episodes.csv dosyasından okunur ve epizot epizot eşleştirilir.

Ayarlar:
  "modeller"   : "hazir,fizik_v1,karma_v1,genel_v1"
                 hazir        -> resmi TimEE ağırlığı (ince ayarsız)
                 X            -> results_pilot2/ince_ayar_X/son
                 X@2000       -> results_pilot2/ince_ayar_X/adim_02000
                 X@*          -> o eğitimin kayıtlı tüm adımları (analiz; model seçimi değil)
  "gorunumler" : "ham,zarf,ham+zarf"  (bkz. views.py)
Yöntem adı: "<model>|<görünüm>", ör. "karma_v1|ham+zarf". "hazir|ham" pilot 1'deki timee_raw ile aynıdır.
"""

from __future__ import annotations

import dataclasses
import glob
import os
import re

import numpy as np
import pandas as pd

BASELINES = ("timee_raw", "mantisv2_lr", "env_proto")
BASE_LABELS = {"timee_raw": "TimEE hazır · ham", "mantisv2_lr": "MantisV2 + LR",
               "env_proto": "Zarf spektrumu + prototip"}
INK, MUTED, GRID = "#2b2b2a", "#76756f", "#e4e3de"
C_BASE, C_OURS = "#a3a29a", "#6250d6"


def parse_models(spec: str, root: str, test: bool) -> dict:
    """{model_adı: ağırlık yolu}"""
    out = {}
    for item in [x.strip() for x in spec.split(",") if x.strip()]:
        if item == "hazir":
            if test:   # akış testinde resmi ağırlık indirilmez; herhangi bir yerel ağırlık kullanılır
                loc = sorted(glob.glob(os.path.join(root, "ince_ayar_*", "son")))
                if loc:
                    out["hazir"] = os.path.abspath(loc[0])
            else:
                out["hazir"] = "liamsbhoo/timee"
            continue
        tag, _, step = item.partition("@")
        base = os.path.join(root, f"ince_ayar_{tag}")
        if step == "*":
            dirs = sorted(glob.glob(os.path.join(base, "adim_*")))
        elif step:
            dirs = [os.path.join(base, f"adim_{int(step):05d}")]
        else:
            dirs = [os.path.join(base, "son")]
        for d in dirs:
            if not os.path.exists(os.path.join(d, "model.safetensors")):
                print(f"uyarı: {d} yok, atlanıyor")
                continue
            b = os.path.basename(d)
            out[tag if b == "son" else f"{tag}@{int(b.split('_')[1])}"] = os.path.abspath(d)
    return out


def label(name: str) -> str:
    if name in BASE_LABELS:
        return BASE_LABELS[name]
    model, _, view = name.partition("|")
    tag, _, step = model.partition("@")
    m = "TimEE hazır" if tag == "hazir" else f"TimEE + {tag}"
    return f"{m}{f' ({step} adım)' if step else ''} · {view}"


def run(s: dict):
    import torch  # noqa: F401  (Windows: TF'den önce)
    from pilot1 import run as p1
    from pilot1.config import Config
    from pilot2.views import VIEWS, TimeeViewMethod

    models = parse_models(s["modeller"], s["out"], s["test"])
    views = [v.strip() for v in s["gorunumler"].split(",") if v.strip()]
    for v in views:
        if v not in VIEWS:
            raise SystemExit(f"bilinmeyen görünüm: {v} ({', '.join(VIEWS)})")
    methods = {f"{m}|{v}": path for m, path in models.items() for v in views if not (m == "hazir" and v == "ham")}
    if not methods:
        raise SystemExit("değerlendirilecek yöntem yok")
    print("değerlendirilecek:", ", ".join(methods))

    orig = p1.build_methods

    def build_methods(cfg, names):
        base = [n for n in names if n not in methods]
        out = orig(cfg, base) if base else {}
        for n, path in methods.items():
            out[n] = TimeeViewMethod(dataclasses.replace(cfg, random_init_fm=False), path, n.partition("|")[2])
        return out

    p1.build_methods = build_methods
    out = os.path.join(s["out"], s.get("degerlendirme_klasoru", "degerlendirme"))
    datasets = tuple(d.strip().lower() for d in s["datasets"].split(",") if d.strip())
    cfg = Config(data_root=s["data_root"], out_dir=out, datasets=datasets, methods=tuple(methods),
                 synthetic=s["test"], episodes=2 if s["test"] else int(s.get("epizot", 10)),
                 seed=int(s.get("degerlendirme_seed", 0)),
                 queries_per_class=20 if s["test"] else Config.queries_per_class,
                 redo=tuple(methods))         # ağırlık değişmiş olabilir: her seferinde yeniden hesapla
    try:
        df = p1.run(cfg)
    finally:
        p1.build_methods = orig                  # yamayı geri al (taban klasörlerine fazladan satır yazılmasın)
    compare(df[df.method.isin(list(methods))], s["pilot1_sonuc"], out, test=s["test"])


# ----------------------------------------------------------------------------- karşılaştırma

def _paired(g, a, b):
    from scipy.stats import wilcoxon
    if a not in g or b not in g:
        return "—"
    d = (g[a] - g[b]).dropna() * 100
    try:
        p = wilcoxon(d).pvalue if len(d) > 1 and np.any(d != 0) else 1.0
    except ValueError:
        p = float("nan")
    star = "**" if p < 0.05 else ""
    return f"{star}{d.mean():+.1f}{star}, p={p:.3f}"


def _twin(name: str):
    """fizik/karma yöntemiyle aynı adım ve görünümdeki genel (kontrol) yöntemi."""
    model, _, view = name.partition("|")
    if not re.search(r"fizik|karma", model):
        return None
    return re.sub(r"fizik|karma", "genel", model, count=1) + "|" + view


def compare(df_ft: pd.DataFrame, pilot1_csv: str, out: str, test: bool = False):
    if os.path.exists(pilot1_csv):
        base = pd.read_csv(pilot1_csv)
        df = pd.concat([base[base.method.isin(BASELINES)], df_ft])
    else:
        print(f"uyarı: pilot 1 sonucu bulunamadı ({pilot1_csv}); yalnızca yeni yöntemler raporlanır")
        df = df_ft
    df = df.drop_duplicates(["dataset", "episode", "shot", "method"], keep="last")
    ours = list(dict.fromkeys(df_ft.method))
    methods = [m for m in BASELINES if m in set(df.method)] + ours
    datasets = list(dict.fromkeys(df.dataset))
    shots = sorted(df.shot.unique())

    lines = ["# v1 değerlendirme: pilot 1 epizotlarında doğruluk (%)\n",
             f"Ortalama ± %95 GA, {df.episode.nunique()} eşleştirilmiş epizot. Fark satırları: epizot başına eşleştirilmiş "
             "ortalama fark, Wilcoxon p; **kalın** = p < 0,05. "
             "\"en iyi taban\": o hücrede ortalaması en yüksek taban çizgisi (TimEE hazır·ham, MantisV2, zarf spektrumu).\n"]
    for d in datasets:
        sub = df[df.dataset == d]
        piv = {k: sub[sub.shot == k].pivot_table(index="episode", columns="method", values="acc") for k in shots}
        best = {k: max([b for b in BASELINES if b in piv[k]], key=lambda b: piv[k][b].mean(), default=None)
                for k in shots}
        lines += [f"\n## {d.upper()}\n", "| yöntem | " + " | ".join(f"{k}-shot" for k in shots) + " |",
                  "|---|" + "---|" * len(shots)]
        for m in methods:
            cells = []
            for k in shots:
                v = sub[(sub.shot == k) & (sub.method == m)]["acc"].to_numpy() * 100
                half = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
                cells.append(f"{v.mean():.1f} ± {half:.1f}" if len(v) else "—")
            lines.append(f"| {label(m)} | " + " | ".join(cells) + " |")
        lines.append("| **farklar** | | | |")
        for m in ours:
            lines.append(f"| *{label(m)} − en iyi taban* | " + " | ".join(
                (_paired(piv[k], m, best[k]) + f" ({best[k]})") if best[k] else "—" for k in shots) + " |")
        for m in ours:
            t = _twin(m)
            if t and t in ours:
                lines.append(f"| *KONTROL: {label(m)} − {label(t)}* | " +
                             " | ".join(_paired(piv[k], m, t) for k in shots) + " |")
    lines += primary_analysis(df)
    md = "\n".join(lines)
    with open(os.path.join(out, "karsilastirma.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)
    _plot(df, methods, datasets, shots, os.path.join(out, "karsilastirma.png"))
    print(f"\nKarşılaştırma: {os.path.abspath(os.path.join(out, 'karsilastirma.md'))}")


def _plot(df, methods, datasets, shots, path):
    """Satır = shot, sütun = veri seti; yatay çubuk = yöntem (etiket eksende, renk yalnız taban/bizim)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(methods)
    fig, axes = plt.subplots(len(shots), len(datasets), figsize=(3.6 * len(datasets) + 2.6, 0.32 * n * len(shots) + 1.2),
                             sharey=True, squeeze=False)
    for i, k in enumerate(shots):
        for j, d in enumerate(datasets):
            ax = axes[i][j]
            sub = df[(df.dataset == d) & (df.shot == k)]
            for r, m in enumerate(methods):
                v = sub[sub.method == m]["acc"].to_numpy() * 100
                if not len(v):
                    continue
                half = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
                ax.barh(r, v.mean(), height=0.62, color=C_BASE if m in BASELINES else C_OURS)
                ax.errorbar(v.mean(), r, xerr=half, color=INK, lw=0.8, capsize=2)
                ax.text(min(v.mean() + half + 1.5, 97), r, f"{v.mean():.0f}", va="center", fontsize=7, color=INK)
            chance = 100 / sub["n_classes"].max() if len(sub) else None
            if chance:
                ax.axvline(chance, color=MUTED, lw=0.8, ls=":")
            ax.set_xlim(0, 100)
            ax.invert_yaxis() if (i == 0 and j == 0) else None
            ax.grid(axis="x", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(d.upper(), fontsize=10, color=INK)
            if j == len(datasets) - 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(f"{k}-shot", rotation=270, labelpad=12, color=INK)
            if i == len(shots) - 1:
                ax.set_xlabel("doğruluk (%)", color=INK)
    axes[0][0].set_yticks(range(n), [label(m) for m in methods], fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- önkayıtlı ana analiz

PRIMARY, CONTROL = "fizik_v1|ham+zarf", "genel_v1|ham+zarf"


def _holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def primary_analysis(df: pd.DataFrame) -> list:
    """pilot2/ONKAYIT.md'deki üç hipotez. Ana yöntem ve karşılaştırmalar veriden ÖNCE sabitlendi."""
    from scipy import stats
    from scipy.stats import wilcoxon

    if PRIMARY not in set(df.method):
        return []
    out = ["\n## ÖNKAYITLI ANA ANALİZ (pilot2/ONKAYIT.md)\n",
           f"Ana yöntem: {label(PRIMARY)}. Fark = ana yöntem − karşılaştırılan, epizot başına eşleştirilmiş.\n"]

    def diffs(d, k, ref):
        g = df[(df.dataset == d) & (df.shot == k)].pivot_table(index="episode", columns="method", values="acc")
        if ref == "best":
            cands = [b for b in BASELINES if b in g]
            ref = max(cands, key=lambda b: g[b].mean())
        return (g[PRIMARY] - g[ref]).dropna().to_numpy() * 100, ref

    def test_block(title, ref):
        rows, ps = [], []
        for k in (5, 10):
            if "pu" not in set(df.dataset):
                continue
            dd, r = diffs("pu", k, ref)
            p = wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
            rows.append((k, r, dd.mean(), p)); ps.append(p)
        if not rows:
            return
        adj = _holm(np.array(ps))
        out.extend([f"**{title}**\n", "| PU | karşılaştırılan | ort. fark | p | Holm p | sonuç |", "|---|---|---|---|---|---|"])
        for (k, r, m, p), pa in zip(rows, adj):
            ok = "DESTEKLENDİ" if (pa < 0.05 and m > 0) else "desteklenmedi"
            out.append(f"| {k}-shot | {r} | {m:+.1f} | {p:.4f} | {pa:.4f} | {ok} |")
        out.append("")

    test_block("H1 — PU'da en iyi taban çizgisini geçer", "best")
    if CONTROL in set(df.method):
        test_block("H2 — PU'da eşit zorluktaki genel önselden iyidir (kazanç fizikten)", CONTROL)

    out.extend(["**H3 — CWRU ve SEU'da zarar vermez** (eşik: fark ortalamasının %95 GA alt sınırı > −3 puan)\n",
                "| veri | shot | karşılaştırılan | ort. fark | %95 GA alt sınır | sonuç |", "|---|---|---|---|---|---|"])
    for d in ("cwru", "seu"):
        if d not in set(df.dataset):
            continue
        for k in (5, 10):
            dd, r = diffs(d, k, "best")
            lo = dd.mean() - stats.t.ppf(0.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd)) if len(dd) > 1 else np.nan
            out.append(f"| {d.upper()} | {k}-shot | {r} | {dd.mean():+.1f} | {lo:+.1f} | "
                       f"{'DESTEKLENDİ' if lo > -3 else 'desteklenmedi'} |")
    return out
