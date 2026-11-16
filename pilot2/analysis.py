"""Önkayıtlı analizler (A ve C) ve keşif amaçlı ablasyon tablosu (B).

Hipotezler pilot2/ONKAYIT_A.md ve pilot2/ONKAYIT_C.md dosyalarında, veriler görülmeden yazıldı.
Bu dosya yalnızca o hipotezleri hesaplar; eşikler ve karşılaştırmalar orada sabitlendi.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon

BASELINES = ("timee_raw", "mantisv2_lr", "env_proto")
VIEW = "ham+zarf"
NI_MARGIN = 3.0          # "zarar yok" eşiği (puan)


def load(*csvs) -> pd.DataFrame:
    dfs = [pd.read_csv(c) for c in csvs if os.path.exists(c)]
    df = pd.concat(dfs).drop_duplicates(["dataset", "episode", "shot", "method"], keep="last")
    return df


def cell(df, d, k) -> pd.DataFrame:
    """episode x method doğruluk (%) tablosu"""
    return df[(df.dataset == d) & (df.shot == k)].pivot_table(index="episode", columns="method", values="acc") * 100


def seed_mean(g: pd.DataFrame, tags: list[str]) -> pd.Series | None:
    cols = [f"{t}|{VIEW}" for t in tags if f"{t}|{VIEW}" in g]
    return g[cols].mean(axis=1) if cols else None


def best_baseline(g: pd.DataFrame) -> str:
    return max([b for b in BASELINES if b in g], key=lambda b: g[b].mean())


def _w(d: np.ndarray) -> float:
    d = np.asarray(d, float)
    return float(wilcoxon(d).pvalue) if len(d) > 1 and np.any(d != 0) else 1.0


def _lb(d: np.ndarray) -> float:
    d = np.asarray(d, float)
    return float(d.mean() - stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))


def _holm(ps):
    ps = np.asarray(ps, float)
    order, adj, run = np.argsort(ps), np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def _ok(b: bool) -> str:
    return "DESTEKLENDİ" if b else "desteklenmedi"


# ----------------------------------------------------------------------------- A

def analysis_A(heldout: pd.DataFrame, dev: pd.DataFrame, datasets, out_md: str):
    prim = ["fizik_v1", "fizik_v1_s1", "fizik_v1_s2"]
    ctrl = ["genel_v1", "genel_v1_s1", "genel_v1_s2"]
    L = ["# A — dışarıda tutulan veri setleri ve eğitim tohumları (ÖNKAYIT_A)\n",
         f"Ana yöntem: TimEE + fizik_v1 · {VIEW}, 3 eğitim tohumunun epizot başına ortalaması. "
         "Kontrol: genel_v1, aynı şekilde. Doğruluk %.\n",
         "## Hücre tablosu\n",
         "| veri | shot | ana yöntem | kontrol | TimEE hazır·ham | MantisV2 | zarf spektrumu | TimEE hazır·ham+zarf | en iyi taban | ana − en iyi (ort., %95 GA alt) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    pooled_b, pooled_c = [], []
    for d in datasets:
        for k in (1, 5, 10):
            g = cell(heldout, d, k)
            if g.empty:
                continue
            p, c, b = seed_mean(g, prim), seed_mean(g, ctrl), best_baseline(g)
            diff = (p - g[b]).dropna().to_numpy()
            hz = g.get(f"hazir|{VIEW}")
            L.append(f"| {d.upper()} | {k} | {p.mean():.1f} | {c.mean():.1f} | {g['timee_raw'].mean():.1f} | "
                     f"{g['mantisv2_lr'].mean():.1f} | {g['env_proto'].mean():.1f} | "
                     f"{hz.mean() if hz is not None else float('nan'):.1f} | {b} | {diff.mean():+.1f} ({_lb(diff):+.1f}) |")
            if k in (5, 10):
                pooled_b.append((p - g[b]).sort_index().to_numpy())
                pooled_c.append((p - c).sort_index().to_numpy())
    L.append("")
    # HA1, HA2: 6 hücrenin epizot başına ortalaması (hücreler bağımsız; epizot sırası eşleştirme birimi)
    n = min(len(x) for x in pooled_b)
    pb = np.mean([x[:n] for x in pooled_b], axis=0)
    pc = np.mean([x[:n] for x in pooled_c], axis=0)
    L += ["## Hipotezler\n", "| hipotez | ölçüt | değer | sonuç |", "|---|---|---|---|",
          f"| HA1 — 3 dış veri × 5/10-shot birleşik: ana > en iyi taban | ort. fark, Wilcoxon p | "
          f"{pb.mean():+.2f}, p={_w(pb):.4f} | {_ok(pb.mean() > 0 and _w(pb) < 0.05)} |",
          f"| HA2 — birleşik: ana > kontrol (genel önsel) | ort. fark, Wilcoxon p | "
          f"{pc.mean():+.2f}, p={_w(pc):.4f} | {_ok(pc.mean() > 0 and _w(pc) < 0.05)} |"]
    lbs = []
    for d in datasets:
        for k in (5, 10):
            g = cell(heldout, d, k)
            if g.empty:
                continue
            diff = (seed_mean(g, prim) - g[best_baseline(g)]).dropna().to_numpy()
            lbs.append((d, k, _lb(diff)))
    ok3 = all(lb > -NI_MARGIN for _, _, lb in lbs)
    L.append(f"| HA3 — her hücrede zarar yok (%95 GA alt sınır > −{NI_MARGIN:.0f}) | en kötü hücre | "
             f"{min(lbs, key=lambda x: x[2])[0].upper()} {min(lbs, key=lambda x: x[2])[1]}-shot: "
             f"{min(x[2] for x in lbs):+.1f} | {_ok(ok3)} |")
    # HA4: tohum tutarlılığı, PU 10-shot geliştirme-doğrulama epizotları (tohum 1)
    g = cell(dev, "pu", 10)
    vals = []
    for t in prim:
        col = f"{t}|{VIEW}"
        if col in g:
            vals.append((t, (g[col] - g["env_proto"]).mean()))
    ok4 = len(vals) == 3 and all(v > 0 for _, v in vals)
    L.append(f"| HA4 — PU 10-shot (doğrulama epizotları): 3 eğitim tohumunun HER BİRİ zarf spektrumundan yüksek | "
             f"tohum başına ort. fark | {', '.join(f'{t}: {v:+.1f}' for t, v in vals)} | {_ok(ok4)} |")
    L.append("\n## Eğitim tohumları arası tutarlılık (geliştirme veri setleri, tohum 1 epizotları)\n")
    L += ["| veri | shot | " + " | ".join(prim) + " | ort. ± std |", "|---|---|" + "---|" * (len(prim) + 1)]
    for d in ("cwru", "seu", "pu"):
        for k in (1, 5, 10):
            g = cell(dev, d, k)
            if g.empty:
                continue
            v = [g[f"{t}|{VIEW}"].mean() for t in prim if f"{t}|{VIEW}" in g]
            L.append(f"| {d.upper()} | {k} | " + " | ".join(f"{x:.1f}" for x in v) +
                     f" | {np.mean(v):.1f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.1f} |")
    md = "\n".join(L)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)


# ----------------------------------------------------------------------------- B

def table_B(df: pd.DataFrame, tags: list[str], out_md: str):
    """Ablasyonlar: her model − tam model (fizik_v1), eşleştirilmiş. Keşif amaçlı (önkayıt yok)."""
    ref = f"fizik_v1|{VIEW}"
    L = ["# B — ablasyonlar ve eğitim süresi (keşif amaçlı)\n",
         f"Tohum 1 doğrulama epizotları (30). Hücre: doğruluk %, parantezde tam modele (fizik_v1, 2000 adım) göre "
         "eşleştirilmiş fark; * = Wilcoxon p < 0,05 (çoklu karşılaştırma düzeltmesi yok).\n"]
    for d in ("cwru", "seu", "pu"):
        L += [f"\n## {d.upper()}\n", "| model | 1-shot | 5-shot | 10-shot |", "|---|---|---|---|"]
        for t in ["fizik_v1"] + tags:
            cells = []
            for k in (1, 5, 10):
                g = cell(df, d, k)
                col = f"{t}|{VIEW}"
                if col not in g or ref not in g:
                    cells.append("—")
                    continue
                diff = (g[col] - g[ref]).dropna().to_numpy()
                star = "*" if _w(diff) < 0.05 else ""
                cells.append(f"{g[col].mean():.1f}" + ("" if t == "fizik_v1" else f" ({diff.mean():+.1f}{star})"))
            L.append(f"| {t} | " + " | ".join(cells) + " |")
    md = "\n".join(L)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)


# ----------------------------------------------------------------------------- C

def analysis_C(df: pd.DataFrame, out_md: str):
    v1, v2 = f"fizik_v1|{VIEW}", f"fizik_v2|{VIEW}"
    L = ["# C — simülatör v1 (CWRU tipi rejim) önkayıtlı doğrulama (ÖNKAYIT_C, tohum 2 epizotları)\n",
         "| veri | shot | fizik_v2 | fizik_v1 | en iyi taban | v2 − en iyi (ort., %95 GA alt) | v2 − v1 (ort., p) |",
         "|---|---|---|---|---|---|---|"]
    for d in sorted(set(df.dataset)):
        for k in (1, 5, 10):
            g = cell(df, d, k)
            if g.empty or v2 not in g:
                continue
            b = best_baseline(g)
            db = (g[v2] - g[b]).dropna().to_numpy()
            dv = (g[v2] - g[v1]).dropna().to_numpy() if v1 in g else np.array([np.nan])
            L.append(f"| {d.upper()} | {k} | {g[v2].mean():.1f} | {g[v1].mean() if v1 in g else float('nan'):.1f} | "
                     f"{b} ({g[b].mean():.1f}) | {db.mean():+.1f} ({_lb(db):+.1f}) | {dv.mean():+.1f} (p={_w(dv):.3f}) |")
    L += ["\n## Hipotezler\n", "| hipotez | değer | sonuç |", "|---|---|---|"]

    def nb(d, k):
        g = cell(df, d, k)
        return (g[v2] - g[best_baseline(g)]).dropna().to_numpy()

    c1 = [(k, _lb(nb("cwru", k))) for k in (5, 10)]
    L.append(f"| HC1 — CWRU 5/10-shot zarar yok (alt sınır > −{NI_MARGIN:.0f}) | "
             f"{', '.join(f'{k}-shot {x:+.1f}' for k, x in c1)} | {_ok(all(x > -NI_MARGIN for _, x in c1))} |")
    ps, ms = [], []
    for k in (5, 10):
        g = cell(df, "cwru", k)
        dv = (g[v2] - g[v1]).dropna().to_numpy()
        ps.append(_w(dv)); ms.append(dv.mean())
    adj = _holm(ps)
    L.append(f"| HC2 — CWRU 5/10-shot fizik_v2 > fizik_v1 (Holm) | "
             f"{', '.join(f'{k}-shot {m:+.1f} (Holm p={a:.3f})' for k, m, a in zip((5, 10), ms, adj))} | "
             f"{_ok(all(m > 0 and a < 0.05 for m, a in zip(ms, adj)))} |")
    d3 = nb("pu", 10)
    L.append(f"| HC3 — PU 10-shot fizik_v2 > en iyi taban (tekrar) | {d3.mean():+.1f}, p={_w(d3):.4f} | "
             f"{_ok(d3.mean() > 0 and _w(d3) < 0.05)} |")
    c4 = [(k, _lb(nb("seu", k))) for k in (5, 10)]
    L.append(f"| HC4 — SEU 5/10-shot zarar yok | {', '.join(f'{k}-shot {x:+.1f}' for k, x in c4)} | "
             f"{_ok(all(x > -NI_MARGIN for _, x in c4))} |")
    md = "\n".join(L)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)
