"""Pilot deney 1 ana betiği.

Örnek:
    python -m pilot1.run --data-root /content/drive/MyDrive/pilot1/data --out /content/drive/MyDrive/pilot1/results
    python -m pilot1.run --synthetic --random-init-fm --episodes 2      # hızlı akış testi
Kesilirse aynı komutla yeniden başlatın: biten (veri seti, epizot, shot, yöntem) satırları atlanır.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from .config import Config
from .data import load_dataset, split_mode
from .episodes import EpisodeSampler

KEYS = ["dataset", "episode", "shot", "method"]
DATASET_SEED_OFFSET = {"cwru": 0, "seu": 1, "pu": 2, "jnu": 3, "hit_mfs": 4, "hit_sb": 5}


# ----------------------------------------------------------------------------- yardımcılar

def metrics(yq, pred, proba, classes) -> dict:
    out = {
        "acc": accuracy_score(yq, pred),
        "bacc": balanced_accuracy_score(yq, pred),
        "f1_macro": f1_score(yq, pred, average="macro"),
        "auc_macro": np.nan,
    }
    try:
        # proba sütunları `classes` sırasında; sorguda olmayan sınıf yoksa OvR makro AUC
        y_idx = np.searchsorted(classes, yq)
        if len(classes) == 2:
            out["auc_macro"] = roc_auc_score(y_idx, proba[:, 1])
        else:
            out["auc_macro"] = roc_auc_score(y_idx, proba, multi_class="ovr", average="macro",
                                             labels=np.arange(len(classes)))
    except ValueError:
        pass
    return out


def build_methods(cfg: Config, names: list[str]) -> dict:
    # Windows'ta PyTorch, TensorFlow'dan SONRA yüklenirse TF'nin CUDA/cuDNN DLL'leriyle çakışıp
    # "WinError 127 ... shm.dll" hatası verebilir. Bu yüzden Torch her zaman önce yüklenir.
    if {"timee", "timee_raw", "mantisv2_lr"} & set(names):
        import torch  # noqa: F401
    m = {}
    for n in names:
        if n == "timee":
            from .methods_torch import TimeeMethod
            m[n] = TimeeMethod(cfg, use_ensemble=True)
        elif n == "timee_raw":
            from .methods_torch import TimeeMethod
            m[n] = TimeeMethod(cfg, use_ensemble=False)
        elif n == "mantisv2_lr":
            from .methods_torch import MantisLRMethod
            m[n] = MantisLRMethod(cfg)
        elif n == "env_proto":
            from .methods_classic import EnvelopeProtoMethod
            m[n] = EnvelopeProtoMethod(cfg)
        elif n == "wdcnn_tf":
            from .methods_tf import WDCNNMethod
            m[n] = WDCNNMethod(cfg)
        elif n == "protonet_tf":
            from .methods_tf import ProtoNetMethod
            m[n] = ProtoNetMethod(cfg, out_dir=os.path.join(cfg.out_dir, "weights"))
        else:
            raise ValueError(f"bilinmeyen yöntem: {n}")
    return m


# ----------------------------------------------------------------------------- ana döngü

def run(cfg: Config) -> pd.DataFrame:
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2, default=list)

    csv_path = os.path.join(cfg.out_dir, "episodes.csv")
    done = set()
    if os.path.exists(csv_path):
        prev = pd.read_csv(csv_path)
        if cfg.redo:
            drop = prev["method"].isin(cfg.redo)
            if drop.any():
                prev[~drop].to_csv(csv_path, index=False)
                print(f"yeniden hesaplanacak: {', '.join(cfg.redo)} ({int(drop.sum())} satır silindi)")
            prev = prev[~drop]
        done = set(map(tuple, prev[KEYS].astype(str).values))
        print(f"devam: {len(done)} satır zaten var")

    print("== veri yükleniyor ==")
    samplers = {}
    for name in cfg.datasets:
        t0 = time.time()
        recs = load_dataset(name, cfg)
        samplers[name] = EpisodeSampler(recs, cfg.window, cfg.max_windows_per_recording,
                                        split_mode(name, cfg))
        print(f"  {samplers[name].describe()}  ({time.time() - t0:.0f} sn)")
        del recs
        gc.collect()

    methods = build_methods(cfg, list(cfg.methods))
    kmax = max(cfg.shots)

    for target, sampler in samplers.items():
        print(f"== hedef: {target} ==")
        if "protonet_tf" in methods:
            sources = [s for n, s in samplers.items() if n != target]
            methods["protonet_tf"].meta_train(target, sources, cfg.window, seed=cfg.seed)

        # Epizot tohumu veri setine sabit: --datasets sırası değişse de aynı epizotlar üretilir
        rng = np.random.default_rng(cfg.seed + 1000 * DATASET_SEED_OFFSET.get(target, 9))
        for ep in range(cfg.episodes):
            episode = sampler.sample(rng, kmax, cfg.queries_per_class)
            for k in cfg.shots:
                Xs, ys = episode.support(k)
                for mname, method in methods.items():
                    key = (target, str(ep), str(k), mname)
                    if key in done:
                        continue
                    t0 = time.time()
                    if mname == "wdcnn_tf":
                        pred, proba, classes = method.fit_predict(Xs, ys, episode.Xq, seed=cfg.seed + ep)
                    else:
                        pred, proba, classes = method.fit_predict(Xs, ys, episode.Xq)
                    row = {"dataset": target, "episode": ep, "shot": k, "method": mname,
                           **metrics(episode.yq, pred, proba, classes),
                           "n_classes": len(classes), "n_support": len(ys), "n_query": len(episode.yq),
                           "support_groups": json.dumps(episode.info["support_groups"]),
                           "seconds": round(time.time() - t0, 2)}
                    pd.DataFrame([row]).to_csv(csv_path, mode="a", index=False,
                                               header=not os.path.exists(csv_path))
                    print(f"  ep{ep} {k:>2}-shot {mname:<12} acc={row['acc']:.3f} "
                          f"f1={row['f1_macro']:.3f} auc={row['auc_macro']:.3f} ({row['seconds']} sn)")

    df = pd.read_csv(csv_path)
    summarize(df, cfg.out_dir)
    return df


# ----------------------------------------------------------------------------- özet ve grafik

def summarize(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    from scipy import stats

    rows = []
    for (d, k, m), g in df.groupby(["dataset", "shot", "method"]):
        n = len(g)
        row = {"dataset": d, "shot": k, "method": m, "n_episodes": n}
        for col in ["acc", "bacc", "f1_macro", "auc_macro"]:
            v = g[col].dropna().to_numpy()
            mean = v.mean() if len(v) else np.nan
            half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
            row[col] = mean
            row[col + "_ci95"] = half
        rows.append(row)
    summ = pd.DataFrame(rows)
    chance = (1.0 / df.groupby("dataset")["n_classes"].max()).to_dict()
    summ["chance"] = summ["dataset"].map(chance)
    summ.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        for metric, title in [("acc", "Doğruluk (%)"), ("f1_macro", "Makro F1 (%)"),
                              ("auc_macro", "Makro ROC-AUC (%)")]:
            f.write(f"## {title} — ortalama ± %95 GA\n\n")
            cell = summ.assign(v=lambda s: [f"{100*a:.1f} ± {100*b:.1f}" if pd.notna(a) else "—"
                                            for a, b in zip(s[metric], s[metric + "_ci95"])])
            tab = cell.pivot_table(index="method", columns=["dataset", "shot"], values="v",
                                   aggfunc="first")
            f.write(tab.to_markdown() + "\n\n")
    print(open(os.path.join(out_dir, "summary.md")).read())
    plot(summ, out_dir)
    return summ


# Sabit sıralı kategorik palet (yöntem rengi her grafikte aynı kalır)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
METHOD_ORDER = ["timee", "timee_raw", "mantisv2_lr", "env_proto", "protonet_tf", "wdcnn_tf"]
LABELS = {"timee": "TimEE (resmi topluluk)", "timee_raw": "TimEE (ham 1024)",
          "mantisv2_lr": "MantisV2 + LR", "env_proto": "Zarf spektrumu + prototip",
          "protonet_tf": "ProtoNet (kaynakta meta-eğitim)", "wdcnn_tf": "WDCNN (sıfırdan)"}


def plot(summ: pd.DataFrame, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = list(dict.fromkeys(summ["dataset"]))
    shots = sorted(summ["shot"].unique())
    methods = [m for m in METHOD_ORDER if m in set(summ["method"])] + \
              [m for m in summ["method"].unique() if m not in METHOD_ORDER]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.8), sharey=True,
                             squeeze=False)
    for ax, d in zip(axes[0], datasets):
        for i, m in enumerate(methods):
            s = summ[(summ.dataset == d) & (summ.method == m)].sort_values("shot")
            if s.empty:
                continue
            x = [shots.index(v) for v in s["shot"]]
            ax.errorbar(x, 100 * s["acc"], yerr=100 * s["acc_ci95"].fillna(0),
                        color=PALETTE[i % len(PALETTE)], marker="o", ms=6, lw=2, capsize=3,
                        label=LABELS.get(m, m))
        ax.axhline(100 * summ.loc[summ.dataset == d, "chance"].iloc[0], color="#8a897f", lw=1,
                   ls="--", zorder=0)
        ax.text(len(shots) - 1, 100 * summ.loc[summ.dataset == d, "chance"].iloc[0] + 1.5, "şans",
                ha="right", va="bottom", fontsize=8, color="#5f5e57")
        ax.set_ylim(0, 100)
        ax.set_title(d.upper(), fontsize=11)
        ax.set_xticks(range(len(shots)), [f"{k}-shot" for k in shots])
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("Doğruluk (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Makineler arası az örnekli rulman teşhisi — ortalama ± %95 GA", y=1.15, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_vs_shots.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- CLI

def _items(text: str) -> tuple:
    """'cwru, seu ,PU' -> ('cwru', 'seu', 'pu'): boşluk ve büyük harf hatalarını tolere eder."""
    return tuple(t.strip().lower() for t in text.split(",") if t.strip())


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=cfg.data_root)
    p.add_argument("--out", default=cfg.out_dir)
    p.add_argument("--datasets", default=",".join(cfg.datasets))
    p.add_argument("--methods", default=",".join(cfg.methods))
    p.add_argument("--shots", default=",".join(map(str, cfg.shots)))
    p.add_argument("--episodes", type=int, default=cfg.episodes)
    p.add_argument("--queries", type=int, default=cfg.queries_per_class)
    p.add_argument("--window", type=int, default=cfg.window)
    p.add_argument("--fs", type=int, default=cfg.fs)
    p.add_argument("--protonet-episodes", type=int, default=cfg.protonet_episodes)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--redo", default="", help="eski satırları silinip yeniden hesaplanacak yöntemler, ör. protonet_tf")
    p.add_argument("--synthetic", action="store_true", help="sahte veriyle akış testi")
    p.add_argument("--random-init-fm", action="store_true", help="TimEE/Mantis ağırlığı indirmeden test")
    a = p.parse_args()
    unknown = [d for d in _items(a.datasets) if d not in DATASET_SEED_OFFSET]
    if unknown:
        p.error(f"bilinmeyen veri seti: {unknown} (geçerli: {', '.join(DATASET_SEED_OFFSET)})")
    return dataclasses.replace(
        cfg, data_root=a.data_root, out_dir=a.out, datasets=_items(a.datasets),
        methods=_items(a.methods), shots=tuple(int(s) for s in _items(a.shots)),
        episodes=a.episodes, queries_per_class=a.queries, window=a.window, fs=a.fs,
        protonet_episodes=a.protonet_episodes, seed=a.seed, synthetic=a.synthetic,
        random_init_fm=a.random_init_fm, redo=_items(a.redo))


if __name__ == "__main__":
    run(parse_args())
