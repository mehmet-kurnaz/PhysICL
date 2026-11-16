"""Makale deney planı: tek F5 ile çalışan üç aşama.

  A : eğitim tohumları (fizik_v1, genel_v1 için tohum 1 ve 2) + dışarıda tutulan 3 veri setinde
      (JNU, HIT-MFS, HIT-SB) önkayıtlı test                              -> pilot2/ONKAYIT_A.md
  B : ablasyonlar (simülatör katmanları) ve eğitim süresi; keşif amaçlı
  C : simülatör v1 (CWRU tipi rejim) ile fizik_v2; tohum 2 epizotlarında önkayıtlı doğrulama
                                                                          -> pilot2/ONKAYIT_C.md
Her aşama kaldığı yerden devam eder: biten eğitimler ve taban çizgisi satırları tekrar hesaplanmaz.
"""

from __future__ import annotations

import os

DEV = ("cwru", "seu", "pu")
HELDOUT = ("jnu", "hit_mfs", "hit_sb")
ABLATIONS = ("birim_yok", "kosul_yok", "bant_yok", "golge_yok", "mil_yok")
LENGTHS = (500, 1000, 4000)
N_EP = 30


def _p(s, *a):
    return os.path.join(s["out"], *a)


def _train(s, etiket, onsel="fizik", seed=0, steps=2000, surum="v0", ablasyon=""):
    from pilot2 import finetune
    done = os.path.join(_p(s, f"ince_ayar_{etiket}", "son"), "model.safetensors")
    log = _p(s, f"ince_ayar_{etiket}", "log.csv")
    if os.path.exists(done) and os.path.exists(log):
        import pandas as pd
        last = pd.read_csv(log)["step"].max()
        if last >= (40 if s["test"] else steps):
            print(f"  {etiket}: zaten eğitilmiş, atlanıyor")
            return
    print(f"\n===== ince ayar: {etiket} =====")
    finetune.run({**s, "onsel": onsel, "etiket": etiket, "seed": seed, "adim_sayisi": steps,
                  "kayit_araligi": min(500, steps), "simulator_surum": surum, "ablasyon": ablasyon})


def _baselines(s, datasets, seed, out):
    import torch  # noqa: F401  (Windows: TF'den önce)
    from pilot1.config import Config
    from pilot1.run import run as p1run
    print(f"\n===== taban çizgileri: {', '.join(datasets)} (tohum {seed}) =====")
    p1run(Config(data_root=s["data_root"], out_dir=out, datasets=tuple(datasets),
                 methods=("timee_raw", "mantisv2_lr", "env_proto"), seed=seed,
                 episodes=2 if s["test"] else N_EP, synthetic=s["test"], random_init_fm=s["test"],
                 queries_per_class=20 if s["test"] else Config.queries_per_class))
    return os.path.join(out, "episodes.csv")


def _eval(s, modeller, datasets, seed, klasor, taban_csv):
    from pilot2 import evaluate
    print(f"\n===== değerlendirme: {klasor} =====")
    evaluate.run({**s, "modeller": modeller, "gorunumler": "ham+zarf", "datasets": ",".join(datasets),
                  "degerlendirme_seed": seed, "epizot": N_EP, "degerlendirme_klasoru": klasor,
                  "pilot1_sonuc": taban_csv})
    return _p(s, klasor, "episodes.csv")


def stage_A(s):
    from pilot2.analysis import analysis_A, load
    for seed in (1, 2):
        _train(s, f"fizik_v1_s{seed}", "fizik", seed)
        _train(s, f"genel_v1_s{seed}", "genel", seed)
    held = list(HELDOUT) if not s["test"] else ["jnu", "hit_mfs"]
    tb = _baselines(s, held, 1, _p(s, "A_taban"))
    ft = _eval(s, "hazir,fizik_v1,fizik_v1_s1,fizik_v1_s2,genel_v1,genel_v1_s1,genel_v1_s2", held, 1, "A_dis_test", tb)
    dev_tb = _baselines(s, DEV, 1, _p(s, "dogrulama_taban"))      # zaten varsa yeniden hesaplanmaz
    dev = _eval(s, "fizik_v1,fizik_v1_s1,fizik_v1_s2", DEV, 1, "A_tohum", dev_tb)
    analysis_A(load(tb, ft), load(dev_tb, dev), held, _p(s, "A_sonuc.md"))


def stage_B(s):
    from pilot2.analysis import load, table_B
    tags = []
    for a in ABLATIONS:
        _train(s, f"abl_{a}", "fizik", 0, ablasyon=a); tags.append(f"abl_{a}")
    for n in LENGTHS:
        _train(s, f"uzunluk_{n}", "fizik", 0, steps=n); tags.append(f"uzunluk_{n}")
    dev_tb = _baselines(s, DEV, 1, _p(s, "dogrulama_taban"))
    ft = _eval(s, ",".join(["fizik_v1"] + tags), DEV, 1, "B_ablasyon", dev_tb)
    table_B(load(dev_tb, ft), tags, _p(s, "B_sonuc.md"))


def stage_C(s):
    from pilot2.analysis import analysis_C, load
    _train(s, "fizik_v2", "fizik", 0, surum="v1")
    tb = _baselines(s, DEV, 2, _p(s, "C_taban"))
    ft = _eval(s, "fizik_v1,fizik_v2", DEV, 2, "C_dogrulama", tb)
    analysis_C(load(tb, ft), _p(s, "C_sonuc.md"))


STAGES = {"A": stage_A, "B": stage_B, "C": stage_C}
