"""Paderborn sağlık kontrolü: PU'daki düşük doğruluk veriden mi, rulmanlar arası kaymadan mı?

Üç bölme, 5-shot, aynı yöntemler:
  ayni_rulman  : destek ve sorgu AYNI rulmanın FARKLI ölçüm dosyalarından (sızıntı yok)
  farkli_rulman: pilot 1'deki gibi; destek sınıfın bir rulmanından, sorgu diğer rulmanlarından
Ek olarak farkli_rulman'da sorgu rulmanı başına doğruluk yazdırılır (hangi rulman bozuyor?).

Yorum:
  ayni_rulman yüksek (≳%90), farkli_rulman düşük → veri doğru okunuyor, sorun gerçek rulmanlar
      arası kayma. Pilot 2 için doğru hedef.
  ayni_rulman da düşük → yükleme/kanal/örnekleme hatası var, önce o düzeltilmeli.

VS Code'da F5. Yaklaşık 5-10 dk.
"""

import sys
from collections import defaultdict

AYARLAR = {
    "data_root": "data",
    "methods": "env_proto,mantisv2_lr,timee_raw",
    "shot": 5,
    "episodes": 10,
    "queries_per_class": 50,
    "test": False,   # True: sahte veri + rastgele ağırlık (yalnızca kodun çalıştığını görmek için)
}


def main(s):
    import numpy as np
    names = [m.strip() for m in s["methods"].split(",") if m.strip()]
    if {"timee", "timee_raw", "mantisv2_lr"} & set(names):
        import torch  # noqa: F401  (Windows DLL çakışması: TF'den önce)
    from pilot1.config import Config
    from pilot1.data import load_dataset
    from pilot1.episodes import EpisodeSampler
    from pilot1.run import build_methods

    cfg = Config(data_root=s["data_root"], synthetic=s["test"], random_init_fm=s["test"])
    recs = load_dataset("cwru" if s["test"] else "pu", cfg)
    smp = EpisodeSampler(recs, cfg.window, cfg.max_windows_per_recording, "per_class_group")
    print(smp.describe())
    methods = build_methods(cfg, names)
    k, q = s["shot"], s["queries_per_class"]

    def episode(rng, mode):
        Xs, ys, Xq, yq, qb = [], [], [], [], []
        for c in smp.classes:
            in_c = smp.lab == c
            bearings = sorted(set(smp.grp[in_c].tolist()))
            if mode == "ayni_rulman":
                ok = [b for b in bearings if len(set(smp.rec_id[in_c & (smp.grp == b)].tolist())) >= 2]
                b = ok[rng.integers(len(ok))]
                rids = sorted(set(smp.rec_id[in_c & (smp.grp == b)].tolist()))
                r = rids[rng.integers(len(rids))]
                s_idx = np.flatnonzero(in_c & (smp.rec_id == r))
                q_idx = np.flatnonzero(in_c & (smp.grp == b) & (smp.rec_id != r))
            else:
                b = bearings[rng.integers(len(bearings))]
                s_idx = np.flatnonzero(in_c & (smp.grp == b))
                q_idx = np.flatnonzero(in_c & (smp.grp != b))
            s_idx, q_idx = rng.permutation(s_idx)[:k], rng.permutation(q_idx)[:q]
            assert not set(smp.rec_id[s_idx]) & set(smp.rec_id[q_idx]), "sızıntı!"
            Xs.append(smp.W[s_idx]); ys += [c] * len(s_idx)
            Xq.append(smp.W[q_idx]); yq += [c] * len(q_idx); qb += smp.grp[q_idx].tolist()
        return np.concatenate(Xs), np.asarray(ys), np.concatenate(Xq), np.asarray(yq), np.asarray(qb)

    results = defaultdict(list)
    per_bearing = defaultdict(lambda: [0, 0])
    for mode in ("ayni_rulman", "farkli_rulman"):
        rng = np.random.default_rng(cfg.seed + 7)
        for ep in range(s["episodes"]):
            Xs, ys, Xq, yq, qb = episode(rng, mode)
            for n, m in methods.items():
                out = m.fit_predict(Xs, ys, Xq, seed=ep) if n == "wdcnn_tf" else m.fit_predict(Xs, ys, Xq)
                pred = out[0]
                results[(mode, n)].append(float((pred == yq).mean()))
                if mode == "farkli_rulman":
                    for b in np.unique(qb):
                        hit = pred[qb == b] == yq[qb == b]
                        per_bearing[(n, b)][0] += int(hit.sum()); per_bearing[(n, b)][1] += len(hit)
            print(f"  {mode} epizot {ep + 1}/{s['episodes']}")

    chance = 100 / len(smp.classes)
    print(f"\n=== {k}-shot doğruluk (%), {s['episodes']} epizot, şans = {chance:.1f} ===")
    print(f"{'yöntem':<14}{'ayni_rulman':>14}{'farkli_rulman':>16}")
    for n in names:
        a, b = (np.mean(results[(md, n)]) * 100 for md in ("ayni_rulman", "farkli_rulman"))
        print(f"{n:<14}{a:>14.1f}{b:>16.1f}")

    print("\n=== farkli_rulman: sorgu rulmanı başına doğruluk (%) ===")
    bearings = sorted({b for (_, b) in per_bearing})
    lab_of = {b: smp.lab[smp.grp == b][0] for b in bearings}
    print(f"{'rulman':<10}{'sınıf':<7}" + "".join(f"{n:>14}" for n in names))
    for b in bearings:
        row = "".join(f"{100 * per_bearing[(n, b)][0] / max(1, per_bearing[(n, b)][1]):>14.1f}" for n in names)
        print(f"{b:<10}{lab_of[b]:<7}{row}")


if __name__ == "__main__":
    if "--test" in sys.argv:
        AYARLAR["test"] = True
    main(AYARLAR)
