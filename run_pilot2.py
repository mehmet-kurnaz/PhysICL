"""Pilot 2 giriş noktası.

VS Code'da: bu dosyayı açıp F5. Aşağıdaki AYARLAR bloğu deneyi yönetir.
Bu dosya run_pilot1.py ile aynı klasörde, pilot1/ ve pilot2/ klasörlerinin yanında durmalı.

v1 sırası:
  1. "degerlendirme" + "modeller": "hazir"   -> eğitimsiz: hazır TimEE'ye zarf görünümü (~10 dk)
  2. "v1_hepsi"                              -> fizik_v1, genel_v1, karma_v1 eğitimi (her biri 2000 adım)
                                                ardından hepsinin değerlendirmesi, tek F5 (~40 dk)
  3. "v1_dogrulama"                          -> ÖNKAYITLI doğrulama: yeni 30 epizot (tohum 1), taban
                                                çizgileri + hazır/fizik_v1/genel_v1 x ham+zarf/zarf (bkz. pilot2/ONKAYIT.md)
Makale aşamaları (bkz. pilot2/plan.py, ONKAYIT_A.md, ONKAYIT_C.md):
  "A"   -> eğitim tohumları + dışarıda tutulan 3 veri seti (önkayıtlı)          ~1,5 saat
  "B"   -> ablasyonlar + eğitim süresi (keşif)                                   ~1,5 saat
  "C"   -> simülatör v1 + tohum 2 doğrulaması (önkayıtlı)                        ~1 saat
  "ABC" -> üçü arka arkaya. Sonuçlar: results_pilot2\A_sonuc.md, B_sonuc.md, C_sonuc.md
"uyum" -> simülatör-gerçek ölçüm kapsama analizi (model yok, ~15-30 dk). Çıktı: results_pilot2/uyum/
Tek bir eğitim için "ince_ayar" (önsel ve etiket ayardan). Ayrıntı: README.md
"""

import sys

# ============================================================================ AYARLAR
AYARLAR = {
    "adim": "ABC",                   # A | B | C | ABC | uyum | onizleme | ince_ayar | degerlendirme | v1_hepsi | v1_dogrulama
    "test": False,                   # True: 1-2 dakikalık akış testi (gerçek ağırlık/veri yok)
    "data_root": "data",
    "out": "results_pilot2",
    "seed": 0,

    # --- ince_ayar
    "onsel": "karma",                # fizik | genel (kontrol) | karma
    "etiket": "karma_v1",            # klasör adı: results_pilot2/ince_ayar_<etiket>
    "karma_fizik_orani": 0.5,        # karma önselde epizotların ne kadarı fizikten
    "adim_sayisi": 2000,
    "batch": 8,                      # GPU belleği yetmezse 4
    "lr": 2e-5,
    "workers": 4,                    # epizot üreten işlemci çekirdeği sayısı
    "kayit_araligi": 500,            # kaç adımda bir ağırlık kaydedilsin
    "timee_checkpoint": "liamsbhoo/timee",

    # --- degerlendirme
    "modeller": "hazir",                       # ör. "hazir,fizik_v1,genel_v1,karma_v1" ya da "fizik_v1@1000"
    "gorunumler": "ham,zarf,ham+zarf",
    "datasets": "cwru,seu,pu",
    "pilot1_sonuc": "results/episodes.csv",   # taban çizgileri buradan okunur (yeniden hesaplanmaz)

    # --- onizleme
    "gercek_veri": "cwru,seu,pu",
    "zorluk_yontemleri": "env_proto,mantisv2_lr,timee_raw",
    "zorluk_makine": 20,
}
# ====================================================================================


if __name__ == "__main__":
    if "--test" in sys.argv:
        AYARLAR["test"] = True
    for a in sys.argv[1:]:                       # ör. python run_pilot2.py adim=degerlendirme onsel=genel
        if "=" in a:
            k, v = a.split("=", 1)
            AYARLAR[k] = type(AYARLAR[k])(v) if k in AYARLAR and not isinstance(AYARLAR[k], bool) else v
    adim = AYARLAR["adim"]
    if adim == "onizleme":
        from pilot2.preview import run
    elif adim == "ince_ayar":
        from pilot2.finetune import run
    elif adim == "degerlendirme":
        from pilot2.evaluate import run
    elif adim == "uyum":
        from pilot2.fidelity import run
    elif adim == "v1_hepsi":
        from pilot2 import evaluate, finetune

        def run(s):
            for onsel in ("fizik", "genel", "karma"):
                print(f"\n===== ince ayar: {onsel}_v1 =====")
                finetune.run({**s, "onsel": onsel, "etiket": f"{onsel}_v1"})
            print("\n===== değerlendirme =====")
            evaluate.run({**s, "modeller": "hazir,fizik_v1,genel_v1,karma_v1"})
    elif adim in ("A", "B", "C", "ABC"):
        from pilot2.plan import STAGES

        def run(s):
            for st in adim:
                print(f"\n#################### AŞAMA {st} ####################")
                STAGES[st](s)
    elif adim == "v1_dogrulama":
        import os

        def run(s):
            import torch  # noqa: F401  (Windows: TF'den önce)
            from pilot1.config import Config
            from pilot1.run import run as p1run
            from pilot2 import evaluate
            seed, n = 1, (2 if s["test"] else 30)            # ÖNKAYIT: değiştirmeyin
            base_out = os.path.join(s["out"], "dogrulama_taban")
            datasets = tuple(d.strip().lower() for d in s["datasets"].split(",") if d.strip())
            print("\n===== doğrulama: taban çizgileri (yeni epizotlar) =====")
            p1run(Config(data_root=s["data_root"], out_dir=base_out, datasets=datasets,
                         methods=("timee_raw", "mantisv2_lr", "env_proto"), seed=seed, episodes=n,
                         synthetic=s["test"], random_init_fm=s["test"],
                         queries_per_class=20 if s["test"] else Config.queries_per_class))
            print("\n===== doğrulama: ince ayarlı modeller =====")
            evaluate.run({**s, "modeller": "hazir,fizik_v1,genel_v1", "gorunumler": "ham+zarf,zarf",
                          "degerlendirme_seed": seed, "epizot": n, "degerlendirme_klasoru": "dogrulama",
                          "pilot1_sonuc": os.path.join(base_out, "episodes.csv")})
    else:
        raise SystemExit(f"bilinmeyen adım: {adim} (A | B | C | ABC | uyum | onizleme | ince_ayar | degerlendirme | v1_hepsi | v1_dogrulama)")
    run(AYARLAR)
