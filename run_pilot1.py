"""Pilot deney 1 giriş noktası.

VS Code'da: bu dosyayı açıp F5 (ya da sağ üstteki ▷ Run Python File).
Aşağıdaki AYARLAR bloğunu değiştirerek deneyi yönetin; komut satırı argümanı verilirse
AYARLAR yerine onlar kullanılır (python run_pilot1.py --help).
"""

import sys

# ============================================================================ AYARLAR
AYARLAR = {
    # Önce True ile çalıştırıp her şeyin kurulu olduğunu görün (2-3 dk, indirme yok).
    # Sonra False yapın: gerçek veri ve gerçek TimEE/MantisV2 ağırlıkları kullanılır.
    "akis_testi": True,

    "data_root": "data",        # veriler buraya indirilir
    "out": "results",           # sonuçlar buraya yazılır
    # Paderborn (pu) için 7-Zip gerekir; önce cwru,seu ile başlayıp sonra pu eklenebilir.
    "datasets": "cwru,seu",
    "methods": "timee,timee_raw,mantisv2_lr,env_proto,wdcnn_tf,protonet_tf",
    "shots": "1,5,10",
    "episodes": 10,
    # Burada adı geçen yöntemlerin eski sonuçları silinip yeniden hesaplanır (boş bırakılabilir).
    # PU eklenirken ProtoNet'in iki kaynakla yeniden değerlendirilmesi için: "protonet_tf"
    "redo": "",
}
# ====================================================================================


def _args_from_settings(s: dict) -> list[str]:
    if s["akis_testi"]:
        return ["--synthetic", "--random-init-fm", "--episodes", "2", "--queries", "20",
                "--protonet-episodes", "60", "--out", s["out"] + "_akis_testi"]
    return ["--data-root", s["data_root"], "--out", s["out"], "--datasets", s["datasets"],
            "--methods", s["methods"], "--shots", s["shots"], "--episodes", str(s["episodes"]),
            "--redo", s.get("redo", "")]


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv += _args_from_settings(AYARLAR)
        print("Kullanılan ayarlar:", " ".join(sys.argv[1:]))
    from pilot1.run import parse_args, run
    run(parse_args())
