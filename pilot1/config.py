"""Deney ayarları. Komut satırından da değiştirilebilir (bkz. run.py --help)."""

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Yollar ---
    data_root: str = "data"          # ham veri burada tutulur
    out_dir: str = "results"         # sonuç CSV'leri, özet ve grafik

    # --- Ortak sinyal temsili ---
    fs: int = 12_000                 # tüm veri setleri bu örnekleme hızına getirilir (Hz)
    window: int = 1024               # pencere uzunluğu (örnek); 12 kHz'de ~85 ms
    max_windows_per_recording: int = 400  # bellek sınırı: kayıt başına en fazla pencere

    # --- Deney tasarımı ---
    datasets: tuple = ("cwru", "pu", "seu")
    methods: tuple = ("timee", "timee_raw", "mantisv2_lr", "env_proto", "wdcnn_tf", "protonet_tf")
    shots: tuple = (1, 5, 10)
    episodes: int = 10               # her veri seti için epizot sayısı
    queries_per_class: int = 50      # epizot başına sınıf başı sorgu penceresi
    seed: int = 0

    # --- Veri seti ayrıntıları ---
    seu_channel: int = 1             # SEU CSV'sinde kullanılacak kanal (0 tabanlı)
    pu_condition: str = "N15_M07_F10"  # Paderborn çalışma koşulu
    pu_max_measurements: int = 5     # rulman başına okunacak ölçüm sayısı (20'ye kadar)
    pu_bearings: dict = field(default_factory=lambda: {
        # Gerçek (doğal) hasarlı rulmanlar; birleşik hasarlılar (KB23/24/27) dışarıda
        "N": ["K001", "K002", "K003", "K004", "K005", "K006"],
        "OR": ["KA04", "KA15", "KA16", "KA22", "KA30"],
        "IR": ["KI04", "KI14", "KI16", "KI18", "KI21"],
    })

    # --- Yöntem ayarları ---
    timee_checkpoint: str = "liamsbhoo/timee"
    mantis_checkpoint: str = "paris-noah/MantisV2"
    mantis_layer: int = 2            # MantisV2 için önerilen ara katman
    mantis_len: int = 512            # MantisV2 ön eğitim uzunluğu
    env_fmax: float = 1000.0         # zarf spektrumunda tutulacak üst frekans (Hz)
    wdcnn_steps: int = 150           # destek kümesinde sıfırdan eğitim adımı
    protonet_episodes: int = 1500    # kaynak veri setlerinde meta-eğitim epizodu
    protonet_shots: int = 5
    protonet_queries: int = 10

    # --- Hata ayıklama ---
    random_init_fm: bool = False     # TimEE/Mantis'i ağırlık indirmeden rastgele başlat (yalnızca test)
    redo: tuple = ()                 # bu yöntemlerin eski sonuçları silinip yeniden hesaplanır
    synthetic: bool = False          # gerçek veri yerine sentetik veri (yalnızca test)
