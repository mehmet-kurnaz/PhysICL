"""Veri indirme ve yükleme: CWRU, Paderborn (PU), SEU.

Her veri seti bir `Recording` listesine dönüşür. Bir kayıt = tek bir ölçüm dosyası.
`group` alanı destek/sorgu ayrımında kullanılır:
  - CWRU: yük (0-3 HP)        -> destek bir yükten, sorgular diğer yüklerden
  - SEU : çalışma koşulu      -> destek bir koşuldan, sorgular diğerinden
  - PU  : rulman kimliği       -> destek bir rulmandan, sorgular aynı sınıfın diğer rulmanlarından
Böylece destek ve sorgu pencereleri hiçbir zaman aynı kayıttan gelmez.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

RAW_BASE = "https://raw.githubusercontent.com/cathysiyu/Mechanical-datasets/master/"
_CWRU_NAMES = (["normal"] + [f"{f}{s}" for f in ("B", "IR") for s in ("007", "014", "021")]
               + [f"OR{s}@6" for s in ("007", "014", "021")])
CWRU_FILES = [f"dataset/{n}_{load}.mat" for n in _CWRU_NAMES for load in range(4)]
SEU_FILES = [f"gearbox/bearingset/{c}_{cond}.csv"
             for c in ("ball", "comb", "health", "inner", "outer") for cond in ("20_0", "30_2")]
PU_URL = "https://zenodo.org/records/15845309/files/{bearing}.rar?download=1"


@dataclass
class Recording:
    dataset: str
    label: str      # sınıf etiketi (N, IR, OR, B, IR+OR)
    group: str      # bölme birimi (yük, koşul veya rulman)
    unit: str       # dosya adı
    signal: np.ndarray  # cfg.fs hızına örneklenmiş, float32


def resample_to(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if fs_in == fs_out:
        return x.astype(np.float32)
    fr = Fraction(fs_out, fs_in).limit_denominator(1000)
    return resample_poly(x, fr.numerator, fr.denominator).astype(np.float32)


# ----------------------------------------------------------------------------- indirme

def _run(cmd: list[str], cwd: str | None = None) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _download(url: str, dest: str, tries: int = 20) -> None:
    """Dosyayı indirir. Bağlantı koparsa kaldığı bayttan devam eder (HTTP Range), 20 kez dener."""
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    print(f"  indiriliyor: {url}")
    attempt, last = 0, -1
    while attempt < tries:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        attempt = 1 if have > last else attempt + 1   # ilerleme varsa sayaç sıfırlanır
        last = have
        req = urllib.request.Request(url, headers={"User-Agent": "pilot1/1.0",
                                                   **({"Range": f"bytes={have}-"} if have else {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:       # sunucu devam etmeyi desteklemiyorsa baştan
                    have = 0
                total = have + int(r.headers.get("Content-Length", 0))
                with open(part, "ab" if have else "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        if total:
                            print(f"\r    {have / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
            print()
            if total and have < total:
                raise OSError(f"eksik indirme ({have}/{total} bayt)")
            os.replace(part, dest)
            return
        except urllib.error.HTTPError as e:
            if e.code == 416:                      # .part zaten tam
                os.replace(part, dest)
                return
            err = e
        except (OSError, urllib.error.URLError) as e:
            err = e
        wait = 2 if attempt == 1 else min(60, 5 * attempt)
        print(f"\n    bağlantı koptu ({err}); {wait} sn sonra devam (ilerlemesiz deneme {attempt}/{tries})")
        time.sleep(wait)
    raise RuntimeError(f"{url} indirilemedi; tekrar çalıştırınca kaldığı yerden devam eder.")


def fetch_github_files(data_root: str, files: list[str]) -> str:
    """CWRU/SEU dosyalarını GitHub'dan doğrudan indirir (git gerekmez)."""
    dest = os.path.join(data_root, "Mechanical-datasets")
    for rel in files:
        _download(RAW_BASE + urllib.parse.quote(rel), os.path.join(dest, *rel.split("/")))
    return dest


def _find_7z() -> str | None:
    exe = shutil.which("7z") or shutil.which("7z.exe")
    if exe:
        return exe
    adaylar = [
        r"C:\Program Files\7-Zip\7z.exe",                                   # winget (makine geneli)
        r"C:\Program Files (x86)\7-Zip\7z.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\7-Zip\7z.exe"),        # winget (kullanıcıya özel)
        os.path.expandvars(r"%ProgramW6432%\7-Zip\7z.exe"),
    ]
    try:                                                                    # 7-Zip kurulum yolunu kayıt defterine yazar
        import winreg
        for kok in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                with winreg.OpenKey(kok, r"SOFTWARE\7-Zip") as k:
                    adaylar.append(os.path.join(winreg.QueryValueEx(k, "Path")[0], "7z.exe"))
            except OSError:
                pass
    except ImportError:                                                     # Windows değil
        pass
    for p in adaylar:
        if os.path.exists(p):
            return p
    return None


def _extract_rar(rar_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if shutil.which("unrar"):
        _run(["unrar", "x", "-o+", "-inul", rar_path, out_dir + os.sep])
    elif _find_7z():
        _run([_find_7z(), "x", "-y", f"-o{out_dir}", rar_path])
    elif shutil.which("unar"):
        _run(["unar", "-f", "-o", out_dir, rar_path])
    else:
        raise RuntimeError("Paderborn .rar dosyalarını açmak için 7-Zip gerekli "
                           "(Windows: https://www.7-zip.org, Ubuntu: sudo apt install unrar). "
                           "Ya da --datasets cwru,seu ile Paderborn'suz çalıştırın.")


def fetch_pu(data_root: str, bearings: list[str]) -> str:
    root = os.path.join(data_root, "paderborn")
    os.makedirs(root, exist_ok=True)
    for b in bearings:
        if glob.glob(os.path.join(root, "**", f"*_{b}_*.mat"), recursive=True):
            continue
        rar = os.path.join(root, f"{b}.rar")
        if not os.path.exists(rar):
            _download(PU_URL.format(bearing=b), rar)
        _extract_rar(rar, root)
    return root


# ----------------------------------------------------------------------------- CWRU

_CWRU_LABEL = {"normal": "N", "IR": "IR", "B": "B", "OR": "OR"}


def load_cwru(data_root: str, cfg) -> list[Recording]:
    repo = fetch_github_files(data_root, CWRU_FILES)
    recs = []
    for path in sorted(glob.glob(os.path.join(repo, "dataset", "*.mat"))):
        name = os.path.splitext(os.path.basename(path))[0]      # ör. IR007_0, OR014@6_2, normal_1
        m = re.match(r"^(normal|IR|B|OR)[^_]*_(\d)$", name)
        if not m:
            continue
        mat = loadmat(path)
        key = next((k for k in mat if k.endswith("_DE_time")), None)
        if key is None:
            print(f"  uyarı: {name} içinde _DE_time yok, atlandı")
            continue
        sig = resample_to(mat[key], 12_000, cfg.fs)
        recs.append(Recording("cwru", _CWRU_LABEL[m.group(1)], f"load{m.group(2)}", name, sig))
    return recs


# ----------------------------------------------------------------------------- SEU

_SEU_LABEL = {"health": "N", "inner": "IR", "outer": "OR", "ball": "B", "comb": "IR+OR"}
SEU_FS = 5_120  # 2,56 x 2000 Hz frekans sınırı


def _read_seu_csv(path: str, channel: int) -> np.ndarray:
    import pandas as pd

    with open(path, "r", errors="ignore") as f:
        header_lines = 0
        for line in f:
            header_lines += 1
            if line.strip().strip(",\t").lower() == "data":
                break
        first = f.readline()
    sep = "," if first.count(",") > first.count("\t") else "\t"
    df = pd.read_csv(path, skiprows=header_lines, sep=sep, header=None, usecols=[channel],
                     dtype=np.float64, engine="c")
    return df.iloc[:, 0].to_numpy()


def load_seu(data_root: str, cfg) -> list[Recording]:
    repo = fetch_github_files(data_root, SEU_FILES)
    recs = []
    for path in sorted(glob.glob(os.path.join(repo, "gearbox", "bearingset", "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]      # ör. inner_30_2
        m = re.match(r"^(health|inner|outer|ball|comb)_(\d+_\d+)$", name)
        if not m:
            continue
        x = _read_seu_csv(path, cfg.seu_channel)
        recs.append(Recording("seu", _SEU_LABEL[m.group(1)], m.group(2), name,
                              resample_to(x, SEU_FS, cfg.fs)))
    return recs


# ----------------------------------------------------------------------------- Paderborn

PU_FS = 64_000


def _field(rec, name):
    """MATLAB struct alanını (numpy structured array) güvenle okur."""
    if rec.dtype.names and name in rec.dtype.names:
        return rec[name]
    return None


def _pu_vibration(mat: dict) -> np.ndarray:
    key = [k for k in mat if not k.startswith("__")][0]
    s = mat[key][0, 0]
    Y = _field(s, "Y")
    if Y is None:
        raise ValueError("PU dosyasında 'Y' alanı yok")
    candidates = []
    for ch in np.ravel(Y):
        nm = _field(ch, "Name")
        data = _field(ch, "Data")
        if data is None:
            continue
        data = np.asarray(data).ravel()
        label = str(np.ravel(nm)[0]) if nm is not None and np.size(nm) else ""
        if "vibration" in label.lower():
            return data
        if data.size > 200_000:          # 64 kHz kanallar: akım_1, akım_2, titreşim_1
            candidates.append(data)
    if not candidates:
        raise ValueError("PU dosyasında titreşim kanalı bulunamadı")
    return candidates[-1]                # titreşim, yüksek hızlı kanalların sonuncusudur


def load_pu(data_root: str, cfg) -> list[Recording]:
    all_bearings = [b for bs in cfg.pu_bearings.values() for b in bs]
    root = fetch_pu(data_root, all_bearings)
    recs = []
    for label, bearings in cfg.pu_bearings.items():
        for b in bearings:
            files = sorted(glob.glob(os.path.join(root, "**", f"{cfg.pu_condition}_{b}_*.mat"),
                                     recursive=True))[: cfg.pu_max_measurements]
            if not files:
                print(f"  uyarı: {b} için {cfg.pu_condition} ölçümü bulunamadı")
            for path in files:
                try:
                    x = _pu_vibration(loadmat(path))
                except Exception as e:  # bilinen birkaç bozuk dosya var
                    print(f"  uyarı: {os.path.basename(path)} okunamadı ({e})")
                    continue
                recs.append(Recording("pu", label, b, os.path.basename(path),
                                      resample_to(x, PU_FS, cfg.fs)))
    return recs


# ----------------------------------------------------------------------------- sentetik (test)

def make_synthetic(name: str, cfg, seed: int) -> list[Recording]:
    """Akışı gerçek veri olmadan test etmek için fiziksel olarak makul sahte rulman sinyalleri.

    Her sınıf farklı bir karakteristik frekansta (devire oranla) darbe dizisi üretir;
    gruplar farklı devirleri temsil eder.
    """
    rng = np.random.default_rng(seed)
    fs, dur = cfg.fs, 4.0
    t = np.arange(int(fs * dur)) / fs
    order = {"N": None, "IR": 5.4, "OR": 3.6, "B": 2.3}
    groups = {"g0": 25.0, "g1": 29.0, "g2": 33.0}
    fres = rng.uniform(2_000, 4_000)
    recs = []
    for g, fr in groups.items():
        for label, k in order.items():
            for u in range(2):
                x = 0.3 * np.sin(2 * np.pi * fr * t) + rng.normal(0, 0.5, t.size)
                if k is not None:
                    fc = k * fr * rng.uniform(0.98, 1.02)
                    for t0 in np.arange(rng.uniform(0, 1 / fc), dur, 1 / fc):
                        idx = int(t0 * fs)
                        n = min(200, t.size - idx)
                        tt = np.arange(n) / fs
                        x[idx:idx + n] += 2.0 * np.exp(-800 * tt) * np.sin(2 * np.pi * fres * tt)
                recs.append(Recording(name, label, g, f"{label}_{g}_{u}", x.astype(np.float32)))
    return recs


# ----------------------------------------------------------------------------- dışarıda tutulan test setleri
# Aşağıdaki veri setleri yöntem geliştirmede HİÇ kullanılmadı (bkz. pilot2/ONKAYIT_A.md).

def _fetch_raw(data_root: str, sub: str, base: str, files: list[str]) -> str:
    dest = os.path.join(data_root, sub)
    for rel in files:
        _download(base + urllib.parse.quote(rel), os.path.join(dest, *rel.split("/")))
    return dest


JNU_BASE = "https://raw.githubusercontent.com/ClarkGableWang/JNU-Bearing-Dataset/main/"
JNU_FS = 50_000
_JNU_LABEL = {"n": "N", "ib": "IR", "ob": "OR", "tb": "B"}
JNU_FILES = [f"{c}{s}_2.csv" for c in ("ib", "ob", "tb") for s in (600, 800, 1000)] + \
            [f"n{s}_3_2.csv" for s in (600, 800, 1000)]


def load_jnu(data_root: str, cfg) -> list[Recording]:
    """JNU (Jiangnan Üniv.): 4 sınıf, her sınıf tek rulman, 600/800/1000 rpm. Bölme: devir (CWRU/SEU gibi)."""
    root = _fetch_raw(data_root, "jnu", JNU_BASE, JNU_FILES)
    recs = []
    for f in JNU_FILES:
        m = re.match(r"^(n|ib|ob|tb)(\d+)_", f)
        x = np.loadtxt(os.path.join(root, f), dtype=np.float64)
        recs.append(Recording("jnu", _JNU_LABEL[m.group(1)], f"{m.group(2)}rpm", f,
                              resample_to(x, JNU_FS, cfg.fs)))
    return recs


HIT_BASE = "https://raw.githubusercontent.com/hitwzc/Bearing-datasets/main/"
HIT_FS = 51_200
HIT_DIRS = {"hit_mfs": "SpectraQuest MFS dataset", "hit_sb": "Self-built dataset"}
_HIT_STATES = ["Normal"] + [f"{c}{d}" for c in ("IR", "OR") for d in (2, 5, 8)]


def _mat_signal(path: str) -> np.ndarray:
    mat = loadmat(path)
    arrs = [np.asarray(v).ravel() for k, v in mat.items()
            if not k.startswith("__") and isinstance(v, np.ndarray) and v.dtype.kind == "f"]
    if not arrs:
        raise ValueError(f"{path}: sayısal sinyal yok")
    return max(arrs, key=len)


def _load_hit(name: str, data_root: str, cfg) -> list[Recording]:
    """HIT-SM (Harbin): N, IR, OR; IR/OR için 3 hasar boyutu (2/5/8 derece) = 3 ayrı birim; 600/900/1200 rpm.
    Bölme (PU gibi, birim düzeyinde): IR/OR'da birim = hasar boyutu; N tek rulman olduğundan birim = devir."""
    files = [f"{HIT_DIRS[name]}/{st}_{rpm}.mat" for st in _HIT_STATES for rpm in (600, 900, 1200)]
    root = _fetch_raw(data_root, "hit", HIT_BASE, files)
    recs = []
    for rel in files:
        st, rpm = os.path.splitext(os.path.basename(rel))[0].split("_")
        label = "N" if st == "Normal" else st[:2]
        group = f"{rpm}rpm" if label == "N" else f"{st}"
        x = _mat_signal(os.path.join(root, *rel.split("/")))
        recs.append(Recording(name, label, group, f"{st}_{rpm}", resample_to(x, HIT_FS, cfg.fs)))
    return recs


LOADERS = {"cwru": load_cwru, "pu": load_pu, "seu": load_seu, "jnu": load_jnu,
           "hit_mfs": lambda r, c: _load_hit("hit_mfs", r, c), "hit_sb": lambda r, c: _load_hit("hit_sb", r, c)}
SPLIT_MODE = {"cwru": "global_group", "seu": "global_group", "pu": "per_class_group",
              "jnu": "global_group", "hit_mfs": "per_class_group", "hit_sb": "per_class_group"}


def load_dataset(name: str, cfg) -> list[Recording]:
    if cfg.synthetic:
        import zlib
        return make_synthetic(name, cfg, seed=zlib.crc32(name.encode()) % 1000)
    return LOADERS[name](cfg.data_root, cfg)


def split_mode(name: str, cfg) -> str:
    return "global_group" if cfg.synthetic else SPLIT_MODE[name]
