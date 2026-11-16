"""TimEE'nin sentetik önselle ince ayarı.

Model gerçek veri GÖRMEZ. Her adımda simülatörden (veya kontrol için genel önselden) yeni
sanal makineler üretilir; TimEE desteği görüp sorguları sınıflandırır, kayıp yalnızca sorgu
logit'leri üzerinden hesaplanır (resmi çıkarımdaki gibi ilk C logit kullanılır).

Model seçimi gerçek veriye BAKILMADAN yapılır: değerlendirmede son ağırlık ("son") kullanılır.
Sentetik doğrulama yalnızca eğitimi izlemek içindir.

Kesilirse aynı ayarlarla yeniden başlatın: "son" klasöründen kaldığı adımdan devam eder.
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def _sampler(prior: str, karma_orani: float = 0.5):
    """fizik | genel | karma (her epizot olasılıkla fizik ya da genel önselden)."""
    if prior == "fizik":
        from .simulator import sample_episode
        return sample_episode
    if prior == "genel":
        from .generic_prior import sample_episode
        return sample_episode
    if prior == "karma":
        from .generic_prior import sample_episode as gen
        from .simulator import sample_episode as phys

        def mixed(rng, *a, **k):
            return (phys if rng.random() < karma_orani else gen)(rng, *a, **k)
        return mixed
    raise ValueError(f"bilinmeyen önsel: {prior} (fizik | genel | karma)")


class EpisodeStream(IterableDataset):
    """Sonsuz epizot yığını akışı. Bir yığındaki epizotlar aynı (C, k, q) şeklini paylaşır."""

    def __init__(self, prior: str, batch: int, seed: int, L: int = 1024, max_series: int = 160,
                 karma_orani: float = 0.5, opts: dict | None = None):
        self.prior, self.batch, self.seed, self.L, self.max_series = prior, batch, seed, L, max_series
        self.karma_orani = karma_orani
        self.opts = dict(opts or {})
        self.max_way = 8 if self.opts.get("mil", True) else 5   # mil arızaları kapalıysa 5 sınıf

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self.seed, wid])
        fn = _sampler(self.prior, self.karma_orani)
        while True:
            C = int(rng.integers(2, self.max_way + 1))
            k = int(rng.integers(1, 11))
            q = int(rng.integers(5, 16))
            q = max(2, min(q, self.max_series // C - k))      # bellek: C*(k+q) <= max_series
            Xb, yb = [], []
            for _ in range(self.batch):
                X, y, ns, _ = fn(rng, self.L, n_way=(C, C), k_shot=(k, k), q_per_class=(q, q),
                                 **({"opts": self.opts} if self.opts else {}))
                Xb.append(X); yb.append(y)
            yield torch.from_numpy(np.stack(Xb)), torch.from_numpy(np.stack(yb)), ns, C


def load_timee(checkpoint: str, random_init: bool = False):
    from timee.model.model import TimeeModel
    model = TimeeModel()
    if random_init:
        return model
    from safetensors.torch import load_file
    from timee.classifier import _resolve_weights_path
    model.load_state_dict(load_file(_resolve_weights_path(checkpoint)))
    return model


def save_checkpoint(model, path, state: dict | None = None):
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    tmp = os.path.join(path, "model.safetensors.tmp")
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, tmp)
    os.replace(tmp, os.path.join(path, "model.safetensors"))
    if state is not None:
        torch.save(state, os.path.join(path, "train_state.pt.tmp"))
        os.replace(os.path.join(path, "train_state.pt.tmp"), os.path.join(path, "train_state.pt"))


def make_val_set(n: int = 32, seed: int = 999):
    """Sabit sentetik doğrulama: fizik önselinden, 4 sınıf, 5-shot, farklı birim ve koşul."""
    from .simulator import sample_episode
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        X, y, ns, _ = sample_episode(rng, n_way=(4, 4), k_shot=(5, 5), q_per_class=(15, 15),
                                     p_cross_cond=1.0, p_cross_unit=1.0)
        out.append((X, y, ns))
    return out


@torch.no_grad()
def evaluate_val(model, val, dev):
    model.eval()
    acc = []
    for X, y, ns in val:
        y_in = y.copy()
        y_in[ns:] = 0
        logits = model(x=torch.from_numpy(X)[None].to(dev), y=torch.from_numpy(y_in)[None].to(dev),
                       eval_pos=ns)[0, :, :4]
        acc.append((logits.argmax(-1).cpu().numpy() == y[ns:]).mean())
    model.train()
    return float(np.mean(acc))


ABLASYONLAR = {"birim_yok": ("birim", False), "kosul_yok": ("kosul", False), "bant_yok": ("bant", False),
               "golge_yok": ("golge", False), "mil_yok": ("mil", False)}


def sim_opts(s: dict) -> dict:
    """AYARLAR'dan simülatör seçenekleri: "simulator_surum" (v0|v1) ve "ablasyon" (virgüllü liste)."""
    o = {}
    if s.get("simulator_surum", "v0") != "v0":
        o["surum"] = s["simulator_surum"]
    for a in [x.strip() for x in str(s.get("ablasyon", "")).split(",") if x.strip()]:
        if a not in ABLASYONLAR:
            raise ValueError(f"bilinmeyen ablasyon: {a} ({', '.join(ABLASYONLAR)})")
        k, v = ABLASYONLAR[a]
        o[k] = v
    return o


def run(s: dict):
    prior = s["onsel"]
    test = s["test"]
    etiket = s.get("etiket") or prior               # klasör adı: ince_ayar_<etiket>
    opts = sim_opts(s)
    torch.manual_seed(int(s["seed"]))
    out = os.path.join(s["out"], f"ince_ayar_{etiket}")
    last = os.path.join(out, "son")
    os.makedirs(out, exist_ok=True)
    steps = 40 if test else int(s["adim_sayisi"])
    batch = 2 if test else int(s["batch"])
    workers = 0 if test else int(s["workers"])
    eval_every = 20 if test else int(s["kayit_araligi"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"etiket={etiket}  önsel={prior}  seçenek={opts or '-'}  tohum={s['seed']}  adım={steps}  yığın={batch}  cihaz={dev}  işçi={workers}")

    model = load_timee(s["timee_checkpoint"], random_init=test)
    start = 0
    state = None
    if os.path.exists(os.path.join(last, "train_state.pt")):
        model = load_timee(last)
        state = torch.load(os.path.join(last, "train_state.pt"), map_location="cpu", weights_only=False)
        start = state["step"]
        print(f"devam: adım {start}'den")
    model.to(dev).train()

    lr, warm = float(s["lr"]), min(300, steps // 10)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(i, steps) / steps))))
    use_amp = dev.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    if state is not None:
        opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])

    stream = EpisodeStream(prior, batch, seed=int(s["seed"]) * 100_003 + start,
                           karma_orani=float(s.get("karma_fizik_orani", 0.5)), opts=opts)
    loader = DataLoader(stream, batch_size=None, num_workers=workers,
                        persistent_workers=workers > 0, prefetch_factor=4 if workers else None)
    val = make_val_set(8 if test else 32)
    log_path = os.path.join(out, "log.csv")
    if start == 0:
        with open(log_path, "w") as f:
            f.write("step,loss,train_acc,val_acc_fizik,lr,sec_per_step\n")
        v0 = evaluate_val(model, val, dev)
        print(f"başlangıç sentetik doğrulama (fizik, 4 sınıf, 5-shot, farklı birim): {100 * v0:.1f}%")
        with open(log_path, "a") as f:
            f.write(f"0,,,{v0:.4f},,\n")

    run_loss, run_acc, t0 = [], [], time.time()
    step, skipped = start, 0
    for X, y, ns, C in loader:
        if step >= steps:
            break
        X, y = X.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=use_amp):
            y_in = y.clone()
            y_in[:, ns:] = 0                          # sorgu etiketleri modele verilmez (zaten yok sayılıyor)
            logits = model(x=X, y=y_in, eval_pos=ns)[..., :C]
        yq = y[:, ns:]
        loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, C), yq.reshape(-1))
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):                  # sayısal taşma: bu yığını atla, ağırlığa dokunma
            skipped += 1
            print(f"  uyarı: adım {step + 1} kayıp sonsuz/NaN, atlandı (toplam {skipped})")
            if skipped > 50:
                raise RuntimeError("çok fazla NaN adımı; öğrenme oranını düşürün")
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gnorm):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            scaler.update()
            continue
        scaler.step(opt); scaler.update(); sched.step()
        step += 1
        run_loss.append(loss.item())
        run_acc.append((logits.argmax(-1) == yq).float().mean().item())

        if step % 50 == 0 or step == steps or test:
            sps = (time.time() - t0) / len(run_loss)
            eta = (steps - step) * sps / 60
            print(f"  adım {step}/{steps}  kayıp {np.mean(run_loss):.3f}  eğitim doğr. {100 * np.mean(run_acc):.1f}%  "
                  f"{sps:.2f} sn/adım  kalan ~{eta:.0f} dk")
        if step % eval_every == 0 or step == steps:
            v = evaluate_val(model, val, dev)
            print(f"  >> sentetik doğrulama: {100 * v:.1f}%  (kayıt: adim_{step:05d})")
            with open(log_path, "a") as f:
                f.write(f"{step},{np.mean(run_loss):.4f},{np.mean(run_acc):.4f},{v:.4f},"
                        f"{sched.get_last_lr()[0]:.2e},{(time.time() - t0) / len(run_loss):.3f}\n")
            st = {"step": step, "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict()}
            save_checkpoint(model, os.path.join(out, f"adim_{step:05d}"))
            save_checkpoint(model, last, st)
            run_loss, run_acc, t0 = [], [], time.time()

    with open(os.path.join(out, "ayarlar.json"), "w") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)
    print(f"bitti. Son ağırlık: {os.path.abspath(last)}")
