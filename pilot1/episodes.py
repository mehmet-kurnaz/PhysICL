"""Sızıntısız few-shot epizot üretimi.

Bir epizotta:
  1. Destek ve sorgu için kayıt grupları ayrılır (bkz. data.py başındaki açıklama).
  2. Sorgu kümesi bir kez seçilir; tüm shot değerlerinde AYNI sorgu kullanılır.
  3. Destek pencereleri iç içe seçilir: 1-shot kümesi 5-shot'un, 5-shot da 10-shot'un alt kümesidir.
Böylece shot değerleri ve yöntemler aynı epizotta eşleştirilmiş olarak karşılaştırılır.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Recording


def zscore(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / sd).astype(np.float32)


def windows_of(sig: np.ndarray, L: int, max_n: int) -> np.ndarray:
    n = len(sig) // L
    W = sig[: n * L].reshape(n, L)
    if n > max_n:  # kayıt boyunca eşit aralıklı alt örnekleme
        W = W[np.linspace(0, n - 1, max_n).astype(int)]
    return W


@dataclass
class Episode:
    classes: list
    support_pool: dict      # sınıf -> (Kmax, L) karıştırılmış destek pencereleri
    Xq: np.ndarray
    yq: np.ndarray
    info: dict

    def support(self, k: int):
        Xs = np.concatenate([self.support_pool[c][:k] for c in self.classes])
        ys = np.concatenate([[c] * len(self.support_pool[c][:k]) for c in self.classes])
        return Xs, ys


class EpisodeSampler:
    def __init__(self, recs: list[Recording], window: int, max_windows: int, split_mode: str):
        if not recs:
            raise ValueError("Boş veri seti")
        self.name = recs[0].dataset
        self.split_mode = split_mode
        self.W, self.lab, self.grp, self.rec_id = [], [], [], []
        for i, r in enumerate(recs):
            W = windows_of(r.signal, window, max_windows)
            self.W.append(W)
            self.lab += [r.label] * len(W)
            self.grp += [r.group] * len(W)
            self.rec_id += [i] * len(W)
        self.W = zscore(np.concatenate(self.W))
        self.lab, self.grp, self.rec_id = map(np.asarray, (self.lab, self.grp, self.rec_id))
        self.classes = sorted(set(self.lab.tolist()))

    def describe(self) -> str:
        parts = [f"{c}:{int((self.lab == c).sum())}" for c in self.classes]
        return f"{self.name}: {len(self.W)} pencere, gruplar={sorted(set(self.grp.tolist()))}, " + " ".join(parts)

    def _support_groups(self, rng) -> dict:
        if self.split_mode == "global_group":
            common = set.intersection(*[set(self.grp[self.lab == c].tolist()) for c in self.classes])
            # sorgu tarafında da her sınıfın en az bir grubu kalmalı
            ok = [g for g in sorted(common)
                  if all(((self.lab == c) & (self.grp != g)).any() for c in self.classes)]
            if not ok:
                raise ValueError(f"{self.name}: destek/sorgu için uygun ortak grup yok")
            g = ok[rng.integers(len(ok))]
            return {c: g for c in self.classes}
        # per_class_group: her sınıf kendi birimlerinden birini desteğe ayırır
        out = {}
        for c in self.classes:
            gs = sorted(set(self.grp[self.lab == c].tolist()))
            if len(gs) < 2:
                raise ValueError(f"{self.name}: '{c}' sınıfında en az 2 birim gerekli")
            out[c] = gs[rng.integers(len(gs))]
        return out

    def sample(self, rng: np.random.Generator, kmax: int, q_per_class: int) -> Episode:
        sg = self._support_groups(rng)
        pool, Xq, yq = {}, [], []
        for c in self.classes:
            s_idx = np.flatnonzero((self.lab == c) & (self.grp == sg[c]))
            q_idx = np.flatnonzero((self.lab == c) & (self.grp != sg[c]))
            s_idx = rng.permutation(s_idx)[:kmax]
            q_idx = rng.permutation(q_idx)[:q_per_class]
            assert not set(self.rec_id[s_idx]) & set(self.rec_id[q_idx]), "sızıntı!"
            pool[c] = self.W[s_idx]
            Xq.append(self.W[q_idx])
            yq += [c] * len(q_idx)
        return Episode(self.classes, pool, np.concatenate(Xq), np.asarray(yq),
                       {"support_groups": sg})
