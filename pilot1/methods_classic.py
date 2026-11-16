"""Fizik temelli klasik taban çizgisi: zarf spektrumu + en yakın prototip.

Rulman arızaları, rezonansı uyaran periyodik darbeler üretir; Hilbert zarfının spektrumu
bu darbelerin tekrar frekansını (BPFO, BPFI, BSF) gösterir. Model öğrenmez: her sınıfın
destek pencerelerinden ortalama zarf spektrumu (prototip) çıkarılır, sorgu en yakına atanır.
Genel temel modellerin bu basit fiziksel çizgiyi geçip geçmediği pilotun ana sorularından biri.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


class EnvelopeProtoMethod:
    def __init__(self, cfg, temperature: float = 20.0):
        self.cfg, self.tau = cfg, temperature

    def features(self, X: np.ndarray) -> np.ndarray:
        X = X - X.mean(axis=1, keepdims=True)
        env = np.abs(hilbert(X, axis=1))
        env = env - env.mean(axis=1, keepdims=True)
        spec = np.abs(np.fft.rfft(env * np.hanning(X.shape[1]), axis=1))
        freqs = np.fft.rfftfreq(X.shape[1], d=1.0 / self.cfg.fs)
        F = np.log1p(spec[:, (freqs > 0) & (freqs <= self.cfg.env_fmax)])
        F = F - F.mean(axis=1, keepdims=True)
        return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)

    def fit_predict(self, Xs, ys, Xq):
        classes = np.unique(ys)
        Fs, Fq = self.features(Xs), self.features(Xq)
        P = np.stack([Fs[ys == c].mean(0) for c in classes])
        P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-8
        sim = Fq @ P.T
        z = np.exp(self.tau * (sim - sim.max(1, keepdims=True)))
        proba = z / z.sum(1, keepdims=True)
        return classes[proba.argmax(1)], proba, classes
