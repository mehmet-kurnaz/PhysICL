"""TimEE'ye fizikten esinlenen girdi görünümleri.

  ham       : pencere olduğu gibi (pilot 1'deki timee_raw)
  zarf      : 500 Hz üstü süzülmüş sinyalin Hilbert zarfı. Mil harmoniklerini atar, rezonansın
              darbe darbe genlik değişimini (yani BPFO/BPFI ritmini) bırakır. Zarf spektrumu
              yönteminin sinyal düzeyindeki karşılığı; ancak sınıflandırmayı TimEE yapar.
  ham+zarf  : iki görünümün sınıf olasılıklarının ortalaması

Model aynı kalır, ağırlık güncellenmez; yalnızca girdinin hangi biçimde verildiği değişir.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

VIEWS = ("ham", "zarf", "ham+zarf")
_SOS = {}
_CLF = {}


def _z(X):
    return ((X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)).astype(np.float32)


def envelope(X: np.ndarray, fs: int = 12_000, hp: float = 500.0) -> np.ndarray:
    key = (fs, hp)
    if key not in _SOS:
        _SOS[key] = butter(4, hp, btype="highpass", fs=fs, output="sos")
    Y = sosfiltfilt(_SOS[key], X, axis=1)
    return _z(np.abs(hilbert(Y, axis=1)))


def _classifier(checkpoint: str):
    if checkpoint not in _CLF:                 # aynı ağırlık birden çok görünümde tekrar yüklenmesin
        from timee import TimeeClassifier
        _CLF[checkpoint] = TimeeClassifier.from_pretrained(checkpoint, use_ensemble=False)
    return _CLF[checkpoint]


class TimeeViewMethod:
    def __init__(self, cfg, checkpoint: str, view: str):
        if view not in VIEWS:
            raise ValueError(f"bilinmeyen görünüm: {view} ({', '.join(VIEWS)})")
        self.cfg, self.checkpoint, self.view = cfg, checkpoint, view

    def fit_predict(self, Xs, ys, Xq):
        clf = _classifier(self.checkpoint)
        classes = np.unique(ys)
        parts = self.view.split("+")
        probs = []
        for p in parts:
            A, B = (Xs, Xq) if p == "ham" else (envelope(Xs, self.cfg.fs), envelope(Xq, self.cfg.fs))
            probs.append(clf.predict(A[:, None, :].astype(np.float32), ys, B[:, None, :].astype(np.float32))[1])
        proba = np.mean(probs, axis=0)
        return classes[proba.argmax(1)], proba, classes
