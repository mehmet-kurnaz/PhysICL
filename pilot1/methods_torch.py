"""Hazır temel modeller (resmi PyTorch paketleri, DEĞİŞTİRİLMEDEN, yalnızca çıkarım).

- timee      : TimEE, resmi 4'lü ön işleme topluluğu (256/512'ye interpolasyon x {ham, fark})
- timee_raw  : TimEE, topluluk kapalı; pencere doğrudan (1024 örnek) verilir.
               Bu varyant, 256/512'ye küçültmenin yüksek frekanslı rezonans bilgisini
               kaybettirip kaybettirmediğini görmek için eklendi.
- mantisv2_lr: MantisV2 donuk öznitelikleri (katman 2, 'combined') + StandardScaler + LogisticRegression
"""

from __future__ import annotations

import numpy as np


class TimeeMethod:
    def __init__(self, cfg, use_ensemble: bool = True):
        self.cfg, self.use_ensemble = cfg, use_ensemble
        self._clf = None

    def _load(self):
        if self._clf is not None:
            return
        import torch
        from timee import TimeeClassifier
        from timee.transforms import default_ensemble_transforms

        if self.cfg.random_init_fm:  # yalnızca akış testi: ağırlık indirmeden
            from timee.model.model import TimeeModel
            torch.manual_seed(0)
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tr = default_ensemble_transforms() if self.use_ensemble else None
            self._clf = TimeeClassifier(model=TimeeModel(), device=dev, transforms=tr)
        else:
            self._clf = TimeeClassifier.from_pretrained(self.cfg.timee_checkpoint,
                                                        use_ensemble=self.use_ensemble)

    def fit_predict(self, Xs, ys, Xq):
        self._load()
        classes = np.unique(ys)
        pred, proba = self._clf.predict(Xs[:, None, :].astype(np.float32), ys,
                                        Xq[:, None, :].astype(np.float32))
        return pred, proba, classes   # TimEE olasılık sütunları LabelEncoder sırası = np.unique


class MantisLRMethod:
    def __init__(self, cfg):
        self.cfg = cfg
        self._trainer = None

    def _load(self):
        if self._trainer is not None:
            return
        import torch
        from mantis.architecture import MantisV2
        from mantis.trainer import MantisTrainer

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        net = MantisV2(device=dev, return_transf_layer=self.cfg.mantis_layer, output_token="combined")
        if not self.cfg.random_init_fm:
            net = net.from_pretrained(self.cfg.mantis_checkpoint)
        self._trainer = MantisTrainer(device=dev, network=net)

    def _embed(self, X):
        import torch
        import torch.nn.functional as F

        Xt = F.interpolate(torch.tensor(X[:, None, :], dtype=torch.float32),
                           size=self.cfg.mantis_len, mode="linear", align_corners=False)
        return self._trainer.transform(Xt.numpy())

    def fit_predict(self, Xs, ys, Xq):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self._load()
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(self._embed(Xs), ys)
        proba = clf.predict_proba(self._embed(Xq))
        classes = clf.classes_
        return classes[proba.argmax(1)], proba, classes
