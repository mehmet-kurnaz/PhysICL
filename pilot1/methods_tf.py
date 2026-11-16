"""Öğrenen taban çizgileri (TensorFlow / Keras).

- wdcnn_tf   : WDCNN (Zhang vd., 2017) yalnızca hedef makinenin destek pencereleriyle sıfırdan eğitilir.
               "Hiç ön bilgi yoksa ne olur?" alt sınırı.
- protonet_tf: Prototip ağı (Snell vd., 2017), WDCNN gövdesiyle. DİĞER veri setlerinde epizodik
               meta-eğitim yapılır, hedef veri setinde ağırlık güncellemeden prototiple sınıflar.
               Literatürdeki "makineler arası az örnekli meta-öğrenme" çizgisinin temsilcisi.
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

for _gpu in tf.config.list_physical_devices("GPU"):   # PyTorch ile aynı GPU'yu paylaşmak için
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

keras = tf.keras


def wdcnn_trunk(L: int, name: str = "wdcnn") -> keras.Model:
    inp = keras.Input((L, 1))
    x = keras.layers.Conv1D(16, 64, strides=16, padding="same")(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPool1D(2)(x)
    for f in (32, 64, 64, 64):
        x = keras.layers.Conv1D(f, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPool1D(2)(x)
    x = keras.layers.Flatten()(x)
    return keras.Model(inp, x, name=name)


def augment(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Dairesel kaydırma + genlik ölçekleme + Gauss gürültüsü."""
    n, L = X.shape
    shifts = rng.integers(0, L, n)
    idx = (np.arange(L)[None, :] + shifts[:, None]) % L
    Xa = np.take_along_axis(X, idx, axis=1)
    Xa = Xa * rng.uniform(0.8, 1.2, (n, 1)) + rng.normal(0, 0.1, Xa.shape)
    return Xa.astype(np.float32)


class WDCNNMethod:
    def __init__(self, cfg):
        self.cfg = cfg

    def fit_predict(self, Xs, ys, Xq, seed: int = 0):
        tf.keras.utils.set_random_seed(seed)
        rng = np.random.default_rng(seed)
        classes = np.unique(ys)
        y_idx = np.searchsorted(classes, ys)
        trunk = wdcnn_trunk(Xs.shape[1])
        out = keras.layers.Dense(len(classes), activation="softmax")(
            keras.layers.Dropout(0.3)(keras.layers.Dense(100, activation="relu")(trunk.output)))
        model = keras.Model(trunk.input, out)
        model.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy")
        reps = max(1, int(np.ceil(32 / len(Xs))))          # küçük destekleri 32'lik yığına tamamla
        Xrep, yrep = np.tile(Xs, (reps, 1)), np.tile(y_idx, reps)
        for _ in range(self.cfg.wdcnn_steps):
            model.train_on_batch(augment(Xrep, rng)[..., None], yrep)
        proba = model.predict(Xq[..., None], verbose=0, batch_size=256)
        del model, trunk
        return classes[proba.argmax(1)], proba, classes


def _sq_dist(a, b):
    return tf.reduce_sum(tf.square(a[:, None, :] - b[None, :, :]), axis=-1)


class ProtoNetMethod:
    """Kaynak veri setlerinde meta-eğitilir, hedefte ağırlık güncellemeden çalışır."""

    def __init__(self, cfg, out_dir: str):
        self.cfg, self.out_dir = cfg, out_dir
        self.encoder = None
        self.trained_for = None

    def _build(self, L):
        trunk = wdcnn_trunk(L, name="proto_trunk")
        emb = keras.layers.Dense(64)(trunk.output)
        return keras.Model(trunk.input, emb, name="proto_encoder")

    def meta_train(self, target: str, sources: list, L: int, seed: int = 0):
        """sources: hedef DIŞINDAKİ veri setlerinin EpisodeSampler nesneleri."""
        # Dosya adı kaynak veri setlerini içerir: kaynaklar değişirse (ör. PU eklenince) yeniden eğitilir
        src_names = "+".join(sorted(s.name for s in sources))
        path = os.path.join(self.out_dir, f"protonet_target-{target}_src-{src_names}.weights.h5")
        tf.keras.utils.set_random_seed(seed)
        self.encoder = self._build(L)
        self.trained_for = target
        if os.path.exists(path):
            self.encoder.load_weights(path)
            print(f"  protonet: kayıtlı ağırlık yüklendi ({os.path.basename(path)})")
            return
        if not sources:
            raise ValueError("protonet için en az bir kaynak veri seti gerekli")
        opt = keras.optimizers.Adam(1e-3)
        enc = self.encoder

        @tf.function(reduce_retracing=True)
        def step(xs, ys_onehot, xq, yq):
            with tf.GradientTape() as tape:
                zs, zq = enc(xs, training=True), enc(xq, training=True)
                protos = tf.matmul(ys_onehot, zs, transpose_a=True) / tf.reduce_sum(ys_onehot, 0)[:, None]
                logits = -_sq_dist(zq, protos)
                loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(yq, logits, from_logits=True))
            grads = tape.gradient(loss, enc.trainable_variables)
            opt.apply_gradients(zip(grads, enc.trainable_variables))
            return loss

        rng = np.random.default_rng(seed)
        k, q = self.cfg.protonet_shots, self.cfg.protonet_queries
        running = []
        for it in range(self.cfg.protonet_episodes):
            src = sources[rng.integers(len(sources))]
            xs, ys, xq, yq = [], [], [], []
            for ci, c in enumerate(src.classes):
                idx = rng.permutation(np.flatnonzero(src.lab == c))[: k + q]
                xs.append(src.W[idx[:k]]); xq.append(src.W[idx[k:]])
                ys += [ci] * len(idx[:k]); yq += [ci] * len(idx[k:])
            xs, xq = augment(np.concatenate(xs), rng), augment(np.concatenate(xq), rng)
            ys_oh = np.eye(len(src.classes), dtype=np.float32)[ys]
            loss = step(xs[..., None], ys_oh, xq[..., None], np.asarray(yq, np.int32))
            running.append(float(loss))
            if (it + 1) % 250 == 0:
                print(f"  protonet[{target} hariç] epizot {it + 1}: kayıp {np.mean(running[-250:]):.3f}")
        os.makedirs(self.out_dir, exist_ok=True)
        enc.save_weights(path)

    def fit_predict(self, Xs, ys, Xq):
        classes = np.unique(ys)
        zs = self.encoder.predict(Xs[..., None], verbose=0, batch_size=256)
        zq = self.encoder.predict(Xq[..., None], verbose=0, batch_size=256)
        protos = np.stack([zs[ys == c].mean(0) for c in classes])
        logits = -((zq[:, None, :] - protos[None]) ** 2).sum(-1)
        z = np.exp(logits - logits.max(1, keepdims=True))
        proba = z / z.sum(1, keepdims=True)
        return classes[proba.argmax(1)], proba, classes
