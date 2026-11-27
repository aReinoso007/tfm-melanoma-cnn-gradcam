"""Grad-CAM y Grad-CAM++ para los modelos del Módulo 2 (OE5).

Los modelos tienen el backbone "aplanado" (src/models.py), así que la última capa convolucional se
pide por nombre con `model.get_layer(...)`. La cabeza es GAP -> Dropout -> Dense(7, softmax).

Decisiones:

  * El gradiente se calcula sobre el LOGIT (Dense sin la softmax), no sobre la probabilidad. Con la
    softmax saturada, el gradiente de la clase predicha casi se anula; y Grad-CAM++ está definido
    sobre puntuaciones previas a la softmax. No se crea ningún modelo nuevo: se toman la salida de
    la capa convolucional y la de `head_gap`, y los logits se calculan con el kernel y el sesgo de
    `head_softmax`. Los pesos entrenados no se tocan.
  * El modelo se llama siempre con `training=False` (Dropout inactivo, BatchNorm en inferencia).
  * La imagen pasa por la MISMA `normalizar()` de src/data.py que se usó al entrenar y evaluar.
  * El mapa tiene el tamaño de la entrada (224×224), pasa por ReLU y se reescala a [0, 1]. Si el mapa
    es todo ceros se devuelve de ceros (no NaN).
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from src.data import IMAGES_DIR, _cargar, normalizar

METODOS = ("gradcam", "gradcam++")
CAPAS_CONV = {                       # = configs/base.yaml: gradcam.last_conv_layer
    "resnet50": "conv5_block3_out",
    "densenet121": "relu",
    "efficientnetb0": "top_activation",
}


def cargar_imagen(image_id: str, arch: str, images_dir: str | Path = IMAGES_DIR):
    """Devuelve (uint8 224×224×3 para mostrar, float32 224×224×3 normalizada para la red)."""
    uint8 = _cargar(str(Path(images_dir) / f"{image_id}.jpg")).numpy()
    return uint8, normalizar(uint8, arch).numpy()


def _normalizar_mapa(m: np.ndarray) -> np.ndarray:
    m = m - m.min()
    tope = m.max()
    if not np.isfinite(tope) or tope <= 1e-12:
        return np.zeros_like(m, dtype=np.float32)
    return (m / tope).astype(np.float32)


class Explicador:
    """Calcula mapas de un modelo concreto. Se crea una vez y se reutiliza para todas las imágenes."""

    def __init__(self, model: keras.Model, arch: str, capa: str | None = None):
        self.arch = arch
        self.capa = capa or CAPAS_CONV[arch]
        nombres = {l.name for l in model.layers}
        for n in (self.capa, "head_gap", "head_softmax"):
            if n not in nombres:
                raise ValueError(f"{arch}: el modelo no tiene la capa '{n}'")
        salida_conv = model.get_layer(self.capa).output
        if len(salida_conv.shape) != 4:
            raise ValueError(f"{self.capa} no es una capa convolucional (salida {salida_conv.shape})")
        self._parcial = keras.Model(model.input, [salida_conv, model.get_layer("head_gap").output])
        denso = model.get_layer("head_softmax")
        self._k = tf.constant(np.asarray(denso.kernel), dtype=tf.float32)
        self._b = tf.constant(np.asarray(denso.bias), dtype=tf.float32)
        self.tam = int(model.input_shape[1])

    @staticmethod
    def _lote(x) -> tf.Tensor:
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return x[None] if len(x.shape) == 3 else x

    def logits(self, x) -> np.ndarray:
        _, agrupado = self._parcial(self._lote(x), training=False)
        return (tf.matmul(agrupado, self._k) + self._b).numpy()[0]

    def probabilidades(self, x) -> np.ndarray:
        return tf.nn.softmax(self.logits(x)).numpy()

    def mapa(self, x, clase: int, metodo: str = "gradcam") -> np.ndarray:
        """Mapa (tam×tam, float32, [0, 1]) de la clase `clase` para la imagen ya normalizada `x`."""
        if metodo not in METODOS:
            raise ValueError(f"método desconocido: {metodo}")
        x = self._lote(x)
        with tf.GradientTape() as cinta:
            A, agrupado = self._parcial(x, training=False)
            puntuacion = (tf.matmul(agrupado, self._k) + self._b)[:, clase]
        g = cinta.gradient(puntuacion, A)
        if metodo == "gradcam":
            pesos = tf.reduce_mean(g, axis=(1, 2))
        else:
            g2, g3 = g ** 2, g ** 3
            denom = 2.0 * g2 + tf.reduce_sum(A, axis=(1, 2), keepdims=True) * g3
            denom = tf.where(denom != 0.0, denom, tf.ones_like(denom))
            alfa = g2 / denom
            pesos = tf.reduce_sum(alfa * tf.nn.relu(g), axis=(1, 2))
        cam = tf.nn.relu(tf.reduce_sum(A * pesos[:, None, None, :], axis=-1, keepdims=True))
        cam = tf.image.resize(cam, [self.tam, self.tam], method="bilinear")[0, :, :, 0].numpy()
        return _normalizar_mapa(cam)


# ---------------------------------------------------------------------------
# Prueba de cordura de Adebayo et al. (2018)
# ---------------------------------------------------------------------------

def correlacion_rangos(a: np.ndarray, b: np.ndarray) -> float:
    """Correlación de rangos de Spearman entre dos mapas (NaN si alguno es constante)."""
    import pandas as pd
    ra = pd.Series(np.ravel(a)).rank().to_numpy()
    rb = pd.Series(np.ravel(b)).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _capas_aleatorizables(model: keras.Model) -> list:
    return [l for l in model.layers
            if not l.name.startswith("head_")
            and not isinstance(l, keras.layers.BatchNormalization)
            and any("kernel" in w.name or "bias" in w.name for w in l.weights)]


@contextlib.contextmanager
def base_aleatorizada(model: keras.Model, semilla: int, ultimas: int | None = None):
    """Aleatoriza EN SITIO los kernels de la base (He normal) y pone los sesgos a cero; al salir del
    bloque restaura los pesos originales exactamente.

    `ultimas=None`: toda la base. `ultimas=k`: solo las k capas con pesos más cercanas a la salida
    (aleatorización en cascada de Adebayo). La cabeza, las BatchNorm y las capas de normalización de
    entrada no se tocan. No se clona el modelo: `clone_model` no reproduce exactamente EfficientNetB0
    (capa Normalization), y con la copia la fracción 0 no daba correlación 1.
    """
    guardado = model.get_weights()
    capas = _capas_aleatorizables(model)
    if ultimas is not None:
        capas = capas[len(capas) - ultimas:] if ultimas > 0 else []
    rng = np.random.default_rng(semilla)
    try:
        for capa in capas:
            for w in capa.weights:
                if "kernel" in w.name:
                    forma = tuple(w.shape)
                    fan_in = max(1, int(np.prod(forma[:-1])))
                    w.assign(rng.normal(0.0, np.sqrt(2.0 / fan_in), size=forma).astype("float32"))
                elif "bias" in w.name:
                    w.assign(np.zeros(tuple(w.shape), dtype="float32"))
        yield model
    finally:
        model.set_weights(guardado)


def n_capas_base_con_pesos(model: keras.Model) -> int:
    return len(_capas_aleatorizables(model))


def prueba_cordura(model: keras.Model, arch: str, imagenes_norm: list[np.ndarray], clase: int,
                   fracciones=(0.0, 0.25, 0.5, 0.75, 1.0), semilla: int = 0) -> dict:
    """Correlación de rangos media entre el mapa del modelo original y el del modelo con la parte
    superior de la base aleatorizada, para cada método. Con fracción 0 debe dar ≈ 1.

    Devuelve {"rho": {(método, fracción): media}, "vacios": {(método, fracción): n}}. Un mapa
    aleatorizado vacío (todo ceros) no tiene correlación definida: se excluye de la media y se cuenta.
    """
    expl = Explicador(model, arch)
    total = n_capas_base_con_pesos(model)
    originales = {m: [expl.mapa(x, clase, m) for x in imagenes_norm] for m in METODOS}
    resultado: dict = {"rho": {}, "vacios": {}}
    for f in fracciones:
        k = int(round(f * total))
        with base_aleatorizada(model, semilla, ultimas=k):
            for metodo in METODOS:
                rhos = [correlacion_rangos(o, expl.mapa(x, clase, metodo))
                        for o, x in zip(originales[metodo], imagenes_norm)]
                validos = [r for r in rhos if r == r]
                resultado["rho"][(metodo, f)] = float(np.mean(validos)) if validos else float("nan")
                resultado["vacios"][(metodo, f)] = len(rhos) - len(validos)
    return resultado


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def superponer(imagen_uint8: np.ndarray, mapa: np.ndarray, alfa: float = 0.4) -> np.ndarray:
    """Mapa en color (jet) sobre la imagen original. Devuelve uint8."""
    import matplotlib
    color = matplotlib.colormaps["jet"](np.clip(mapa, 0, 1))[..., :3] * 255.0
    mezcla = (1 - alfa) * imagen_uint8.astype("float32") + alfa * color
    return np.clip(mezcla, 0, 255).astype("uint8")


def figura_caso(imagen_uint8: np.ndarray, filas: list[tuple[str, dict[str, np.ndarray]]], titulo: str):
    """Una fila por clase explicada: imagen original | Grad-CAM | Grad-CAM++ superpuestos."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(len(filas), 3, figsize=(9, 3.2 * len(filas) + 0.6), squeeze=False)
    for i, (etiqueta, mapas) in enumerate(filas):
        ejes[i, 0].imshow(imagen_uint8)
        ejes[i, 0].set_title(f"original · mapas de «{etiqueta}»" if i == 0 else f"mapas de «{etiqueta}»",
                             fontsize=9)
        ejes[i, 1].imshow(superponer(imagen_uint8, mapas["gradcam"]))
        ejes[i, 1].set_title("Grad-CAM", fontsize=9)
        ejes[i, 2].imshow(superponer(imagen_uint8, mapas["gradcam++"]))
        ejes[i, 2].set_title("Grad-CAM++", fontsize=9)
    for e in ejes.ravel():
        e.axis("off")
    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout()
    return fig
