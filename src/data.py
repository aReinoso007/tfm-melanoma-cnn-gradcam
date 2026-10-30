"""Pipeline de datos del Módulo 1 (Etapa 7): contrato M1 -> M2.

Uso en el Módulo 2:

    from src.data import make_dataset, split_df, load_class_weights
    train_ds = make_dataset("train", "resnet50", training=True)
    val_ds   = make_dataset("val",   "resnet50", training=False)
    model.fit(train_ds, validation_data=val_ds, class_weight=load_class_weights())
    # Predicciones: val/test no se barajan; el orden es el de split_df("val") / split_df("test")

Orden de los pasos (no se debe alterar):
  1. leer y decodificar el JPEG (RGB)
  2. recorte central 600×450 -> 450×450
  3. reducción a 224×224 por área
  4. caché en memoria (uint8)            -- todo lo anterior es fijo e igual para los 3 conjuntos
  5. solo train: barajar y aumentar (src/augmentation.py), con una semilla distinta por imagen y época
  6. normalización oficial de cada arquitectura (keras.applications.<red>.preprocess_input)
  7. lotes y precarga
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from src.augmentation import CLASES, preparar

SPLIT_CSV = Path("data/splits/split_v1.csv")
CLASS_WEIGHTS_JSON = Path("data/splits/class_weights.json")
SPLIT_SHA256_V1 = "98485423d758cae1ed8b7adf64c3e92d31a6afd4d48de7a26e89355238e34b38"
IMAGES_DIR = Path("/content/data/HAM10000/images")      # = paths.images_dir de configs/base.yaml
IMG_SIZE = 224
BATCH_SIZE = 32
SEMILLA = 42
ARQUITECTURAS = ("resnet50", "densenet121", "efficientnetb0")


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_df(split: str, split_csv: str | Path = SPLIT_CSV) -> pd.DataFrame:
    """Filas del split oficial (verificado por su huella), en el orden en que las entrega make_dataset."""
    assert _sha256(split_csv) == SPLIT_SHA256_V1, "El CSV del split no es el split v1 publicado"
    df = pd.read_csv(split_csv)
    assert split in ("train", "val", "test"), split
    df = df[df.split == split].sort_values("image_id").reset_index(drop=True)
    df["label"] = df.dx.map({c: i for i, c in enumerate(CLASES)}).astype("int32")
    return df


def load_class_weights(version: str = "sqrt", path: str | Path = CLASS_WEIGHTS_JSON) -> dict[int, float]:
    """Pesos por clase para model.fit(class_weight=...), calculados solo con train del split v1."""
    d = json.loads(Path(path).read_text())
    assert d["split_sha256"] == SPLIT_SHA256_V1, "class_weights.json no corresponde al split v1"
    return {i: float(d[version][c]) for i, c in enumerate(d["clases"])}


def _cargar(ruta):
    """Pasos 1–3: JPEG -> recorte central cuadrado -> 224×224 por área -> uint8."""
    img = tf.io.decode_jpeg(tf.io.read_file(ruta), channels=3)
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    lado = tf.minimum(h, w)
    img = tf.image.crop_to_bounding_box(img, (h - lado) // 2, (w - lado) // 2, lado, lado)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="area")
    return tf.cast(tf.round(tf.clip_by_value(img, 0, 255)), tf.uint8)


def normalizar(x, arch: str):
    """Paso 6: preprocesado OFICIAL de cada red (entrada float32 RGB en [0, 255])."""
    apps = tf.keras.applications
    f = {"resnet50": apps.resnet50.preprocess_input,          # BGR y resta de la media de ImageNet
         "densenet121": apps.densenet.preprocess_input,       # [0, 1], media y desviación de ImageNet
         "efficientnetb0": apps.efficientnet.preprocess_input  # sin cambios: el modelo normaliza dentro
         }[arch]
    return f(tf.cast(x, tf.float32))


def make_dataset(split: str, arch: str, training: bool, batch_size: int = BATCH_SIZE,
                 seed: int = SEMILLA, images_dir: str | Path = IMAGES_DIR,
                 split_csv: str | Path = SPLIT_CSV, aug_cfg: dict | None = None,
                 one_hot: bool = True, cache: bool = True) -> tf.data.Dataset:
    """Devuelve (imagen normalizada para `arch`, etiqueta) en lotes.

    training=True: baraja en cada época y aplica el aumento selectivo. Solo para split="train".
    training=False: sin barajar ni aumentar; el orden es el de split_df(split).
    """
    assert arch in ARQUITECTURAS, arch
    assert not (training and split != "train"), "El aumento solo se aplica al conjunto de entrenamiento"
    AUTO = tf.data.AUTOTUNE
    df = split_df(split, split_csv)
    rutas = [str(Path(images_dir) / f"{i}.jpg") for i in df.image_id]

    ds = tf.data.Dataset.from_tensor_slices((rutas, df.label.values))
    ds = ds.map(lambda r, y: (_cargar(r), y), num_parallel_calls=AUTO)
    if cache:
        ds = ds.cache()                                        # paso 4: antes del aumento
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
        semillas = tf.data.Dataset.random(seed=seed, rerandomize_each_iteration=True).batch(2)
        ds = tf.data.Dataset.zip((ds, semillas))
        ds = ds.map(lambda xy, s: (preparar(xy[0], xy[1], s, True, aug_cfg), xy[1]), num_parallel_calls=AUTO)
    else:
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTO)
    k = len(CLASES)
    ds = ds.map(lambda x, y: (normalizar(x, arch), tf.one_hot(y, k) if one_hot else y), num_parallel_calls=AUTO)
    return ds.batch(batch_size).prefetch(AUTO)


def desnormalizar(x, arch: str):
    """Inverso aproximado de normalizar(), solo para la comprobación visual (devuelve RGB en [0, 255])."""
    x = tf.cast(x, tf.float32)
    if arch == "resnet50":
        x = (x + tf.constant([103.939, 116.779, 123.68]))[..., ::-1]
    elif arch == "densenet121":
        x = (x * tf.constant([0.229, 0.224, 0.225]) + tf.constant([0.485, 0.456, 0.406])) * 255.0
    return tf.clip_by_value(x, 0, 255)
