"""
tfm_common.py — Código compartido por los tres módulos del TFM
================================================================
Clasificación de lesiones cutáneas (HAM10000) con ResNet50, DenseNet121 y
EfficientNetB0 + explicabilidad con Grad-CAM.

Este archivo es el "contrato" entre los tres notebooks. Cualquier cambio en
constantes, split o preprocesamiento debe acordarse en grupo y versionarse
(subir SPLIT_VERSION si cambia el split).

Secciones y responsable principal:
  1. Constantes y rutas ............................ común
  2. Datos: descarga, redimensionado, split ........ Módulo 1 (OBSOLETA: ver abajo)
  3. Pipeline tf.data y augmentation ............... Módulo 1 (OBSOLETA: ver abajo)
  4. Modelos y congelado de capas .................. Módulo 2
  5. Métricas ...................................... Módulo 2
  6. Utilidades (semillas, JSON, MLflow, gráficos).. común

IMPORTANTE (acuerdo del grupo, octubre de 2026): los datos de los Módulos 2 y 3 se
cargan SOLO con el código del Módulo 1, que es el contrato M1 -> M2 de
docs/02_flujo_de_trabajo.md:
    from src.data import make_dataset, split_df, load_class_weights
    split oficial: data/splits/split_v1.csv · pesos: data/splits/class_weights.json
Las secciones 2 y 3 de este archivo no siguen las decisiones del Módulo 1 (aplastan
la imagen a cuadrado, cambian tono y saturación, no normalizan los pesos suavizados
y regeneran el split), así que no deben usarse para entrenar ni evaluar.
"""
from __future__ import annotations

import glob
import json
import os
import random
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# =============================================================================
# 1. CONSTANTES Y RUTAS
# =============================================================================
# Orden alfabético FIJO: el índice de cada clase es la etiqueta numérica.
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES_ES = {
    "akiec": "Queratosis actínica / carcinoma intraepitelial",
    "bcc": "Carcinoma basocelular",
    "bkl": "Lesión queratósica benigna",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Nevus melanocítico",
    "vasc": "Lesión vascular",
}
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
MEL_IDX = CLASS_TO_IDX["mel"]

IMG_SIZE = 224
RESIZE_MODE = "stretch"          # "stretch" | "center_crop" (decisión del Módulo 1)
N_IMAGES_EXPECTED = 10015

SPLIT_VERSION = "v1"
SPLIT_SEED = 42
SPLIT_N_FOLDS = 20               # 20 folds de ~5 % -> 3 test, 3 val, 14 train = 15/15/70

SEEDS = [42, 123, 2024]          # = training.seeds de configs/base.yaml
ARCHS = ["resnet50", "densenet121", "efficientnetb0"]
MIN_MEL_SENSITIVITY = 0.80       # requisito mínimo de selección (validación)

# Última capa convolucional (objetivo de Grad-CAM). Se usa como referencia;
# get_last_conv_layer_name() la detecta automáticamente.
LAST_CONV_LAYER = {
    "resnet50": "conv5_block3_out",
    "densenet121": "relu",
    "efficientnetb0": "top_activation",
}
# Prefijos desde los que se descongela en cada etapa del fine-tuning progresivo.
UNFREEZE_STAGES = {
    "resnet50": ["conv5_block1", "conv4_block1"],
    "densenet121": ["conv5_block1", "conv4_block1"],
    "efficientnetb0": ["block6a", "block5a"],
}

KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"
DAGSHUB_OWNER = "aReinoso007"
DAGSHUB_REPO = "tfm-melanoma-cnn-gradcam"
# Nombres de experimentos = mlflow.experiments de configs/base.yaml (docs/03_mlflow_convenciones.md)
MLFLOW_EXPERIMENTS = {
    "m1": "m1-datos",
    "m2_train": "m2-entrenamiento",
    "m2_test": "m2-evaluacion-test",
    "m3": "m3-gradcam",
    "sandbox": "sandbox",
}
MLFLOW_EXPERIMENT = MLFLOW_EXPERIMENTS["m2_train"]    # el experimento principal del Módulo 2


def get_paths(drive_root: str = "/content/drive/MyDrive/TFM",
              local_root: str = "/content/tfm_local") -> SimpleNamespace:
    """Rutas estándar. Drive = persistente y compartido; local = rápido (sesión)."""
    p = SimpleNamespace(
        drive_root=Path(drive_root),
        drive_data=Path(drive_root) / "data",
        drive_models=Path(drive_root) / "models",
        drive_results=Path(drive_root) / "results",
        drive_figures=Path(drive_root) / "figures",
        local_root=Path(local_root),
        local_raw=Path(local_root) / "raw",
        local_images=Path(local_root) / f"images_{IMG_SIZE}_{RESIZE_MODE}",
        local_ckpt=Path(local_root) / "ckpt",
    )
    for v in vars(p).values():
        v.mkdir(parents=True, exist_ok=True)
    p.images_zip = p.drive_data / f"ham10000_images_{IMG_SIZE}_{RESIZE_MODE}.zip"
    p.metadata_csv = p.drive_data / "HAM10000_metadata.csv"
    p.split_csv = p.drive_data / f"split_{SPLIT_VERSION}.csv"
    p.class_weights_json = p.drive_data / f"class_weights_{SPLIT_VERSION}.json"
    p.representative_json = p.drive_results / "representative_models.json"
    return p


# =============================================================================
# 2. DATOS: DESCARGA, REDIMENSIONADO Y SPLIT  (Módulo 1)
#    OBSOLETA: usar data/splits/split_v1.csv y src/data.py (split_df, load_class_weights)
# =============================================================================
def download_ham10000():
    """Descarga HAM10000 desde Kaggle con kagglehub.
    Devuelve (ruta_metadata_csv, {image_id: ruta_jpg})."""
    import kagglehub
    root = kagglehub.dataset_download(KAGGLE_DATASET)
    meta = glob.glob(os.path.join(root, "**", "HAM10000_metadata.csv"), recursive=True)
    if not meta:
        raise FileNotFoundError(f"No se encontró HAM10000_metadata.csv en {root}")
    images = {}
    for f in glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True):
        images.setdefault(Path(f).stem, f)   # el dataset trae carpetas duplicadas
    return meta[0], images


def _resize_one(args):
    from PIL import Image
    src, dst = args
    if os.path.exists(dst):
        return
    img = Image.open(src).convert("RGB")
    if RESIZE_MODE == "center_crop":
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    img.save(dst, quality=95)


def resize_images(image_map: dict, out_dir: Path, workers: int = 8):
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(src, str(out_dir / f"{iid}.jpg")) for iid, src in image_map.items()]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(_resize_one, jobs))
    return len(jobs)


def zip_dir(src_dir: Path, zip_path: Path):
    tmp = Path("/content") / zip_path.name
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:   # jpg ya comprimido
        for f in sorted(src_dir.glob("*.jpg")):
            z.write(f, f.name)
    shutil.copy(tmp, zip_path)
    tmp.unlink()


def ensure_local_images(paths: SimpleNamespace):
    """Deja las imágenes redimensionadas en disco local (rápido).
    Requiere que el Módulo 1 haya subido el zip a Drive."""
    n = len(list(paths.local_images.glob("*.jpg")))
    if n >= N_IMAGES_EXPECTED:
        return n
    if not paths.images_zip.exists():
        raise FileNotFoundError(
            f"No existe {paths.images_zip}. Ejecutar antes el notebook del Módulo 1.")
    local_zip = paths.local_root / paths.images_zip.name
    shutil.copy(paths.images_zip, local_zip)
    with zipfile.ZipFile(local_zip) as z:
        z.extractall(paths.local_images)
    local_zip.unlink()
    return len(list(paths.local_images.glob("*.jpg")))


def make_split(meta: pd.DataFrame) -> pd.DataFrame:
    """Split 70/15/15 estratificado por clase y agrupado por lesion_id
    (ninguna lesión aparece en más de un subconjunto). Determinista."""
    from sklearn.model_selection import StratifiedGroupKFold
    meta = meta.reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=SPLIT_N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    fold = np.empty(len(meta), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(meta, meta["dx"], groups=meta["lesion_id"])):
        fold[idx] = k
    split = np.where(fold < 3, "test", np.where(fold < 6, "val", "train"))
    out = meta[["image_id", "lesion_id", "dx"]].copy()
    out["split"] = split
    return out


def check_no_leakage(df: pd.DataFrame):
    n = df.groupby("lesion_id")["split"].nunique()
    leaks = int((n > 1).sum())
    assert leaks == 0, f"{leaks} lesiones aparecen en más de un subconjunto"
    return True


def load_split(paths: SimpleNamespace, with_metadata: bool = True) -> pd.DataFrame:
    """Carga el split oficial compartido y añade ruta local y etiqueta numérica."""
    df = pd.read_csv(paths.split_csv)
    if with_metadata and paths.metadata_csv.exists():
        meta = pd.read_csv(paths.metadata_csv)
        df = df.merge(meta.drop(columns=["lesion_id", "dx"]), on="image_id", how="left")
    df["label"] = df["dx"].map(CLASS_TO_IDX).astype(int)
    df["path"] = df["image_id"].map(lambda i: str(paths.local_images / f"{i}.jpg"))
    check_no_leakage(df)
    return df


def compute_class_weights(train_df: pd.DataFrame, mode: str = "balanced") -> dict | None:
    """mode: 'balanced' (n / (K * n_k)), 'sqrt' (versión suavizada) o None."""
    if mode in (None, "none"):
        return None
    counts = train_df["label"].value_counts().reindex(range(NUM_CLASSES)).fillna(0).values
    w = len(train_df) / (NUM_CLASSES * np.maximum(counts, 1))
    if mode == "sqrt":
        w = np.sqrt(w)
    return {int(i): float(v) for i, v in enumerate(w)}


def oversample(train_df: pd.DataFrame, factors: dict | None, seed: int = 42) -> pd.DataFrame:
    """Replica filas de clases minoritarias. factors = {'df': 4, 'vasc': 4, ...}."""
    if not factors:
        return train_df
    parts = [train_df]
    for cls, f in factors.items():
        if f > 1:
            sub = train_df[train_df["dx"] == cls]
            parts += [sub] * (int(f) - 1)
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


# =============================================================================
# 3. PIPELINE tf.data Y AUGMENTATION  (Módulo 1)
#    OBSOLETA: usar src/data.py (make_dataset) y src/augmentation.py
# =============================================================================
_CAFFE_MEAN = [103.939, 116.779, 123.68]
_TORCH_MEAN = [0.485, 0.456, 0.406]
_TORCH_STD = [0.229, 0.224, 0.225]


def preprocess(x, arch: str):
    """Preprocesado específico de cada arquitectura (equivalente a preprocess_input
    de keras.applications). Entrada: float32 RGB en [0, 255]."""
    import tensorflow as tf
    x = tf.cast(x, tf.float32)
    if arch == "resnet50":                       # modo "caffe": BGR y resta de medias
        x = x[..., ::-1] - tf.constant(_CAFFE_MEAN)
    elif arch == "densenet121":                  # modo "torch"
        x = (x / 255.0 - tf.constant(_TORCH_MEAN)) / tf.constant(_TORCH_STD)
    elif arch == "efficientnetb0":               # el modelo normaliza internamente
        pass
    else:
        raise ValueError(arch)
    return x


def _load_uint8(path):
    import tensorflow as tf
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])   # no-op si ya está a IMG_SIZE
    return tf.cast(tf.round(img), tf.uint8)


def augment_strong(img):
    """Augmentation para clases minoritarias. img float32 [0, 255]."""
    import tensorflow as tf
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, tf.random.uniform([], 0, 4, dtype=tf.int32))
    # zoom aleatorio (recorte 85-100 % y reescalado)
    frac = tf.random.uniform([], 0.85, 1.0)
    side = tf.cast(frac * IMG_SIZE, tf.int32)
    img = tf.image.random_crop(img, [side, side, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    # color (en [0, 1])
    img = img / 255.0
    img = tf.image.random_brightness(img, 0.08)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    img = tf.image.random_saturation(img, 0.9, 1.1)
    img = tf.image.random_hue(img, 0.02)
    return tf.clip_by_value(img, 0.0, 1.0) * 255.0


def augment_light(img):
    """Augmentation ligera (solo geometría sin pérdida) para la clase mayoritaria."""
    import tensorflow as tf
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img


def make_dataset(df: pd.DataFrame, arch: str, training: bool, batch_size: int = 32,
                 seed: int = 42, aug_mode: str = "selective",
                 aug_classes: list | None = None, cache: bool | None = None):
    """Construye un tf.data.Dataset -> (imagen preprocesada, one-hot).

    aug_mode: 'selective' (fuerte en aug_classes, ligera en el resto),
              'all' (fuerte en todas), 'none'.
    En validación/test no hay augmentation ni barajado (el orden = orden del df).
    """
    import tensorflow as tf
    AUTOTUNE = tf.data.AUTOTUNE
    aug_classes = aug_classes or [c for c in CLASSES if c != "nv"]
    aug_idx = tf.constant([CLASS_TO_IDX[c] for c in aug_classes], dtype=tf.int32)
    cache = (not training) if cache is None else cache

    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values.astype("int32")))
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (_load_uint8(p), y), num_parallel_calls=AUTOTUNE)
    if cache:
        ds = ds.cache()

    def _aug(img, y):
        img = tf.cast(img, tf.float32)
        if training and aug_mode == "all":
            img = augment_strong(img)
        elif training and aug_mode == "selective":
            strong = tf.reduce_any(tf.equal(y, aug_idx))
            img = tf.cond(strong, lambda: augment_strong(img), lambda: augment_light(img))
        return img, y

    ds = ds.map(_aug, num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess(x, arch), tf.one_hot(y, NUM_CLASSES)),
                num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def load_image_uint8(path: str) -> np.ndarray:
    """Imagen (IMG_SIZE, IMG_SIZE, 3) uint8 para visualización / Grad-CAM."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    return np.asarray(img)


# =============================================================================
# 4. MODELOS  (Módulo 2)
# =============================================================================
def build_model(arch: str, dropout: float = 0.3, weights: str | None = "imagenet",
                mixed_precision: bool = False):
    """Backbone ImageNet sin top + GAP + Dropout + Dense(logits) + softmax.

    Las capas del backbone quedan "aplanadas" en el modelo (no anidadas), por lo
    que model.get_layer('conv5_block3_out') funciona directamente para Grad-CAM.
    La capa 'head_logits' expone los logits (pre-softmax) para Grad-CAM.
    """
    import tensorflow as tf
    keras = tf.keras
    keras.backend.clear_session()
    keras.mixed_precision.set_global_policy("mixed_float16" if mixed_precision else "float32")
    ctor = {
        "resnet50": keras.applications.ResNet50,
        "densenet121": keras.applications.DenseNet121,
        "efficientnetb0": keras.applications.EfficientNetB0,
    }[arch]
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
    base = ctor(include_top=False, weights=weights, input_tensor=inputs)
    x = keras.layers.GlobalAveragePooling2D(name="head_gap")(base.output)
    x = keras.layers.Dropout(dropout, name="head_dropout")(x)
    logits = keras.layers.Dense(NUM_CLASSES, name="head_logits", dtype="float32")(x)
    probs = keras.layers.Activation("softmax", name="head_softmax", dtype="float32")(logits)
    return keras.Model(inputs, probs, name=f"{arch}_ham10000")


def _base_layers(model):
    import tensorflow as tf
    return [l for l in model.layers
            if not l.name.startswith("head_") and not isinstance(l, tf.keras.layers.InputLayer)]


def set_trainable(model, unfreeze_from: str | None = None, train_bn: bool = False) -> dict:
    """unfreeze_from=None -> backbone congelado (fase 1).
    unfreeze_from='conv5_block1' -> se entrena desde esa capa hasta el final.
    Las BatchNormalization se mantienen congeladas (modo inferencia) salvo train_bn."""
    import tensorflow as tf
    base = _base_layers(model)
    if unfreeze_from is None:
        start = len(base)
    else:
        start = next((i for i, l in enumerate(base) if l.name.startswith(unfreeze_from)), None)
        if start is None:
            raise ValueError(f"No hay capas que empiecen por '{unfreeze_from}'")
    for i, l in enumerate(base):
        t = i >= start
        if isinstance(l, tf.keras.layers.BatchNormalization) and not train_bn:
            t = False
        l.trainable = t
    for l in model.layers:
        if l.name.startswith("head_"):
            l.trainable = True
    n_train = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    n_total = int(sum(np.prod(w.shape) for w in model.weights))
    return {"unfreeze_from": unfreeze_from, "trainable_params": n_train, "total_params": n_total}


def get_last_conv_layer_name(model) -> str:
    """Última capa con salida 4D antes de la cabeza (objetivo de Grad-CAM)."""
    for l in reversed(model.layers):
        if l.name.startswith("head_"):
            continue
        shape = getattr(l, "output", None)
        if shape is not None and len(l.output.shape) == 4:
            return l.name
    raise ValueError("No se encontró capa convolucional")


# =============================================================================
# 5. MÉTRICAS  (Módulo 2)
# =============================================================================
def compute_metrics(y_true, probs):
    """Métricas multiclase (macro) + por clase one-vs-rest.
    Devuelve (dict_plano, DataFrame_por_clase, matriz_confusion)."""
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_pred = probs.argmax(axis=1)
    K = NUM_CLASSES
    cm = confusion_matrix(y_true, y_pred, labels=list(range(K)))
    total = cm.sum()
    rows = []
    for k in range(K):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = total - tp - fn - fp
        prec = tp / (tp + fp) if tp + fp else 0.0
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
        yk = (y_true == k).astype(int)
        auc = roc_auc_score(yk, probs[:, k]) if 0 < yk.sum() < len(yk) else np.nan
        rows.append(dict(clase=CLASSES[k], soporte=int(tp + fn), precision=prec,
                         sensibilidad=sens, especificidad=spec, f1=f1, auc=auc))
    per_class = pd.DataFrame(rows).set_index("clase")
    try:
        auc_macro = roc_auc_score(y_true, probs, multi_class="ovr", average="macro",
                                  labels=list(range(K)))
    except ValueError:
        auc_macro = float(np.nanmean(per_class["auc"]))
    m = {
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(per_class["sensibilidad"].mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(K)), zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=list(range(K)), zero_division=0)),
        "auc_macro_ovr": float(auc_macro),
        "macro_precision": float(per_class["precision"].mean()),
        "macro_specificity": float(per_class["especificidad"].mean()),
        "mel_sensitivity": float(per_class.loc["mel", "sensibilidad"]),
        "mel_specificity": float(per_class.loc["mel", "especificidad"]),
        "mel_precision": float(per_class.loc["mel", "precision"]),
    }
    for c in CLASSES:
        for col in ["precision", "sensibilidad", "especificidad", "f1", "auc"]:
            m[f"{c}_{col}"] = float(per_class.loc[c, col])
    return m, per_class, cm


def selection_score(m: dict, min_mel_sens: float = MIN_MEL_SENSITIVITY):
    """Regla de selección: primero cumplir sensibilidad de melanoma >= umbral,
    después mayor macro-F1. Las tuplas se comparan en ese orden."""
    return (m["mel_sensitivity"] >= min_mel_sens, m["macro_f1"])


# =============================================================================
# 6. UTILIDADES
# =============================================================================
def set_seed(seed: int, deterministic: bool = False):
    import tensorflow as tf
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministic:                       # más lento; usar solo si hace falta
        tf.config.experimental.enable_op_determinism()


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def init_mlflow(experiment: str = MLFLOW_EXPERIMENTS["sandbox"]):
    """Conecta MLflow al servidor de DagsHub. Token en el secreto de Colab
    'DAGSHUB_TOKEN' (panel de la llave a la izquierda).

    `experiment`: uno de MLFLOW_EXPERIMENTS (p. ej. MLFLOW_EXPERIMENTS["m3"]).
    Sin argumento va a "sandbox", que nunca se cita en la memoria."""
    import dagshub
    import mlflow
    try:
        from google.colab import userdata
        token = userdata.get("DAGSHUB_TOKEN")
        if token:
            dagshub.auth.add_app_token(token)
    except Exception:
        pass
    dagshub.init(repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment(experiment)
    return mlflow


def safe(fn, *args, **kwargs):
    """Ejecuta una llamada (p. ej. a MLflow) sin tumbar el entrenamiento si falla."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        print(f"[aviso] {getattr(fn, '__name__', fn)} falló: {e}")
        return None


def plot_confusion_matrix(cm, normalize: bool = True, title: str = ""):
    import matplotlib.pyplot as plt
    import seaborn as sns
    data = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(data, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax, cbar=False)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(title)
    fig.tight_layout()
    return fig
