"""Modelos del Módulo 2 (OE2): las tres arquitecturas con la MISMA cabeza y el MISMO protocolo de congelado.

Estructura de cada modelo (idéntica para las tres redes, solo cambia el `backbone`):

    Input(224, 224, 3)
      -> backbone ImageNet sin cabeza (ResNet50 | DenseNet121 | EfficientNetB0)
      -> GlobalAveragePooling2D        (head_gap)
      -> Dropout(dropout)              (head_dropout)
      -> Dense(7, softmax)             (head_softmax)

Decisiones y su razón (el detalle completo está en docs/modulo2_registro_decisiones.md):

  * Capas del backbone "aplanadas" dentro del modelo (se construye con `input_tensor`) y no
    anidadas como un sub-modelo. Así el Módulo 3 puede hacer `model.get_layer("conv5_block3_out")`
    directamente para Grad-CAM, con los nombres de capa de `configs/base.yaml: gradcam`.
  * Todas las capas de la cabeza empiezan por `head_`. Es la marca que usan `congelar_base()` y
    `descongelar_fraccion()` para distinguir "base" de "cabeza" sin depender de la arquitectura.
  * El modelo NO normaliza la entrada: `src.data.make_dataset()` ya aplica el `preprocess_input`
    oficial de cada red. Normalizar dos veces degradaría en silencio el rendimiento.
  * Las capas BatchNormalization de la base se mantienen SIEMPRE congeladas, también en la fase 2.
    Con lotes de 32 y un dominio tan distinto de ImageNet, actualizar sus estadísticas
    (media/varianza móviles) degrada los pesos preentrenados. En Keras 3 una BN con
    `trainable=False` funciona en modo inferencia (usa las estadísticas guardadas), que es
    exactamente lo que pide `docs/modulo2_entrenamiento_evaluacion.md` §3.
  * Keras exige volver a llamar a `model.compile()` después de cambiar `trainable`. Los
    helpers de este módulo NO compilan: lo hace el código de entrenamiento con el LR de cada fase.
"""
from __future__ import annotations

import keras
from keras import layers

ARQUITECTURAS = ("resnet50", "densenet121", "efficientnetb0")
NUM_CLASES = 7
IMG_SIZE = 224

_CONSTRUCTORES = {
    "resnet50": keras.applications.ResNet50,
    "densenet121": keras.applications.DenseNet121,
    "efficientnetb0": keras.applications.EfficientNetB0,
}


def build_model(arch: str, num_classes: int = NUM_CLASES, dropout: float = 0.3,
                weights: str | None = "imagenet", img_size: int = IMG_SIZE) -> keras.Model:
    """Construye el modelo de `arch` con la base ENTRENABLE; llamar luego a `congelar_base()`.

    weights="imagenet" para entrenar; weights=None solo para pruebas rápidas sin descargar pesos.
    `dropout` viene de `configs/base.yaml: training.dropout` (igual para las tres redes).
    """
    assert arch in ARQUITECTURAS, f"arquitectura desconocida: {arch}"
    keras.backend.clear_session()        # evita acumular grafos y nombres de capa con sufijo _1, _2...
    entrada = keras.Input(shape=(img_size, img_size, 3), name="input_image")
    base = _CONSTRUCTORES[arch](include_top=False, weights=weights, input_tensor=entrada)
    x = layers.GlobalAveragePooling2D(name="head_gap")(base.output)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    # dtype float32 en la salida: si algún día se activa precisión mixta, la softmax no pierde precisión.
    salida = layers.Dense(num_classes, activation="softmax", dtype="float32", name="head_softmax")(x)
    return keras.Model(entrada, salida, name=f"{arch}_ham10000")


def capas_base(model: keras.Model) -> list[keras.layers.Layer]:
    """Capas del backbone en orden de entrada -> salida (sin la capa Input ni la cabeza)."""
    return [l for l in model.layers
            if not l.name.startswith("head_") and not isinstance(l, layers.InputLayer)]


def _capas_cabeza(model: keras.Model) -> list[keras.layers.Layer]:
    return [l for l in model.layers if l.name.startswith("head_")]


def congelar_base(model: keras.Model) -> dict:
    """FASE 1: base congelada, solo se entrena la cabeza (head_*).

    Razón: la cabeza se inicializa al azar; si se entrenara todo a la vez, los gradientes
    grandes de una cabeza sin entrenar destruirían los pesos de ImageNet de la base.
    """
    for l in capas_base(model):
        l.trainable = False
    for l in _capas_cabeza(model):
        l.trainable = True
    return resumen_entrenable(model)


def descongelar_fraccion(model: keras.Model, fraccion: float = 0.3,
                         congelar_bn: bool = True) -> dict:
    """FASE 2: descongela la fracción SUPERIOR de la base (las capas más cercanas a la salida).

    - `fraccion` = `training.phase2.unfreeze_fraction` (0.3 = el 30 % final de las capas de la base).
    - Se cuenta por número de capas y se aplica la misma regla a las tres redes. Como cada red
      tiene un número distinto de capas, el 30 % cae a distinta profundidad en cada una; es lo
      que implica "mismos hiperparámetros base" y queda registrado en el run (n_capas_base,
      n_capas_descongeladas, params entrenables).
    - Se descongelan las capas finales y no las iniciales porque las primeras detectan bordes
      y texturas genéricas (valen para dermatoscopia), mientras que las últimas son las más
      específicas de ImageNet y las que más conviene adaptar.
    - `congelar_bn=True` deja las BatchNormalization congeladas aunque estén en la zona descongelada.
    """
    assert 0.0 < fraccion <= 1.0, "fraccion debe estar en (0, 1]"
    base = capas_base(model)
    n_desc = max(1, round(fraccion * len(base)))
    inicio = len(base) - n_desc
    for i, l in enumerate(base):
        entrenable = i >= inicio
        if congelar_bn and isinstance(l, layers.BatchNormalization):
            entrenable = False
        l.trainable = entrenable
    for l in _capas_cabeza(model):
        l.trainable = True
    r = resumen_entrenable(model)
    r.update(n_capas_base=len(base), n_capas_descongeladas=n_desc, primera_capa_descongelada=base[inicio].name)
    return r


def resumen_entrenable(model: keras.Model) -> dict:
    """Conteos para registrar en MLflow y para comprobar que el congelado hizo lo esperado."""
    def n(ws):
        return int(sum(int(keras.ops.prod(w.shape)) for w in ws))
    return {
        "params_entrenables": n(model.trainable_weights),
        "params_no_entrenables": n(model.non_trainable_weights),
        "params_total": n(model.weights),
    }


def bn_entrenables(model: keras.Model) -> int:
    """Número de BatchNormalization de la base con trainable=True (debe ser 0 en las dos fases)."""
    return sum(1 for l in capas_base(model)
               if isinstance(l, layers.BatchNormalization) and l.trainable)
