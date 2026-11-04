"""Métricas del Módulo 2 (OE3): todo se calcula a partir de `y_true` y `y_prob`.

Por qué un módulo aparte y no dentro del notebook:
  - Las mismas funciones se usan en validación (9 runs) y en prueba (3 runs). Si el
    cálculo viviera en dos notebooks, un cambio en uno de ellos haría que los números de
    validación y de prueba dejaran de ser comparables.
  - Los nombres de las métricas son los de `docs/03_mlflow_convenciones.md` §5, así lo que
    devuelve `calcular_metricas()` se puede pasar tal cual a `mlflow.log_metrics()`.
  - No importa TensorFlow: se puede probar y reutilizar (p. ej. en el Módulo 3) sin GPU.

Convenciones:
  - `y_true`: vector de enteros (0–6) en el orden de `CLASES` (alfabético, el mismo que
    `src/augmentation.py` y `configs/base.yaml: data.classes`).
  - `y_prob`: matriz (N, 7) de probabilidades (salida softmax). `y_pred = argmax`.
  - Las métricas por clase son *uno contra el resto* (OvR); los promedios son *macro*,
    es decir, cada clase pesa lo mismo aunque tenga pocas imágenes (df, vasc).

Uso:
    from src.metrics import calcular_metricas, matriz_confusion, tabla_predicciones
    m = calcular_metricas(y_true, y_prob)                    # sin prefijo
    mlflow.log_metrics(prefijar(m, "val"))                   # -> val_macro_f1, val_mel_sensitivity, ...
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             roc_auc_score)

# Mismo orden que src/augmentation.py::CLASES y configs/base.yaml (data.classes).
# El índice de cada clase es su etiqueta numérica; no se debe reordenar nunca.
CLASES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
IDX_MEL = CLASES.index("mel")


def _validar(y_true, y_prob):
    """Convierte a arrays y comprueba formas. Falla pronto si algo no cuadra."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    assert y_prob.ndim == 2 and y_prob.shape[1] == len(CLASES), (
        f"y_prob debe tener forma (N, {len(CLASES)}), no {y_prob.shape}")
    assert len(y_true) == len(y_prob), "y_true e y_prob tienen distinto número de filas"
    assert y_true.min() >= 0 and y_true.max() < len(CLASES), "y_true fuera de 0..6"
    # Renormalizar por si la salida llega en float16/float32 y la suma no es exactamente 1.
    # Una fila de ceros (no debería ocurrir) se deja como está para no dividir entre 0.
    suma = y_prob.sum(axis=1, keepdims=True)
    y_prob = np.divide(y_prob, suma, out=y_prob.copy(), where=suma > 0)
    return y_true, y_prob


def matriz_confusion(y_true, y_pred, normalizar: bool = True) -> np.ndarray:
    """Matriz (7, 7): filas = clase real, columnas = clase predicha.

    `normalizar=True` divide cada fila entre su total, así la diagonal es la sensibilidad de
    cada clase y se pueden comparar clases con tamaños muy distintos (docs/01_objetivos.md).
    Una clase sin muestras produce una fila de ceros, no NaN.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASES))).astype(float)
    if normalizar:
        filas = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, filas, out=np.zeros_like(cm), where=filas > 0)
    return cm


def calcular_metricas(y_true, y_prob) -> dict[str, float]:
    """Todas las métricas de `docs/03_mlflow_convenciones.md` §5, sin prefijo.

    Devuelve un diccionario plano {nombre: float}. Con `prefijar(m, "val")` o
    `prefijar(m, "test")` se obtienen los nombres exactos que se registran en MLflow.

    Claves:
      Resumen multiclase:
        macro_f1, balanced_accuracy, auc_macro_ovr, accuracy (complementaria, no se usa para elegir)
      Selección (regla de `configs/base.yaml: selection`):
        mel_sensitivity, min_class_sensitivity
      Melanoma:
        mel_precision, mel_f1, mel_ap (+ mel_sensitivity)
      Por clase c en CLASES:
        sensitivity_<c>, precision_<c>, f1_<c>, specificity_<c>, auc_<c>, ap_<c>
    """
    y_true, y_prob = _validar(y_true, y_prob)
    y_pred = y_prob.argmax(axis=1)
    k = len(CLASES)

    cm = confusion_matrix(y_true, y_pred, labels=range(k))
    total = cm.sum()
    m: dict[str, float] = {}
    sens_por_clase = []

    for i, c in enumerate(CLASES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        # Si el denominador es 0 se devuelve 0.0 (p. ej. precisión de una clase que el modelo
        # nunca predice). Es lo que hace sklearn con zero_division=0 y evita NaN en MLflow.
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0

        # AUC y AP de la clase i contra el resto. Necesitan positivos y negativos en y_true;
        # si falta alguno (submuestra pequeña en el sandbox) se devuelve NaN a propósito:
        # un 0 sería un valor inventado.
        es_i = (y_true == i).astype(int)
        hay_ambos = 0 < es_i.sum() < len(es_i)
        auc = roc_auc_score(es_i, y_prob[:, i]) if hay_ambos else float("nan")
        ap = average_precision_score(es_i, y_prob[:, i]) if es_i.sum() > 0 else float("nan")

        sens_por_clase.append(sens)
        m[f"sensitivity_{c}"] = float(sens)
        m[f"precision_{c}"] = float(prec)
        m[f"f1_{c}"] = float(f1)
        m[f"specificity_{c}"] = float(spec)
        m[f"auc_{c}"] = float(auc)
        m[f"ap_{c}"] = float(ap)

    # Resumen multiclase.
    # balanced_accuracy = media de las sensibilidades por clase (definición de sklearn);
    # se calcula aquí para que coincida exactamente con las sensibilidades registradas.
    m["balanced_accuracy"] = float(np.mean(sens_por_clase))
    m["macro_f1"] = float(f1_score(y_true, y_pred, average="macro",
                                   labels=list(range(k)), zero_division=0))
    m["accuracy"] = float((y_pred == y_true).mean())
    try:
        m["auc_macro_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr",
                                                 average="macro", labels=list(range(k))))
    except ValueError:      # falta alguna clase en y_true (solo ocurre con submuestras)
        m["auc_macro_ovr"] = float("nan")

    # Métricas que usa la regla de selección y las del melanoma (clase positiva del TFM).
    m["mel_sensitivity"] = m["sensitivity_mel"]
    m["mel_precision"] = m["precision_mel"]
    m["mel_f1"] = m["f1_mel"]
    m["mel_ap"] = m["ap_mel"]
    # Requisito 2 de la regla: ninguna clase por debajo del 50 %. Se resume en un solo número.
    m["min_class_sensitivity"] = float(np.min(sens_por_clase))
    return m


def prefijar(m: dict[str, float], prefijo: str) -> dict[str, float]:
    """{'macro_f1': 0.7} -> {'val_macro_f1': 0.7}. Prefijos usados: 'val' y 'test'."""
    return {f"{prefijo}_{k}": v for k, v in m.items()}


def informe_clasificacion(y_true, y_prob) -> dict:
    """Equivalente a `classification_report(output_dict=True)` con nombres de clase.

    Se guarda como `classification_report_<val|test>.json` (docs/03_mlflow_convenciones.md §6).
    """
    from sklearn.metrics import classification_report
    y_true, y_prob = _validar(y_true, y_prob)
    return classification_report(y_true, y_prob.argmax(axis=1), labels=list(range(len(CLASES))),
                                 target_names=CLASES, output_dict=True, zero_division=0)


def tabla_predicciones(df_split: pd.DataFrame, y_prob) -> pd.DataFrame:
    """Construye `predicciones_<val|test>.csv`.

    `df_split` es `split_df("val")` o `split_df("test")` (src/data.py). Las probabilidades
    deben estar en ESE orden: `make_dataset(..., training=False)` no baraja, así que
    `model.predict(ds)` ya sale ordenado así. Si el orden no coincide, las filas quedarían
    mal emparejadas sin ningún error visible; por eso se comprueba la longitud y que el
    split esté ordenado por image_id.

    Columnas (contrato M2 -> M3, docs/02_flujo_de_trabajo.md §9):
        image_id, lesion_id, y_true, y_pred, p_akiec ... p_vasc
    `y_true` e `y_pred` van como NOMBRE de clase ('mel', 'nv', ...) y no como índice, para que
    el CSV se entienda solo y no dependa de recordar el orden de CLASES.
    """
    y_prob = np.asarray(y_prob, dtype=np.float64)
    assert len(df_split) == len(y_prob), (
        f"{len(df_split)} filas en el split y {len(y_prob)} predicciones")
    assert df_split.image_id.is_monotonic_increasing, (
        "df_split debe venir de split_df(): ordenado por image_id")
    pred = y_prob.argmax(axis=1)
    out = pd.DataFrame({
        "image_id": df_split.image_id.values,
        "lesion_id": df_split.lesion_id.values,
        "y_true": df_split.dx.values,
        "y_pred": [CLASES[i] for i in pred],
    })
    for i, c in enumerate(CLASES):
        out[f"p_{c}"] = y_prob[:, i]
    return out


def macro_f1_rapido(y_true, y_prob) -> float:
    """Solo F1 macro (sin AUC ni AP). Para el callback por época, donde se llama cientos de veces."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_prob).argmax(axis=1)
    return float(f1_score(y_true, y_pred, average="macro",
                          labels=list(range(len(CLASES))), zero_division=0))
