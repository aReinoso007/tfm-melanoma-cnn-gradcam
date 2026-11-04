"""Pruebas de src/metrics.py. No necesitan TensorFlow ni GPU: `pytest tests/test_metrics.py`."""
import numpy as np
import pandas as pd
import pytest

from src.metrics import (CLASES, calcular_metricas, informe_clasificacion, macro_f1_rapido,
                         matriz_confusion, prefijar, tabla_predicciones)


def _probs_perfectas(y, ruido=0.0, seed=0):
    p = np.full((len(y), 7), 0.01) + ruido * np.random.default_rng(seed).random((len(y), 7))
    p[np.arange(len(y)), y] = 1.0
    return p / p.sum(1, keepdims=True)


def test_prediccion_perfecta():
    y = np.array([0, 1, 2, 3, 4, 5, 6] * 5)
    m = calcular_metricas(y, _probs_perfectas(y))
    for k in ("macro_f1", "balanced_accuracy", "accuracy", "auc_macro_ovr",
              "mel_sensitivity", "mel_precision", "mel_f1", "mel_ap", "min_class_sensitivity"):
        assert m[k] == pytest.approx(1.0), k


def test_nombres_de_la_convencion():
    """docs/03_mlflow_convenciones.md §5: todas estas claves deben existir."""
    y = np.arange(70) % 7
    m = calcular_metricas(y, _probs_perfectas(y, ruido=0.5))
    esperadas = {"macro_f1", "balanced_accuracy", "auc_macro_ovr", "mel_sensitivity", "mel_precision",
                 "mel_f1", "mel_ap", "min_class_sensitivity"}
    for c in CLASES:
        esperadas |= {f"{p}_{c}" for p in ("sensitivity", "precision", "f1", "specificity", "auc", "ap")}
    assert esperadas <= set(m)
    assert all(k.startswith("val_") for k in prefijar(m, "val"))


def test_sensibilidad_melanoma_y_minima_por_clase():
    # 10 melanomas: 7 bien, 3 predichos como nv. El resto de clases, perfectas.
    y = np.array([4] * 10 + [5] * 10 + [0] * 10)
    p = _probs_perfectas(y)
    p[7:10] = _probs_perfectas(np.array([5, 5, 5]))
    m = calcular_metricas(y, p)
    assert m["mel_sensitivity"] == pytest.approx(0.7)
    assert m["sensitivity_nv"] == pytest.approx(1.0)
    assert m["precision_mel"] == pytest.approx(1.0)
    assert m["precision_nv"] == pytest.approx(10 / 13)
    # clases ausentes en y_true valen 0 en sensibilidad: el mínimo es 0 (con 7 clases y solo 3
    # presentes), lo correcto para la regla "cada clase >= 0.5" si esa clase existiera sin aciertos.
    assert m["min_class_sensitivity"] == pytest.approx(0.0)


def test_balanced_accuracy_coincide_con_sklearn():
    from sklearn.metrics import balanced_accuracy_score
    rng = np.random.default_rng(1)
    y = rng.integers(0, 7, 500)
    p = rng.dirichlet(np.ones(7), 500)
    m = calcular_metricas(y, p)
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, p.argmax(1)))
    assert m["macro_f1"] == pytest.approx(macro_f1_rapido(y, p))


def test_clase_ausente_da_nan_en_auc_no_cero():
    y = np.array([0, 1, 2, 3, 4, 5] * 4)          # falta 'vasc'
    m = calcular_metricas(y, _probs_perfectas(y))
    assert np.isnan(m["auc_vasc"])


def test_matriz_confusion_normalizada_suma_1_por_fila():
    y = np.arange(70) % 7
    cm = matriz_confusion(y, _probs_perfectas(y, ruido=2.0).argmax(1))
    assert cm.shape == (7, 7)
    assert cm.sum(1) == pytest.approx(np.ones(7))


def test_tabla_predicciones_formato_contrato_m3():
    df = pd.DataFrame({"image_id": ["ISIC_1", "ISIC_2", "ISIC_3"],
                       "lesion_id": ["HAM_1", "HAM_2", "HAM_3"],
                       "dx": ["mel", "nv", "mel"]})
    p = _probs_perfectas(np.array([4, 5, 5]))
    t = tabla_predicciones(df, p)
    assert list(t.columns) == ["image_id", "lesion_id", "y_true", "y_pred"] + [f"p_{c}" for c in CLASES]
    assert t.y_pred.tolist() == ["mel", "nv", "nv"]          # el 3.º es un falso negativo de melanoma
    assert t.y_true.tolist() == ["mel", "nv", "mel"]


def test_tabla_predicciones_rechaza_orden_o_longitud_incorrectos():
    df = pd.DataFrame({"image_id": ["ISIC_2", "ISIC_1"], "lesion_id": ["a", "b"], "dx": ["nv", "nv"]})
    with pytest.raises(AssertionError):
        tabla_predicciones(df, np.full((2, 7), 1 / 7))
    with pytest.raises(AssertionError):
        tabla_predicciones(df.sort_values("image_id"), np.full((3, 7), 1 / 7))


def test_informe_clasificacion_tiene_las_7_clases():
    y = np.arange(70) % 7
    r = informe_clasificacion(y, _probs_perfectas(y))
    assert set(CLASES) <= set(r)
