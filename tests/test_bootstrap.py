"""El bootstrap agrupa por lesión y se repite con la misma semilla."""
import numpy as np
import pandas as pd

from src.bootstrap import bootstrap_prueba, replicas


def _tabla(filas):
    df = pd.DataFrame(filas)
    for c in ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"):
        if f"p_{c}" not in df:
            df[f"p_{c}"] = 0.1
    df["p_mel"] = [0.9 if y == "mel" else 0.1 for y in df.y_true]
    return df


def test_una_lesion_entra_con_todas_sus_imagenes():
    filas = [
        {"image_id": "a", "lesion_id": "L", "y_true": "mel", "y_pred": "mel", "y_pred_umbral": "mel"},
        {"image_id": "b", "lesion_id": "L", "y_true": "mel", "y_pred": "nv", "y_pred_umbral": "mel"},
    ]
    for clase in ("nv", "df", "bkl", "bcc", "akiec", "vasc"):
        filas.append({
            "image_id": clase, "lesion_id": clase, "y_true": clase,
            "y_pred": clase, "y_pred_umbral": clase,
        })
    tabla = _tabla(filas)
    for idx in replicas(tabla, n=30, semilla=1):
        cuenta_a = int(np.sum(tabla.image_id.to_numpy()[idx] == "a"))
        cuenta_b = int(np.sum(tabla.image_id.to_numpy()[idx] == "b"))
        assert cuenta_a == cuenta_b


def test_la_misma_semilla_repite_el_intervalo():
    filas = []
    for i, clase in enumerate(["mel", "nv", "df", "bkl", "bcc", "akiec", "vasc"]):
        filas.append({
            "image_id": f"i{i}", "lesion_id": f"L{i}", "y_true": clase,
            "y_pred": clase, "y_pred_umbral": "mel" if clase == "nv" else clase,
        })
    tabla = _tabla(filas)
    tablas = {"resnet50": tabla, "densenet121": tabla, "efficientnetb0": tabla.copy()}
    tablas["efficientnetb0"]["y_pred"] = "mel"
    a = bootstrap_prueba(tablas, n=40, semilla=7)
    b = bootstrap_prueba(tablas, n=40, semilla=7)
    assert a["intervalos"]["efficientnetb0"]["argmax"]["macro_f1"] == \
        b["intervalos"]["efficientnetb0"]["argmax"]["macro_f1"]
    assert len(a["comparaciones"]) == 4
