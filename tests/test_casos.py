"""Pruebas de src/casos.py (solo pandas/numpy):  pytest tests/test_casos.py"""
import numpy as np
import pandas as pd
import pytest

from src.casos import CLASES, asignar_cuotas, estrato, seleccionar_casos


def _tabla(n_mel_fn=80, n_mel_tp=60, seed=0):
    rng = np.random.default_rng(seed)
    filas = []
    destinos = rng.choice(["nv", "bkl", "bcc", "akiec", "df"], size=n_mel_fn, p=[.5, .3, .1, .05, .05])
    for i, d in enumerate(destinos):
        filas.append((f"FN{i:03d}", f"L{i // 2:03d}", "mel", d))      # dos imágenes por lesión
    for i in range(n_mel_tp):
        filas.append((f"TP{i:03d}", f"T{i:03d}", "mel", "mel"))
    for i in range(50):
        filas.append((f"NV{i:03d}", f"N{i:03d}", "nv", "nv"))
    df = pd.DataFrame(filas, columns=["image_id", "lesion_id", "y_true", "y_pred"])
    p = rng.dirichlet(np.ones(7), size=len(df))
    for j, c in enumerate(CLASES):
        df[f"p_{c}"] = p[:, j]
    return df


def _fn(df):
    f = df[(df.y_true == "mel") & (df.y_pred != "mel")]
    return f[["image_id", "lesion_id", "y_pred", "p_mel"]]


def test_estrato():
    assert [estrato(c) for c in ("nv", "bkl", "bcc", "mel")] == ["nv", "bkl", "otras", "otras"]


def test_cuotas_suman_n_y_respetan_minimo_y_disponibilidad():
    c = asignar_cuotas({"nv": 100, "bkl": 40, "otras": 3}, n=30, minimo=5)
    assert sum(c.values()) == 30 and c["otras"] == 3 and c["bkl"] >= 5 and c["nv"] > c["bkl"]


def test_cuotas_toma_todo_si_hay_menos_de_n():
    assert asignar_cuotas({"nv": 10, "bkl": 4, "otras": 2}) == {"nv": 10, "bkl": 4, "otras": 2}


@pytest.mark.parametrize("arch,n_fn", [("densenet121", 30), ("efficientnetb0", 30)])
def test_muestra_30_por_red_sin_lesiones_repetidas_y_determinista(arch, n_fn):
    df = _tabla()
    a = seleccionar_casos(arch, df, _fn(df), semilla=42)
    b = seleccionar_casos(arch, df, _fn(df), semilla=42)
    assert a.equals(b)
    assert (a.tipo_caso == "FN").sum() == n_fn and (a.tipo_caso == "TP").sum() == 30
    assert not a[a.tipo_caso == "FN"].lesion_id.duplicated().any()
    assert set(a[a.tipo_caso == "FN"].estrato) <= {"nv", "bkl", "otras"}
    assert set(a[a.tipo_caso == "FN"].clase_predicha) != {"mel"}
    assert (a[a.tipo_caso == "TP"].clase_predicha == "mel").all()


def test_resnet50_incluye_todos_los_fn():
    df = _tabla()
    fn = _fn(df)
    r = seleccionar_casos("resnet50", df, fn, semilla=42)
    assert set(r[r.tipo_caso == "FN"].image_id) == set(fn.image_id)
    assert (r.tipo_caso == "TP").sum() == 30


def test_columnas_de_anotacion_vacias_y_p_pred():
    df = _tabla()
    r = seleccionar_casos("densenet121", df, _fn(df), semilla=1)
    assert (r[["zona_atencion", "criterios_abcde", "comentario"]] == "").all().all()
    fila = r.iloc[0]
    assert fila.p_pred == df.set_index("image_id").loc[fila.image_id, f"p_{fila.clase_predicha}"]


def test_falla_si_los_fn_no_coinciden():
    df = _tabla()
    with pytest.raises(RuntimeError):
        seleccionar_casos("resnet50", df, _fn(df).iloc[1:], semilla=1)
