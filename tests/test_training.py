"""Pruebas de src/training.py con datos sintéticos de 64×64 y pesos aleatorios (CPU, sin MLflow).

    pytest tests/test_training.py
Lo que NO se puede comprobar aquí (GPU, tiempo real por época, DagsHub) se ve en Colab con
`notebooks/m2_01_entrenamiento.ipynb` en modo sandbox.
"""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

tf = pytest.importorskip("tensorflow")
keras = tf.keras

from src.models import build_model, congelar_base
from src.training import (ControlFase, entrenar_fase, entrenar_run, figura_confusion, figura_curvas,
                          lanzar_run, proyeccion_tiempo)

with open("configs/base.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


def _ds(n, seed, bs=8):
    rng = np.random.default_rng(seed)
    x = rng.random((n, 64, 64, 3)).astype("float32")
    y = rng.integers(0, 7, n)
    return tf.data.Dataset.from_tensor_slices((x, tf.one_hot(y, 7))).batch(bs), y


def _modelo():
    keras.utils.set_random_seed(0)
    return build_model("resnet50", weights=None, img_size=64)


def test_dos_fases_registran_con_paso_acumulado_y_dejan_el_mejor_punto(tmp_path):
    train, _ = _ds(32, 1)
    val, y_val = _ds(16, 2)
    df_val = pd.DataFrame({"label": y_val})
    registro = []
    m = _modelo()
    r = entrenar_run(m, CFG, train, val, df_val, tmp_path, class_weight={i: 1.0 for i in range(7)},
                     epochs=(2, 2), registrar=lambda met, paso: registro.append((paso, sorted(met))))
    # pasos 1..4 (época acumulada) y prefijos f1_ / f2_
    assert [p for p, _ in registro] == [1, 2, 3, 4]
    assert all(k.startswith("f1_") for k in registro[0][1]) and all(k.startswith("f2_") for k in registro[2][1])
    assert {"f1_val_loss", "f1_val_macro_f1", "f1_loss"} <= set(registro[0][1])
    # historial de las dos fases, marcas de completada y pesos del mejor punto de la fase 2
    assert [h["epoca"] for h in r["historial"]] == [1, 2, 3, 4]
    assert (tmp_path / "f1_completa.json").exists() and (tmp_path / "f2_completa.json").exists()
    mejor = build_model("resnet50", weights=None, img_size=64)
    mejor.load_weights(tmp_path / "f2_mejor.weights.h5")
    for a, b in zip(m.get_weights(), mejor.get_weights()):
        assert np.array_equal(a, b)
    assert r["fase2"]["mejor_val_loss"] == min(h["val_loss"] for h in r["historial"] if h["fase"] == "f2")
    assert "macro_f1" in r["metricas"] and r["n_val"] == 16


def test_fase2_entrena_mas_parametros_que_fase1(tmp_path):
    train, _ = _ds(16, 1)
    val, y_val = _ds(8, 2)
    r = entrenar_run(_modelo(), CFG, train, val, pd.DataFrame({"label": y_val}), tmp_path, epochs=(1, 1))
    assert r["info_f2"]["params_entrenables"] > r["info_f1"]["params_entrenables"]


def test_control_fase_parada_anticipada_y_restaura_el_mejor(tmp_path):
    m = keras.Sequential([keras.Input((3,)), keras.layers.Dense(2)])
    ctl = ControlFase(tmp_path, "f1", patience=2)
    ctl.set_model(m)
    for epoca, v in enumerate([1.0, 0.9, 0.95, 0.97]):
        if epoca == 1:
            mejor = [w.copy() for w in m.get_weights()]       # el mejor punto es la época 1
        m.set_weights([w + 1 for w in m.get_weights()])        # los pesos cambian cada época
        ctl.on_epoch_end(epoca, {"val_loss": v})
        if epoca == 1:                                          # se guardó justo en la mejor
            mejor = [w.copy() for w in m.get_weights()]
    assert ctl.estado["mejor_epoca"] == 1 and ctl.estado["espera"] == 2
    assert ctl.estado["parada_anticipada"] and m.stop_training
    ctl.on_train_end()
    for a, b in zip(m.get_weights(), mejor):
        assert np.allclose(a, b)
    # El estado sobrevive a una "desconexión": un ControlFase nuevo lo recupera
    otro = ControlFase(tmp_path, "f1", patience=2)
    assert otro.estado["mejor"] == 0.9 and otro.estado["espera"] == 2


def test_reanudar_tras_desconexion_no_duplica_epocas_ni_repite_fases(tmp_path):
    train, _ = _ds(16, 1)
    val, y_val = _ds(8, 2)
    cw = {i: 1.0 for i in range(7)}
    llamadas = []

    def cae_en_la_epoca_2(met, paso):
        llamadas.append(paso)
        if paso == 2 and not (tmp_path / "ya_cayo").exists():
            (tmp_path / "ya_cayo").write_text("x")
            raise RuntimeError("Colab se desconectó")

    m1 = _modelo()
    congelar_base(m1)
    with pytest.raises(RuntimeError):
        entrenar_fase(m1, train, val, y_val, "f1", 1e-3, 3, 5, cw, 0, tmp_path, cae_en_la_epoca_2)
    assert not (tmp_path / "f1_completa.json").exists()

    # Sesión nueva: modelo nuevo, mismo directorio de estado
    m2 = _modelo()
    congelar_base(m2)
    r = entrenar_fase(m2, train, val, y_val, "f1", 1e-3, 3, 5, cw, 0, tmp_path, cae_en_la_epoca_2)
    hist = json.loads((tmp_path / "f1_historial.json").read_text())
    assert [h["epoca"] for h in hist] == [1, 2, 3]                 # cada época una sola vez
    assert r["epocas_ejecutadas"] == 3 and (tmp_path / "f1_completa.json").exists()

    # Tercera vez: la fase ya está completa, no entrena ni registra nada
    n = len(llamadas)
    m3 = _modelo()
    congelar_base(m3)
    r3 = entrenar_fase(m3, train, val, y_val, "f1", 1e-3, 3, 5, cw, 0, tmp_path, cae_en_la_epoca_2)
    assert len(llamadas) == n and r3["epocas_ejecutadas"] == 3
    for a, b in zip(m3.get_weights(), m2.get_weights()):           # recargó el mejor punto
        assert np.array_equal(a, b)


def test_lanzar_run_real_no_admite_acortar_ni_git_sucio(monkeypatch):
    with pytest.raises(ValueError):
        lanzar_run("resnet50", 42, CFG, "real", "alex", max_lotes=2)
    with pytest.raises(ValueError):
        lanzar_run("resnet50", 42, CFG, "real", "alex", epochs=(1, 1))
    import src.tracking as tr
    monkeypatch.setattr(tr, "git_dirty", lambda: True)
    with pytest.raises(RuntimeError, match="git_dirty"):
        lanzar_run("resnet50", 42, CFG, "real", "alex")


def test_proyeccion_tiempo_y_figuras():
    t = proyeccion_tiempo({"a": {"f1": 0.1, "f2": 0.2}, "b": {"f1": 0.05, "f2": 0.1}})
    # a: 220 * (15*0.1 + 30*0.2) / 60 = 27.5 min; b: 220 * (15*0.05 + 30*0.1) / 60 = 13.75 min
    assert t.loc["a", "min_por_run_peor_caso"] == 27.5 and t.loc["b", "min_por_run_peor_caso"] == 13.8
    assert t.loc["TOTAL por cuenta (3 redes)", "min_por_run_peor_caso"] == 41.3
    h = [{"epoca": 1, "fase": "f1", "loss": 1.0, "val_loss": 1.1, "val_macro_f1": .2},
         {"epoca": 2, "fase": "f2", "loss": .9, "val_loss": 1.0, "val_macro_f1": .3}]
    assert figura_curvas(h, "x") is not None and figura_confusion(np.eye(7), "x") is not None
