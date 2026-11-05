"""Entrenamiento en dos fases del Módulo 2 (OE2): un run = una arquitectura × una semilla.

Flujo de `lanzar_run()` (lo que ejecuta el notebook `m2_01_entrenamiento.ipynb`):

    set_seeds -> make_dataset (M1) -> build_model -> FASE 1 (base congelada) -> FASE 2 (30 % final)
              -> evaluación en validación -> artefactos y modelo en MLflow

Decisiones y su razón (detalle en docs/modulo2_registro_decisiones.md, D12-D14):

  * Dos fases en el MISMO run. Las métricas llevan prefijo `f1_` / `f2_` y el paso es la época
    acumulada (docs/03_mlflow_convenciones.md §2): así las dos fases se ven como una sola curva.
  * `val_loss` y `val_macro_f1` se calculan en UNA sola pasada por validación al final de cada
    época (`EvaluarValidacion`). No se usa `validation_data=` de Keras: haría una segunda pasada
    solo para la pérdida. La pérdida de validación es la entropía cruzada SIN pesos de clase
    (los pesos solo se aplican al entrenar), igual que la que calcula Keras.
  * Parada anticipada y "mejor época" PROPIAS y persistentes (`ControlFase`), no `EarlyStopping`:
    el de Keras pierde su contador y sus mejores pesos si Colab se desconecta a mitad de una fase,
    y entonces `restore_best_weights` restauraría la mejor época *posterior al reinicio*, no la mejor
    de verdad. `ControlFase` guarda en Drive el mejor valor, la mejor época, el contador de espera
    y los pesos de la mejor época; al reanudar continúa donde estaba.
  * `BackupAndRestore` solo se usa para lo que sí hace bien: recuperar el número de época y el
    estado del optimizador tras una desconexión.
  * Cada fase deja una marca `<fase>_completa.json` en Drive. Si se vuelve a lanzar el run, una fase
    con marca no se entrena otra vez: solo se recargan sus mejores pesos.
  * Modo "real" frente a "sandbox": un único interruptor elige el experimento de MLflow y, en real,
    se niega a arrancar con cambios sin commit (`git_dirty`), porque ese run no valdría como resultado.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from src.metrics import (CLASES, calcular_metricas, informe_clasificacion, macro_f1_rapido,
                         matriz_confusion, prefijar, tabla_predicciones)
from src.models import bn_entrenables, build_model, congelar_base, descongelar_fraccion


# ---------------------------------------------------------------------------
# Utilidades de estado (Drive)
# ---------------------------------------------------------------------------

def _leer_json(path: Path, defecto):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else defecto


def _escribir_json(path: Path, obj) -> None:
    """Escribe en un archivo temporal y lo renombra: si Colab corta a mitad, no queda un JSON a medias."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class EvaluarValidacion(keras.callbacks.Callback):
    """Al final de cada época: una pasada por validación -> val_loss y val_macro_f1.

    Los pone en `logs` (los lee `ControlFase`, que debe ir DESPUÉS en la lista de callbacks), los
    registra en MLflow con nombre `<fase>_...` y paso = época acumulada, y los guarda en
    `<fase>_historial.json` para poder dibujar las curvas aunque la sesión se reinicie.
    """

    def __init__(self, val_ds, y_val, fase: str, offset: int, state_dir, registrar=None):
        super().__init__()
        self.val_ds, self.y_val = val_ds, np.asarray(y_val).astype(int)
        self.fase, self.offset, self.registrar = fase, offset, registrar
        self.f_hist = Path(state_dir) / f"{fase}_historial.json"
        self.historial = _leer_json(self.f_hist, [])
        self._lotes = 0
        self._t0 = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._lotes, self._t0 = 0, time.perf_counter()

    def on_train_batch_end(self, batch, logs=None):
        self._lotes += 1

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        t_train = time.perf_counter() - self._t0          # solo entrenamiento (incluye esperar a los datos)
        probs = self.model.predict(self.val_ds, verbose=0)
        n = len(probs)
        y = self.y_val[:n]                                  # por si val_ds está recortado (sandbox)
        p_ok = np.clip(probs[np.arange(n), y], 1e-7, 1.0)
        logs["val_loss"] = float(-np.log(p_ok).mean())
        logs["val_macro_f1"] = macro_f1_rapido(y, probs)

        ep = self.offset + epoch + 1
        f = self.fase
        fila = {"epoca": ep, "fase": f, "loss": float(logs["loss"]), "val_loss": logs["val_loss"],
                "val_macro_f1": logs["val_macro_f1"], "s_epoca": round(t_train, 2),
                "s_por_paso": round(t_train / max(self._lotes, 1), 4), "pasos": self._lotes}
        self.historial = [h for h in self.historial if h["epoca"] != ep] + [fila]
        _escribir_json(self.f_hist, self.historial)
        if self.registrar:
            self.registrar({f"{f}_loss": fila["loss"], f"{f}_val_loss": fila["val_loss"],
                            f"{f}_val_macro_f1": fila["val_macro_f1"],
                            f"{f}_s_por_paso": fila["s_por_paso"]}, ep)
        print(f"[{f}] época {ep:02d}  loss={fila['loss']:.4f}  val_loss={fila['val_loss']:.4f}  "
              f"val_macro_f1={fila['val_macro_f1']:.4f}  ({fila['s_por_paso']*1000:.0f} ms/paso)")


class ControlFase(keras.callbacks.Callback):
    """Parada anticipada sobre `val_loss` + mejores pesos de la fase, persistentes en Drive."""

    def __init__(self, state_dir, fase: str, patience: int, monitor: str = "val_loss"):
        super().__init__()
        d = Path(state_dir)
        self.patience, self.monitor = patience, monitor
        self.f_json, self.f_pesos = d / f"{fase}_control.json", d / f"{fase}_mejor.weights.h5"
        self.estado = _leer_json(self.f_json, {"mejor": None, "mejor_epoca": None, "espera": 0,
                                               "parada_anticipada": False, "ultima_epoca": -1})

    def on_epoch_end(self, epoch, logs=None):
        v, e = float(logs[self.monitor]), self.estado
        if e["mejor"] is None or v < e["mejor"]:
            e.update(mejor=v, mejor_epoca=epoch, espera=0)
            self.model.save_weights(self.f_pesos)
        else:
            e["espera"] += 1
            if e["espera"] >= self.patience:
                e["parada_anticipada"] = True
                self.model.stop_training = True
        e["ultima_epoca"] = epoch
        _escribir_json(self.f_json, e)

    def on_train_end(self, logs=None):
        # Se deja el modelo en su mejor punto, haya parado antes o haya agotado las épocas.
        if self.f_pesos.exists():
            self.model.load_weights(self.f_pesos)


# ---------------------------------------------------------------------------
# Una fase y un run completo
# ---------------------------------------------------------------------------

def entrenar_fase(model, train_ds, val_ds, y_val, fase: str, lr: float, epochs: int, patience: int,
                  class_weight, offset: int, state_dir, registrar=None) -> dict:
    """Entrena UNA fase. El modelo debe llegar ya congelado/descongelado según la fase.

    Deja el modelo con los pesos de la mejor época (menor `val_loss`) y devuelve un resumen:
    epocas_ejecutadas, mejor_epoca (1 = primera de la fase), mejor_val_loss, parada_anticipada.
    Si la fase ya tiene marca de completada (run reanudado), no entrena: solo recarga esos pesos.
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    marca, pesos = state_dir / f"{fase}_completa.json", state_dir / f"{fase}_mejor.weights.h5"
    if marca.exists():
        print(f"[{fase}] ya completada en una sesión anterior: se recargan sus mejores pesos")
        model.load_weights(pesos)
        return _leer_json(marca, {})

    # Keras exige compilar tras cambiar `trainable`; el LR y el optimizador son los de la fase.
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="categorical_crossentropy")
    ev = EvaluarValidacion(val_ds, y_val, fase, offset, state_dir, registrar)
    ctl = ControlFase(state_dir, fase, patience)           # va después de `ev`: lee su val_loss
    backup = keras.callbacks.BackupAndRestore(str(state_dir / f"backup_{fase}"))
    model.fit(train_ds, epochs=epochs, class_weight=class_weight, callbacks=[ev, ctl, backup], verbose=0)

    e = ctl.estado
    resumen = {"epocas_ejecutadas": e["ultima_epoca"] + 1, "mejor_epoca": e["mejor_epoca"] + 1,
               "mejor_val_loss": e["mejor"], "parada_anticipada": e["parada_anticipada"],
               "s_por_paso_medio": round(float(np.mean([h["s_por_paso"] for h in ev.historial])), 4)}
    _escribir_json(marca, resumen)
    return resumen


def entrenar_run(model, cfg: dict, train_ds, val_ds, df_val: pd.DataFrame, state_dir,
                 class_weight=None, epochs: tuple[int, int] | None = None, registrar=None) -> dict:
    """Fase 1 -> fase 2 -> evaluación en validación. No toca MLflow salvo por `registrar`.

    `epochs=(e1, e2)` solo se usa en sandbox para acortar; en real se toma de `base.yaml`.
    `df_val` es `split_df("val")` (mismo orden que `val_ds`).
    """
    t = cfg["training"]
    e1, e2 = epochs or (t["phase1"]["epochs"], t["phase2"]["epochs"])
    pat = t["early_stopping"]["patience"]
    y_val = df_val.label.values

    info1 = congelar_base(model)
    assert bn_entrenables(model) == 0
    r1 = entrenar_fase(model, train_ds, val_ds, y_val, "f1", t["phase1"]["lr"], e1, pat,
                       class_weight, 0, state_dir, registrar)

    info2 = descongelar_fraccion(model, t["phase2"]["unfreeze_fraction"], t["phase2"]["freeze_batchnorm"])
    assert bn_entrenables(model) == 0 or not t["phase2"]["freeze_batchnorm"]
    r2 = entrenar_fase(model, train_ds, val_ds, y_val, "f2", t["phase2"]["lr"], e2, pat,
                       class_weight, r1["epocas_ejecutadas"], state_dir, registrar)

    probs = model.predict(val_ds, verbose=0)
    n = len(probs)
    metricas = calcular_metricas(y_val[:n], probs)
    historial = (_leer_json(Path(state_dir) / "f1_historial.json", []) +
                 _leer_json(Path(state_dir) / "f2_historial.json", []))
    return {"metricas": metricas, "probs": probs, "n_val": n, "historial": historial,
            "fase1": r1, "fase2": r2, "info_f1": info1, "info_f2": info2}


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def figura_curvas(historial: list[dict], titulo: str = ""):
    """Pérdida y F1 macro de validación frente a la época acumulada, con el cambio de fase marcado."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = pd.DataFrame(historial).sort_values("epoca")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h.epoca, h.loss, label="entrenamiento (con pesos de clase)")
    ax[0].plot(h.epoca, h.val_loss, label="validación")
    ax[0].set_ylabel("pérdida")
    ax[1].plot(h.epoca, h.val_macro_f1, color="tab:green")
    ax[1].set_ylabel("F1 macro (validación)")
    cambio = h[h.fase == "f2"].epoca.min() if (h.fase == "f2").any() else None
    for a in ax:
        a.set_xlabel("época acumulada")
        if cambio is not None and not math.isnan(cambio):
            a.axvline(cambio - 0.5, color="gray", ls="--", lw=1)
        a.grid(alpha=.3)
    ax[0].legend(fontsize=8)
    fig.suptitle(titulo + "  (línea discontinua: inicio de la fase 2)" if titulo else "")
    fig.tight_layout()
    return fig


def figura_confusion(cm_norm: np.ndarray, titulo: str = ""):
    """Matriz de confusión normalizada por filas (la diagonal es la sensibilidad de cada clase)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    k = len(CLASES)
    ax.set_xticks(range(k), CLASES)
    ax.set_yticks(range(k), CLASES)
    ax.set_xlabel("predicción")
    ax.set_ylabel("real")
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > .5 else "black", fontsize=8)
    ax.set_title(titulo)
    fig.tight_layout()
    return fig


def proyeccion_tiempo(s_por_paso: dict[str, dict[str, float]], pasos_por_epoca: int = 220,
                      epocas: tuple[int, int] = (15, 30)) -> pd.DataFrame:
    """Peor caso (sin parada anticipada) en minutos por run y por cuenta, a partir de s/paso medidos.

    `s_por_paso = {arch: {"f1": s, "f2": s}}`. `pasos_por_epoca` = ceil(7009 / 32) = 220 con el split v1.
    Se usa para comprobar la condición de D10: si la suma por cuenta (las 3 redes, 1 semilla) supera
    ≈ 2 h, se vuelve a plantear el determinismo al grupo. No incluye la pasada por validación de cada época.
    """
    filas = []
    for arch, d in s_por_paso.items():
        minutos = (pasos_por_epoca * (epocas[0] * d["f1"] + epocas[1] * d["f2"])) / 60
        filas.append({"arch": arch, "s_por_paso_f1": d["f1"], "s_por_paso_f2": d["f2"],
                      "min_por_run_peor_caso": round(minutos, 1)})
    t = pd.DataFrame(filas).set_index("arch")
    t.loc["TOTAL por cuenta (3 redes)"] = [np.nan, np.nan, round(t.min_por_run_peor_caso.sum(), 1)]
    return t


# ---------------------------------------------------------------------------
# Punto de entrada: un run completo con MLflow
# ---------------------------------------------------------------------------

def _gpu() -> str:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return "sin GPU"
    try:
        return tf.config.experimental.get_device_details(gpus[0]).get("device_name", "GPU")
    except Exception:  # noqa: BLE001
        return "GPU"


def lanzar_run(arch: str, seed: int, cfg: dict, modo: str, autor: str, max_lotes: int | None = None,
               epochs: tuple[int, int] | None = None, registrar_modelo: bool = True,
               precalentar: bool | None = None, data_kwargs: dict | None = None,
               model_kwargs: dict | None = None) -> dict:
    """Un run completo (arquitectura × semilla) registrado en MLflow.

    modo="real"    -> experimento `m2-entrenamiento`, run `m2-<arch>-s<seed>`. Exige git limpio.
    modo="sandbox" -> experimento `sandbox`, run `m2-<arch>-s<seed>-sandbox`; permite acortar con
                      `epochs=(e1, e2)` y `max_lotes` (nº de lotes de train y de validación).
    En real `epochs` y `max_lotes` no se pueden usar: los 9 runs deben ser idénticos.
    `precalentar`: recorre una vez el train completo para llenar la caché antes de recortarlo con
    `max_lotes` (si no, cada época volvería a decodificar los 7 009 JPEG); por defecto, solo con `max_lotes`.
    """
    import mlflow
    import tensorflow as tf
    from src.data import load_class_weights, make_dataset, split_df
    from src.tracking import (git_dirty, log_config, set_seeds, standard_tags, start_or_resume_run)

    assert modo in ("real", "sandbox"), modo
    t = cfg["training"]
    if modo == "real":
        if epochs is not None or max_lotes is not None:
            raise ValueError("En modo real no se puede acortar el entrenamiento (epochs / max_lotes).")
        if git_dirty():
            raise RuntimeError("Hay cambios sin commit: un run real con git_dirty=true no vale como "
                               "resultado final. Haz commit y push antes de lanzarlo.")
        experimento = cfg["mlflow"]["experiments"]["m2_train"]
        run_name = f"m2-{arch}-s{seed}"
        state_dir = Path(cfg["paths"]["checkpoints_dir"]) / run_name
    else:
        experimento = cfg["mlflow"]["experiments"]["sandbox"]
        run_name = f"m2-{arch}-s{seed}-sandbox"
        state_dir = Path(cfg["paths"]["checkpoints_dir"]) / "sandbox" / run_name

    set_seeds(seed, deterministic=bool(t["deterministic"]))
    kw = data_kwargs or {}
    train_ds = make_dataset("train", arch, True, batch_size=t["batch_size"], seed=seed, **kw)
    val_ds = make_dataset("val", arch, False, batch_size=t["batch_size"], **kw)
    df_val = split_df("val")
    if max_lotes is not None:
        if precalentar is None or precalentar:
            for _ in train_ds:      # una pasada completa: llena la caché (si no, cada época re-decodifica todo)
                pass
        train_ds, val_ds = train_ds.take(max_lotes), val_ds.take(max_lotes)

    model = build_model(arch, dropout=t["dropout"], **(model_kwargs or {}))
    mlflow.set_experiment(experimento)
    tags = standard_tags(cfg, modulo="m2", fase="entrenamiento", autor=autor, arch=arch, seed=seed, modo=modo)
    t0 = time.time()

    with start_or_resume_run(run_name, state_dir, tags=tags):
        log_config(cfg)
        mlflow.log_params({"deterministic": bool(t["deterministic"]), "gpu": _gpu(),
                           "tf_version": tf.__version__, "keras_version": keras.__version__,
                           "epochs_f1_usadas_max": (epochs or (t["phase1"]["epochs"], t["phase2"]["epochs"]))[0],
                           "epochs_f2_usadas_max": (epochs or (t["phase1"]["epochs"], t["phase2"]["epochs"]))[1]})

        def registrar(metricas, paso):
            mlflow.log_metrics(metricas, step=paso)

        r = entrenar_run(model, cfg, train_ds, val_ds, df_val, state_dir,
                         class_weight=load_class_weights(), epochs=epochs, registrar=registrar)

        n = r["n_val"]
        mlflow.log_params({
            "n_capas_base": r["info_f2"]["n_capas_base"],
            "n_capas_descongeladas": r["info_f2"]["n_capas_descongeladas"],
            "params_entrenables_f1": r["info_f1"]["params_entrenables"],
            "params_entrenables_f2": r["info_f2"]["params_entrenables"],
            "params_total": r["info_f2"]["params_total"],
            "n_val_evaluadas": n})
        for f, k in (("f1", "fase1"), ("f2", "fase2")):
            mlflow.log_params({f"{f}_epocas_ejecutadas": r[k]["epocas_ejecutadas"],
                               f"{f}_mejor_epoca": r[k]["mejor_epoca"],
                               f"{f}_parada_anticipada": r[k]["parada_anticipada"]})
            mlflow.log_metrics({f"{f}_mejor_val_loss": r[k]["mejor_val_loss"],
                                f"{f}_s_por_paso_medio": r[k]["s_por_paso_medio"]})
        m = {k: v for k, v in prefijar(r["metricas"], "val").items() if not math.isnan(v)}
        mlflow.log_metrics(m)
        mlflow.log_metric("tiempo_total_s", round(time.time() - t0, 1))

        # Artefactos de docs/03_mlflow_convenciones.md §6
        mlflow.log_figure(figura_curvas(r["historial"], run_name), "curvas.png")
        y = df_val.label.values[:n]
        mlflow.log_figure(figura_confusion(matriz_confusion(y, r["probs"].argmax(1)), run_name),
                          "confusion_val_norm.png")
        mlflow.log_dict(informe_clasificacion(y, r["probs"]), "classification_report_val.json")
        preds = tabla_predicciones(df_val.iloc[:n], r["probs"])
        mlflow.log_text(preds.to_csv(index=False), "predicciones_val.csv")
        if registrar_modelo:
            mlflow.tensorflow.log_model(model, name="modelo")
        r["run_id"] = mlflow.active_run().info.run_id
    r["run_name"], r["state_dir"], r["tiempo_total_s"] = run_name, str(state_dir), time.time() - t0
    return r
