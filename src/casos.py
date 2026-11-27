"""Selección de casos y ejecución de Grad-CAM del Módulo 3.

Casos por red (modelos `ham10000-<arch>@v3-representativo`, predicciones de `m2-test-v3-<arch>`):

  * ResNet50: TODOS los falsos negativos de melanoma + 30 verdaderos positivos.
  * DenseNet121 y EfficientNetB0: 30 falsos negativos estratificados por la clase asignada
    (nv / bkl / otras) + 30 verdaderos positivos.

FN = melanoma real que el argmax no llama melanoma (el mismo criterio que `falsos_negativos_mel.csv`).
TP = melanoma real que el argmax llama melanoma. En las muestras se toma como máximo una imagen por
`lesion_id`, para no anotar dos veces la misma lesión. Todo con semilla fija y orden determinista.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLASES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
ESTRATOS = ("nv", "bkl", "otras")
N_MUESTRA = 30
MINIMO_ESTRATO = 5
SEMILLA_REPRESENTATIVA = {"resnet50": 42, "densenet121": 42, "efficientnetb0": 123}
COLUMNAS_ANOTACION = ["zona_atencion", "criterios_abcde", "comentario"]
ZONAS = ("lesión", "borde", "fuera: regla, esquina, pelo, burbuja, tinta", "difusa")


def estrato(y_pred: str) -> str:
    return y_pred if y_pred in ("nv", "bkl") else "otras"


def asignar_cuotas(tamanos: dict[str, int], n: int = N_MUESTRA, minimo: int = MINIMO_ESTRATO) -> dict[str, int]:
    """Reparto proporcional (D'Hondt) de `n` casos entre estratos, con un mínimo por estrato.

    Si no hay `n` casos en total, se toman todos. Nunca se pide más de lo que hay en un estrato.
    """
    if sum(tamanos.values()) <= n:
        return dict(tamanos)
    cuota = {k: min(minimo, v) for k, v in tamanos.items()}
    while sum(cuota.values()) < n:
        candidatos = [k for k in tamanos if cuota[k] < tamanos[k]]
        k = max(candidatos, key=lambda c: (tamanos[c] / (cuota[c] + 1), -ESTRATOS.index(c) if c in ESTRATOS else 0))
        cuota[k] += 1
    return cuota


def _una_por_lesion(df: pd.DataFrame, semilla: int) -> pd.DataFrame:
    """Orden aleatorio reproducible y una sola imagen por lesión."""
    df = df.sort_values("image_id").reset_index(drop=True)
    orden = np.random.default_rng(semilla).permutation(len(df))
    return df.iloc[orden].drop_duplicates("lesion_id", keep="first").reset_index(drop=True)


def muestra_fn_estratificada(fn: pd.DataFrame, semilla: int, n: int = N_MUESTRA,
                             minimo: int = MINIMO_ESTRATO) -> pd.DataFrame:
    base = _una_por_lesion(fn, semilla)
    base["estrato"] = base.y_pred.map(estrato)
    cuotas = asignar_cuotas(base.estrato.value_counts().to_dict(), n, minimo)
    partes = [base[base.estrato == e].head(c) for e, c in cuotas.items() if c > 0]
    return pd.concat(partes).sort_values("image_id").reset_index(drop=True)


def muestra_tp(tp: pd.DataFrame, semilla: int, n: int = N_MUESTRA) -> pd.DataFrame:
    base = _una_por_lesion(tp, semilla)
    return base.head(n).sort_values("image_id").reset_index(drop=True)


def seleccionar_casos(arch: str, pred: pd.DataFrame, fn_csv: pd.DataFrame, semilla: int,
                      n: int = N_MUESTRA) -> pd.DataFrame:
    """Tabla de casos de una red, con las columnas de anotación vacías."""
    pred = pred.copy()
    mel = pred[pred.y_true == "mel"]
    fn = mel[mel.y_pred != "mel"]
    tp = mel[mel.y_pred == "mel"]
    if set(fn.image_id) != set(fn_csv.image_id):
        raise RuntimeError(f"{arch}: los FN de predicciones_test.csv y falsos_negativos_mel.csv no coinciden")
    pred["p_pred"] = [r[f"p_{r['y_pred']}"] for _, r in pred.iterrows()]
    pred = pred.set_index("image_id")
    fn = fn[["image_id", "lesion_id", "y_pred", "p_mel"]].assign(p_pred=pred.loc[fn.image_id, "p_pred"].values)
    tp = tp[["image_id", "lesion_id", "y_pred", "p_mel"]].assign(p_pred=pred.loc[tp.image_id, "p_pred"].values)
    if arch == "resnet50":
        fn_sel = fn.sort_values("image_id").assign(estrato=lambda d: d.y_pred.map(estrato))
    else:
        fn_sel = muestra_fn_estratificada(fn, semilla, n)
    tp_sel = muestra_tp(tp, semilla, n).assign(estrato="mel")
    casos = pd.concat([fn_sel.assign(tipo_caso="FN"), tp_sel.assign(tipo_caso="TP")], ignore_index=True)
    casos = casos.rename(columns={"y_pred": "clase_predicha"}).assign(red=arch)
    casos["figura"] = ""
    for c in COLUMNAS_ANOTACION:
        casos[c] = ""
    orden = ["image_id", "lesion_id", "red", "tipo_caso", "clase_predicha", "p_mel", "p_pred",
             "estrato", "figura", *COLUMNAS_ANOTACION]
    return casos[orden].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ejecución en Colab (necesita TensorFlow, MLflow y las imágenes en /content)
# ---------------------------------------------------------------------------

def _run_de_prueba(client, cfg: dict, arch: str):
    exp = client.get_experiment_by_name(cfg["mlflow"]["experiments"]["m2_test_v3"])
    if exp is None:
        raise RuntimeError("No existe el experimento m2-evaluacion-test-v3.")
    candidatos = [r for r in client.search_runs([exp.experiment_id], max_results=100)
                  if (r.info.run_name or r.data.tags.get("mlflow.runName")) == f"m2-test-v3-{arch}"
                  and r.data.tags.get("test_evaluado") == "true"]
    if len(candidatos) != 1:
        raise RuntimeError(f"m2-test-v3-{arch}: se esperaba 1 run evaluado y hay {len(candidatos)}.")
    return candidatos[0]


def _preparar(arch: str, cfg: dict):
    """Comprueba el alias y la semilla, descarga las entradas, elige los casos y carga el modelo."""
    import keras
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.evaluation import RUNS_V3
    from src.explain import CAPAS_CONV, Explicador
    from src.tracking import setup_mlflow

    setup_mlflow(cfg)
    client = MlflowClient()
    semilla = int(cfg["gradcam"]["sampling_seed"])
    nombre_modelo = f"ham10000-{arch}"
    version = client.get_model_version_by_alias(nombre_modelo, "v3-representativo")
    if version.run_id != RUNS_V3[arch]:
        raise RuntimeError(f"{nombre_modelo}@v3-representativo apunta a {version.run_id}, no a {RUNS_V3[arch]}.")
    run_modelo = client.get_run(version.run_id)
    if int(run_modelo.data.tags.get("seed", -1)) != SEMILLA_REPRESENTATIVA[arch]:
        raise RuntimeError(f"{arch}: la semilla del modelo no es la {SEMILLA_REPRESENTATIVA[arch]}.")
    run_pred = _run_de_prueba(client, cfg, arch)
    pred = pd.read_csv(client.download_artifacts(run_pred.info.run_id, "predicciones_test.csv"))
    fn_csv = pd.read_csv(client.download_artifacts(run_pred.info.run_id, "falsos_negativos_mel.csv"))
    casos = seleccionar_casos(arch, pred, fn_csv, semilla)

    keras.backend.clear_session()
    modelo = mlflow.tensorflow.load_model(f"models:/{nombre_modelo}@v3-representativo")
    if not hasattr(modelo, "get_layer"):
        raise RuntimeError("El modelo cargado no es un modelo Keras con capas accesibles.")
    capa = cfg["gradcam"]["last_conv_layer"][arch]
    assert capa == CAPAS_CONV[arch]
    expl = Explicador(modelo, arch, capa)

    return client, version, run_pred, casos, modelo, expl, capa, semilla


def lanzar_gradcam(arch: str, cfg: dict, salida: str | Path = "/content/m3_salida",
                   autor: str = "christian", n_cordura: int = 20) -> dict:
    """Carga el modelo y las entradas desde MLflow, genera las figuras y registra el run `m3-gradcam-<arch>`."""
    import keras
    import matplotlib.pyplot as plt
    import mlflow
    import tensorflow as tf

    from src.explain import cargar_imagen, figura_caso, n_capas_base_con_pesos, prueba_cordura
    from src.tracking import standard_tags

    client, version, run_pred, casos, modelo, expl, capa, semilla = _preparar(arch, cfg)
    nombre_modelo = f"ham10000-{arch}"

    carpeta = Path(salida) / "gradcam" / arch
    dif_max = 0.0
    vacios = {"gradcam": 0, "gradcam++": 0}          # mapas de la clase mel que salen todo ceros
    for i, fila in casos.iterrows():
        uint8, x = cargar_imagen(fila.image_id, arch)
        p = expl.probabilidades(x)
        dif_max = max(dif_max, abs(float(p[CLASES.index("mel")]) - float(fila.p_mel)))
        idx_pred = CLASES.index(fila.clase_predicha)
        etiquetas = [(fila.clase_predicha, idx_pred)]
        if fila.clase_predicha != "mel":
            etiquetas.append(("mel", CLASES.index("mel")))
        filas = [(nombre, {m: expl.mapa(x, k, m) for m in ("gradcam", "gradcam++")})
                 for nombre, k in etiquetas]
        for m, mapa_mel in filas[-1][1].items():
            vacios[m] += int(mapa_mel.max() == 0)
        titulo = (f"{arch} · {fila.tipo_caso} · {fila.image_id} · predicha: {fila.clase_predicha} · "
                  f"p(mel)={fila.p_mel:.3f}")
        fig = figura_caso(uint8, filas, titulo)
        ruta = carpeta / fila.tipo_caso / f"{fila.image_id}.png"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, dpi=110)
        plt.close(fig)
        casos.loc[i, "figura"] = f"gradcam/{arch}/{fila.tipo_caso}/{fila.image_id}.png"
    if dif_max > 5e-3:
        raise RuntimeError(f"{arch}: p(mel) recalculada difiere {dif_max:.4f} de la del CSV. "
                           "No es el mismo modelo o las imágenes no se preparan igual.")

    tps = casos[casos.tipo_caso == "TP"].image_id.head(n_cordura)
    cordura = prueba_cordura(modelo, arch, [cargar_imagen(i, arch)[1] for i in tps], CLASES.index("mel"))

    mlflow.set_experiment(cfg["mlflow"]["experiments"]["m3"])
    tags = standard_tags(cfg, modulo="m3", fase="gradcam", autor=autor, arch=arch,
                         seed=SEMILLA_REPRESENTATIVA[arch], modelo_version=version.version,
                         run_id_modelo=version.run_id, run_id_predicciones=run_pred.info.run_id)
    with mlflow.start_run(run_name=f"m3-gradcam-{arch}", tags=tags):
        mlflow.log_params({
            "layer_name": capa, "metodos": "gradcam,gradcam++", "puntuacion": "logit",
            "modelo": f"{nombre_modelo}@v3-representativo", "modelo_version": version.version,
            "semilla_muestreo": semilla, "n_muestra": N_MUESTRA, "minimo_estrato": MINIMO_ESTRATO,
            "tf_version": tf.__version__, "keras_version": keras.__version__,
        })
        mlflow.log_metrics({
            "n_fn": float((casos.tipo_caso == "FN").sum()), "n_tp": float((casos.tipo_caso == "TP").sum()),
            "dif_max_p_mel": dif_max, "n_mapas_mel_vacios_gradcam": float(vacios["gradcam"]),
            "n_mapas_mel_vacios_gradcampp": float(vacios["gradcam++"]), "n_capas_aleatorizables": float(n_capas_base_con_pesos(modelo)),
        })
        for (metodo, f), rho in cordura["rho"].items():
            nombre = f"{metodo.replace('+', 'p')}_f{int(f * 100)}"
            if rho == rho:
                mlflow.log_metric(f"adebayo_rho_{nombre}", rho)
            mlflow.log_metric(f"adebayo_vacios_{nombre}", float(cordura["vacios"][(metodo, f)]))
        mlflow.log_text(casos.to_csv(index=False), f"m3_casos_{arch}.csv")
        mlflow.log_artifacts(str(carpeta), artifact_path=f"gradcam/{arch}")
        mlflow.set_tag("estado", "final")
        run_id = mlflow.active_run().info.run_id
    return {"arch": arch, "run_id": run_id, "casos": casos, "cordura": cordura,
            "modelo_version": version.version, "run_id_modelo": version.run_id,
            "dif_max_p_mel": dif_max}


def publicar_casos(casos: pd.DataFrame, resultados: list[dict], cfg: dict, salida: str | Path,
                   autor: str = "christian") -> str:
    """Escribe `m3_casos.csv` y lo registra, con la versión de cada modelo, en el run `m3-gradcam-casos`."""
    import mlflow

    from src.tracking import setup_mlflow, standard_tags

    setup_mlflow(cfg)
    ruta = Path(salida) / "m3_casos.csv"
    casos.to_csv(ruta, index=False)
    mlflow.set_experiment(cfg["mlflow"]["experiments"]["m3"])
    tags = standard_tags(cfg, modulo="m3", fase="anotacion", autor=autor)
    with mlflow.start_run(run_name="m3-gradcam-casos", tags=tags):
        params = {}
        for r in resultados:
            params[f"modelo_version_{r['arch']}"] = r["modelo_version"]
            params[f"run_id_modelo_{r['arch']}"] = r["run_id_modelo"]
            params[f"run_id_m3_{r['arch']}"] = r["run_id"]
        mlflow.log_params(params)
        mlflow.log_metric("n_casos", float(len(casos)))
        mlflow.log_artifact(str(ruta))
        mlflow.set_tag("estado", "final")
        return mlflow.active_run().info.run_id


def repetir_cordura(arch: str, cfg: dict, run_id: str, n_cordura: int = 20) -> dict:
    """Repite solo la prueba de cordura y la registra en un run `m3-gradcam-<arch>` ya existente.

    Los valores nuevos sustituyen a los anteriores (mismas claves `adebayo_*`; MLflow conserva el historial).
    El run queda con el tag `cordura=v2` (aleatorización en sitio, sin clonar el modelo).
    """
    import mlflow

    from src.explain import cargar_imagen, prueba_cordura

    _, version, _, casos, modelo, _, _, _ = _preparar(arch, cfg)
    tps = casos[casos.tipo_caso == "TP"].image_id.head(n_cordura)
    cordura = prueba_cordura(modelo, arch, [cargar_imagen(i, arch)[1] for i in tps], CLASES.index("mel"))
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("cordura", "v2")
        mlflow.set_tag("n_cordura", str(len(tps)))
        for (metodo, f), rho in cordura["rho"].items():
            nombre = f"{metodo.replace('+', 'p')}_f{int(f * 100)}"
            if rho == rho:
                mlflow.log_metric(f"adebayo_rho_{nombre}", rho)
            mlflow.log_metric(f"adebayo_vacios_{nombre}", float(cordura["vacios"][(metodo, f)]))
    return {"arch": arch, "run_id": run_id, "cordura": cordura, "n": len(tps)}
