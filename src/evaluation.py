"""Evaluación en el conjunto de prueba. Una sola vez por arquitectura.

Carga `models:/ham10000-<arch>@final`, predice el split de prueba y registra
las métricas `test_*` con argmax y `test_umbral_*` con el `umbral_mel` fijado
en validación. Si el run `m2-test-<arch>` ya tiene `test_evaluado=true`, se niega.
"""
from __future__ import annotations

import pandas as pd

from src.metrics import (CLASES, aplicar_umbral_mel, calcular_metricas, matriz_confusion,
                         prefijar, tabla_predicciones)


def clase_minima(metricas: dict) -> str:
    """Clases empatadas en la sensibilidad más baja. El orden es el de CLASES."""
    peor = metricas["min_class_sensitivity"]
    return ",".join(c for c in CLASES if metricas[f"sensitivity_{c}"] == peor)


def falsos_negativos_mel(tabla: pd.DataFrame) -> pd.DataFrame:
    """Melanomas de prueba que el argmax no llama melanoma.

    `recuperado_umbral` es True si la regla p(mel) ≥ umbral sí los llama melanoma.
    """
    fn = tabla[(tabla.y_true == "mel") & (tabla.y_pred != "mel")].copy()
    fn["recuperado_umbral"] = fn.y_pred_umbral == "mel"
    columnas = ["image_id", "lesion_id", "y_pred", "p_mel", "y_pred_umbral", "recuperado_umbral"]
    return fn[columnas].reset_index(drop=True)


def evaluacion_ya_hecha(runs, arch: str) -> bool:
    """True si ya hay un run de prueba de esa red marcado como evaluado."""
    nombre = f"m2-test-{arch}"
    for run in runs:
        tags = run.data.tags
        run_name = run.info.run_name or tags.get("mlflow.runName")
        if run_name == nombre and tags.get("test_evaluado") == "true":
            return True
    return False


def lanzar_evaluacion_test(arch: str, cfg: dict, autor: str = "alex") -> dict:
    """Una pasada por el conjunto de prueba. No se puede repetir."""
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.data import make_dataset, split_df
    from src.tracking import setup_mlflow, standard_tags
    from src.training import figura_confusion

    if arch not in ("resnet50", "densenet121", "efficientnetb0"):
        raise ValueError(arch)
    setup_mlflow(cfg)
    client = MlflowClient()
    experimento = cfg["mlflow"]["experiments"]["m2_test"]
    exp = client.get_experiment_by_name(experimento)
    if exp is None:
        mlflow.set_experiment(experimento)
        exp = client.get_experiment_by_name(experimento)
    previos = client.search_runs([exp.experiment_id], max_results=100)
    if evaluacion_ya_hecha(previos, arch):
        raise RuntimeError(
            f"m2-test-{arch} ya tiene test_evaluado=true. La prueba no se repite.")

    nombre_modelo = f"ham10000-{arch}"
    version = client.get_model_version_by_alias(nombre_modelo, "final")
    origen = client.get_run(version.run_id)
    if "umbral_mel" not in origen.data.params:
        raise RuntimeError(
            f"El run final de {arch} no tiene el parámetro umbral_mel. "
            "No se abre el conjunto de prueba.")
    # umbral_mel_preciso es el mismo corte, con los decimales que el redondeo a 6
    # se comía. Sin él, DenseNet y EfficientNet se quedan por debajo de 0,80.
    clave_umbral = "umbral_mel_preciso" if "umbral_mel_preciso" in origen.data.params else "umbral_mel"
    umbral = float(origen.data.params[clave_umbral])

    df = split_df("test")
    ds = make_dataset("test", arch, training=False, batch_size=cfg["training"]["batch_size"])
    modelo = mlflow.tensorflow.load_model(f"models:/{nombre_modelo}@final")
    probs = modelo.predict(ds, verbose=0)
    if len(probs) != len(df):
        raise RuntimeError(f"{len(df)} imágenes de prueba y {len(probs)} predicciones.")

    y = df.label.to_numpy()
    metricas = calcular_metricas(y, probs)
    pred_umbral = aplicar_umbral_mel(probs, umbral)
    metricas_umbral = calcular_metricas(y, probs, y_pred=pred_umbral)
    tabla = tabla_predicciones(df, probs)
    tabla["y_pred_umbral"] = [CLASES[i] for i in pred_umbral]
    fn = falsos_negativos_mel(tabla)
    minima = clase_minima(metricas)
    minima_umbral = clase_minima(metricas_umbral)

    mlflow.set_experiment(experimento)
    tags = standard_tags(cfg, modulo="m2", fase="evaluacion-test", autor=autor, arch=arch)
    with mlflow.start_run(run_name=f"m2-test-{arch}", tags=tags):
        mlflow.log_params({
            "umbral_mel": umbral,
            "run_id_modelo": version.run_id,
            "modelo_version": version.version,
            "test_clase_minima": minima,
            "test_umbral_clase_minima": minima_umbral,
        })
        mlflow.log_metrics({k: v for k, v in prefijar(metricas, "test").items() if v == v})
        mlflow.log_metrics({k: v for k, v in prefijar(metricas_umbral, "test_umbral").items() if v == v})
        mlflow.log_metrics({
            "test_n_fn_mel": float(len(fn)),
            "test_n_fn_mel_recuperados": float(fn.recuperado_umbral.sum()) if len(fn) else 0.0,
        })
        mlflow.log_figure(figura_confusion(matriz_confusion(y, probs.argmax(1)), f"{arch} argmax"),
                          "confusion_test_norm.png")
        mlflow.log_figure(figura_confusion(matriz_confusion(y, pred_umbral), f"{arch} umbral {umbral:.4f}"),
                          "confusion_test_umbral_norm.png")
        mlflow.log_text(tabla.to_csv(index=False), "predicciones_test.csv")
        mlflow.log_text(fn.to_csv(index=False), "falsos_negativos_mel.csv")
        mlflow.set_tag("test_evaluado", "true")
        mlflow.set_tag("estado", "final")
        run_id = mlflow.active_run().info.run_id
    por_clase = fn.y_pred.value_counts().to_dict() if len(fn) else {}
    return {
        "run_id": run_id,
        "arch": arch,
        "umbral_mel": umbral,
        "run_id_modelo": version.run_id,
        "metricas": metricas,
        "metricas_umbral": metricas_umbral,
        "clase_minima": minima,
        "clase_minima_umbral": minima_umbral,
        "fn_total": int(len(fn)),
        "fn_por_clase": por_clase,
        "fn_recuperados": int(fn.recuperado_umbral.sum()) if len(fn) else 0,
    }


NOTA_PRUEBA_V3 = (
    "El conjunto de prueba ya se había usado para la v2. "
    "Esta pasada es complementaria y no sustituye a esa evaluación."
)
RUNS_V3 = {
    "resnet50": "efb52439ac1548458c1d8bd255c1d72e",
    "densenet121": "9f05dd1428cd4a5bb34766e39198f56e",
    "efficientnetb0": "4d4a61ce581348b0986a3fd3db84ba06",
}


def comprobar_alias_v3(client) -> None:
    """`v3-representativo` en la versión 3. `final` y `seleccionado` siguen en la v2."""
    for arch, run_id in RUNS_V3.items():
        nombre = f"ham10000-{arch}"
        modelo = client.get_registered_model(nombre)
        alias = {nombre_alias: str(version) for nombre_alias, version in dict(modelo.aliases).items()}
        if alias.get("v3-representativo") != "3":
            raise RuntimeError(f"{nombre}: v3-representativo está en la versión {alias.get('v3-representativo')}.")
        if alias.get("final") != "2":
            raise RuntimeError(f"{nombre}: final está en la versión {alias.get('final')}.")
        if arch == "efficientnetb0":
            if alias.get("seleccionado") != "2":
                raise RuntimeError(f"{nombre}: seleccionado está en la versión {alias.get('seleccionado')}.")
        elif "seleccionado" in alias:
            raise RuntimeError(f"{nombre} no debería llevar el alias seleccionado.")
        version = client.get_model_version_by_alias(nombre, "v3-representativo")
        if version.run_id != run_id:
            raise RuntimeError(f"{nombre}@v3-representativo apunta a {version.run_id}, no a {run_id}.")


def anotar_umbrales_v3(cfg: dict) -> list[dict]:
    """Escribe `umbral_mel` en las tres representativas, medido en validación.

    El valor guardado es el corte exacto. Redondearlo a seis decimales baja la
    sensibilidad en melanoma de 0,801 a 0,795. No abre el conjunto de prueba.
    Si el parámetro ya está y coincide, no lo vuelve a escribir.
    """
    import numpy as np
    import pandas as pd
    from mlflow.tracking import MlflowClient

    from src.metrics import CLASES, barrer_umbral_mel
    from src.tracking import setup_mlflow

    if cfg.get("config_version") != 3:
        raise RuntimeError("Los umbrales de la v3 se anotan con configs/m2_v3.yaml.")
    setup_mlflow(cfg)
    client = MlflowClient()
    comprobar_alias_v3(client)
    salida = []
    for arch, run_id in RUNS_V3.items():
        ruta = client.download_artifacts(run_id, "predicciones_val.csv")
        df = pd.read_csv(ruta)
        probs = df[[f"p_{c}" for c in CLASES]].to_numpy(dtype=np.float64)
        y = df.y_true.map({c: i for i, c in enumerate(CLASES)}).to_numpy()
        corte = barrer_umbral_mel(y, probs)
        if not corte["alcanza"]:
            raise RuntimeError(f"{arch} no llega a sensibilidad 0,80 en validación.")
        preciso = repr(float(corte["t"]))
        params = client.get_run(run_id).data.params
        if "umbral_mel" not in params:
            client.log_param(run_id, "umbral_mel", preciso)
            client.log_param(run_id, "umbral_mel_preciso", preciso)
        elif params.get("umbral_mel_preciso", params["umbral_mel"]) != preciso:
            raise RuntimeError(f"{arch} ya tiene un umbral_mel distinto de {preciso}.")
        salida.append({"arch": arch, "run_id": run_id, **corte})
    return salida


def _texto_bootstrap_v3(arch: str, intervalos: dict, contra_v2: dict) -> str:
    from src.bootstrap import METRICAS, _fmt
    lineas = [
        f"# {arch}: intervalos en prueba y diferencia frente a la v2",
        "",
        NOTA_PRUEBA_V3,
        "",
        "Bootstrap estratificado por clase y agrupado por `lesion_id`: "
        "2 000 réplicas, semilla 42, percentil 2,5–97,5. "
        "El puntual es el de la muestra completa.",
        "",
        "| Criterio | macro-F1 | Balanced accuracy | Sens. mel | Prec. mel | Sens. df | AUC mel |",
        "|---|---|---|---|---|---|---|",
    ]
    for criterio in ("argmax", "umbral"):
        celdas = []
        for metrica in METRICAS:
            m = intervalos[criterio][metrica]
            celdas.append(f"{_fmt(m['punto'])} [{_fmt(m['bajo'])}; {_fmt(m['alto'])}]")
        lineas.append(f"| {criterio} | " + " | ".join(celdas) + " |")
    lineas += [
        "",
        f"## v3 menos v2, misma arquitectura (`{arch}`)",
        "",
        "El criterio umbral usa el corte de validación de cada versión.",
        "",
        "| Criterio | Métrica | Diferencia | Intervalo 95 % | Incluye el 0 |",
        "|---|---|---|---|---|",
    ]
    for criterio in ("argmax", "umbral"):
        for metrica in ("macro_f1", "mel_sensitivity"):
            d = contra_v2[criterio][metrica]
            lineas.append(
                f"| {criterio} | {metrica} | {_fmt(d['punto'])} | "
                f"[{_fmt(d['bajo'])}; {_fmt(d['alto'])}] | {d['incluye_cero']} |"
            )
    lineas.append("")
    return "\n".join(lineas)


def lanzar_evaluacion_test_v3(arch: str, cfg: dict, autor: str = "alex") -> dict:
    """Una pasada de prueba de la v3. Argmax y umbral, más el bootstrap de esa red.

    Carga `models:/ham10000-<arch>@v3-representativo`. No toca los runs de la v2
    ni los alias `final` y `seleccionado`. Si `m2-test-v3-<arch>` ya tiene
    `test_evaluado=true`, se niega.
    """
    import mlflow
    import pandas as pd
    from mlflow.tracking import MlflowClient

    from src.bootstrap import N_REPLICAS, RUNS_PRUEBA, SEMILLA, diferencia_pareada, intervalos_tabla
    from src.data import make_dataset, split_df
    from src.metrics import calcular_metricas, matriz_confusion, prefijar, tabla_predicciones
    from src.tracking import setup_mlflow, standard_tags
    from src.training import figura_confusion

    if cfg.get("config_version") != 3:
        raise RuntimeError("La evaluación v3 lee configs/m2_v3.yaml.")
    if arch not in RUNS_PRUEBA:
        raise ValueError(arch)
    setup_mlflow(cfg)
    client = MlflowClient()
    comprobar_alias_v3(client)
    experimento = cfg["mlflow"]["experiments"]["m2_test_v3"]
    exp = client.get_experiment_by_name(experimento)
    if exp is None:
        mlflow.set_experiment(experimento)
        exp = client.get_experiment_by_name(experimento)
    nombre = f"m2-test-v3-{arch}"
    previos = client.search_runs([exp.experiment_id], max_results=100)
    for run in previos:
        run_name = run.info.run_name or run.data.tags.get("mlflow.runName")
        if run_name == nombre and run.data.tags.get("test_evaluado") == "true":
            raise RuntimeError(f"{nombre} ya tiene test_evaluado=true. La prueba no se repite.")

    origen = client.get_run(RUNS_V3[arch])
    if "umbral_mel" not in origen.data.params:
        raise RuntimeError(f"{arch} no tiene umbral_mel. No se abre el conjunto de prueba.")
    clave = "umbral_mel_preciso" if "umbral_mel_preciso" in origen.data.params else "umbral_mel"
    umbral = float(origen.data.params[clave])
    version = client.get_model_version_by_alias(f"ham10000-{arch}", "v3-representativo")
    df = split_df("test")
    ds = make_dataset("test", arch, training=False, batch_size=cfg["training"]["batch_size"])
    modelo = mlflow.tensorflow.load_model(f"models:/ham10000-{arch}@v3-representativo")
    probs = modelo.predict(ds, verbose=0)
    if len(probs) != len(df):
        raise RuntimeError(f"{len(df)} imágenes de prueba y {len(probs)} predicciones.")

    y = df.label.to_numpy()
    metricas = calcular_metricas(y, probs)
    pred_umbral = aplicar_umbral_mel(probs, umbral)
    metricas_umbral = calcular_metricas(y, probs, y_pred=pred_umbral)
    tabla = tabla_predicciones(df, probs)
    tabla["y_pred_umbral"] = [CLASES[i] for i in pred_umbral]
    fn = falsos_negativos_mel(tabla)
    v2 = pd.read_csv(client.download_artifacts(RUNS_PRUEBA[arch], "predicciones_test.csv"))
    intervalos = intervalos_tabla(tabla, N_REPLICAS, SEMILLA)
    contra_v2 = {
        criterio: diferencia_pareada(tabla, v2, N_REPLICAS, SEMILLA, criterio)
        for criterio in ("argmax", "umbral")
    }
    texto = _texto_bootstrap_v3(arch, intervalos, contra_v2)

    mlflow.set_experiment(experimento)
    tags = standard_tags(cfg, modulo="m2", fase="evaluacion-test", autor=autor,
                         arch=arch, experimento="v3", analisis="complementario")
    with mlflow.start_run(run_name=nombre, tags=tags):
        mlflow.set_tag("nota", NOTA_PRUEBA_V3)
        mlflow.log_params({
            "umbral_mel": umbral,
            "run_id_modelo": version.run_id,
            "modelo_version": version.version,
            "run_id_v2": RUNS_PRUEBA[arch],
            "test_clase_minima": clase_minima(metricas),
            "test_umbral_clase_minima": clase_minima(metricas_umbral),
            "n_replicas": N_REPLICAS,
            "semilla_bootstrap": SEMILLA,
        })
        mlflow.log_metrics({k: v for k, v in prefijar(metricas, "test").items() if v == v})
        mlflow.log_metrics({k: v for k, v in prefijar(metricas_umbral, "test_umbral").items() if v == v})
        mlflow.log_metrics({
            "test_n_fn_mel": float(len(fn)),
            "test_n_fn_mel_recuperados": float(fn.recuperado_umbral.sum()) if len(fn) else 0.0,
        })
        mlflow.log_figure(figura_confusion(matriz_confusion(y, probs.argmax(1)), f"{arch} v3 argmax"),
                          "confusion_test_norm.png")
        mlflow.log_figure(figura_confusion(matriz_confusion(y, pred_umbral), f"{arch} v3 umbral"),
                          "confusion_test_umbral_norm.png")
        mlflow.log_text(tabla.to_csv(index=False), "predicciones_test.csv")
        mlflow.log_text(fn.to_csv(index=False), "falsos_negativos_mel.csv")
        mlflow.log_text(texto, "bootstrap_v3.md")
        mlflow.log_text(NOTA_PRUEBA_V3 + "\n", "nota_prueba.txt")
        mlflow.set_tag("test_evaluado", "true")
        mlflow.set_tag("estado", "final")
        run_id = mlflow.active_run().info.run_id
    return {
        "run_id": run_id, "arch": arch, "umbral_mel": umbral,
        "test_macro_f1": metricas["macro_f1"],
        "test_mel_sensitivity": metricas["mel_sensitivity"],
        "test_umbral_macro_f1": metricas_umbral["macro_f1"],
        "test_umbral_mel_sensitivity": metricas_umbral["mel_sensitivity"],
    }


def publicar_bootstrap_redes_v3(cfg: dict) -> str:
    """Diferencias pareadas entre las tres redes de la v3, cuando las tres pruebas existen.

    Lo deja en `bootstrap_redes_v3.md` de cada run. No vuelve a predecir.
    """
    import pandas as pd
    from mlflow.tracking import MlflowClient

    from src.bootstrap import N_REPLICAS, SEMILLA, _fmt, diferencias_entre_redes
    from src.tracking import setup_mlflow

    if cfg.get("config_version") != 3:
        raise RuntimeError("La comparación v3 lee configs/m2_v3.yaml.")
    setup_mlflow(cfg)
    client = MlflowClient()
    experimento = cfg["mlflow"]["experiments"]["m2_test_v3"]
    exp = client.get_experiment_by_name(experimento)
    if exp is None:
        raise RuntimeError(f"No existe el experimento {experimento}.")
    runs = client.search_runs([exp.experiment_id], max_results=100)
    ids = {}
    for arch in RUNS_V3:
        nombre = f"m2-test-v3-{arch}"
        hechos = [
            r for r in runs
            if (r.info.run_name or r.data.tags.get("mlflow.runName")) == nombre
            and r.data.tags.get("test_evaluado") == "true"
        ]
        if len(hechos) != 1:
            raise RuntimeError(f"{nombre} tiene que estar evaluado una vez antes de comparar las redes.")
        ids[arch] = hechos[0].info.run_id
    if all("bootstrap_redes_v3.md" in {a.path for a in client.list_artifacts(rid)} for rid in ids.values()):
        return ids["resnet50"]
    tablas = {arch: pd.read_csv(client.download_artifacts(rid, "predicciones_test.csv")) for arch, rid in ids.items()}
    filas = diferencias_entre_redes(tablas, N_REPLICAS, SEMILLA)
    lineas = [
        "# Diferencias pareadas entre las tres redes de la v3",
        "",
        NOTA_PRUEBA_V3,
        "",
        "La diferencia es la primera red menos la segunda, en las mismas 2 000 réplicas "
        "(estratificadas por clase, agrupadas por `lesion_id`, semilla 42).",
        "",
        "| Comparación | Criterio | Métrica | Diferencia | Intervalo 95 % | Incluye el 0 |",
        "|---|---|---|---|---|---|",
    ]
    for fila in filas:
        lineas.append(
            f"| {fila['izquierda']} − {fila['derecha']} | {fila['criterio']} | {fila['metrica']} | "
            f"{_fmt(fila['punto'])} | [{_fmt(fila['bajo'])}; {_fmt(fila['alto'])}] | {fila['incluye_cero']} |"
        )
    texto = "\n".join(lineas) + "\n"
    for rid in ids.values():
        client.log_text(rid, texto, "bootstrap_redes_v3.md")
    return ids["resnet50"]
