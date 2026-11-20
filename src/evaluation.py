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


def lanzar_evaluacion_test_v3(arch: str, cfg: dict, autor: str = "alex") -> dict:
    """Una pasada de prueba de la v3 y el bootstrap pareado contra la v2 de esa red.

    Carga `models:/ham10000-<arch>@v3-representativo`. No toca los runs de la v2
    ni los alias `final` y `seleccionado`. No se llama hasta cerrar la selección v3.
    """
    import mlflow
    import pandas as pd
    from mlflow.tracking import MlflowClient

    from src.bootstrap import RUNS_PRUEBA, diferencia_pareada
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

    version = client.get_model_version_by_alias(f"ham10000-{arch}", "v3-representativo")
    df = split_df("test")
    ds = make_dataset("test", arch, training=False, batch_size=cfg["training"]["batch_size"])
    modelo = mlflow.tensorflow.load_model(f"models:/ham10000-{arch}@v3-representativo")
    probs = modelo.predict(ds, verbose=0)
    if len(probs) != len(df):
        raise RuntimeError(f"{len(df)} imágenes de prueba y {len(probs)} predicciones.")
    y = df.label.to_numpy()
    metricas = calcular_metricas(y, probs)
    tabla = tabla_predicciones(df, probs)
    v2 = pd.read_csv(client.download_artifacts(RUNS_PRUEBA[arch], "predicciones_test.csv"))
    pareado = diferencia_pareada(tabla, v2)
    lineas = [
        f"# {arch}: v3 menos v2, argmax, bootstrap pareado",
        "",
        "| Métrica | Diferencia | Intervalo 95 % | Incluye el 0 |",
        "|---|---|---|---|",
    ]
    for clave in ("macro_f1", "mel_sensitivity"):
        d = pareado[clave]
        lineas.append(
            f"| {clave} | {d['punto']:.3f} | [{d['bajo']:.3f}; {d['alto']:.3f}] | {d['incluye_cero']} |"
        )
    mlflow.set_experiment(experimento)
    tags = standard_tags(cfg, modulo="m2", fase="evaluacion-test", autor=autor,
                         arch=arch, experimento="v3")
    with mlflow.start_run(run_name=nombre, tags=tags):
        mlflow.log_params({"run_id_modelo": version.run_id, "modelo_version": version.version,
                           "run_id_v2": RUNS_PRUEBA[arch]})
        mlflow.log_metrics({k: v for k, v in prefijar(metricas, "test").items() if v == v})
        mlflow.log_figure(figura_confusion(matriz_confusion(y, probs.argmax(1)), f"{arch} v3"),
                          "confusion_test_norm.png")
        mlflow.log_text(tabla.to_csv(index=False), "predicciones_test.csv")
        mlflow.log_text("\n".join(lineas) + "\n", "bootstrap_v3_vs_v2.md")
        mlflow.set_tag("test_evaluado", "true")
        mlflow.set_tag("estado", "final")
        run_id = mlflow.active_run().info.run_id
    return {"run_id": run_id, "arch": arch, "pareado": pareado,
            "test_macro_f1": metricas["macro_f1"],
            "test_mel_sensitivity": metricas["mel_sensitivity"]}
