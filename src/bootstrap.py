"""Intervalos de confianza en prueba, sin volver a predecir.

Bootstrap estratificado por clase y agrupado por lesión: dentro de cada clase
se remuestrean con reemplazo sus lesiones, y entran todas las imágenes de cada
lesión elegida. Las tres redes comparten las mismas réplicas, así la diferencia
entre redes es pareada. El intervalo es el percentil 2,5–97,5.

No abre el conjunto de prueba y no modifica los runs `m2-test-*`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.metrics import CLASES, IDX_MEL

N_REPLICAS = 2000
SEMILLA = 42
ARQUITECTURAS = ("resnet50", "densenet121", "efficientnetb0")
RUNS_PRUEBA = {
    "resnet50": "3b89d4d64f6f46d390d74d257a31aa33",
    "densenet121": "364bcd927a4b46d4acf897452b611fe7",
    "efficientnetb0": "fd8db8ac7a3140e294820740608294ed",
}
METRICAS = (
    "macro_f1",
    "balanced_accuracy",
    "mel_sensitivity",
    "mel_precision",
    "df_sensitivity",
    "auc_mel",
)
IDX_DF = CLASES.index("df")


def _codificar(tabla: pd.DataFrame) -> dict:
    y = tabla.y_true.map({c: i for i, c in enumerate(CLASES)}).to_numpy(dtype=np.int8)
    pred = tabla.y_pred.map({c: i for i, c in enumerate(CLASES)}).to_numpy(dtype=np.int8)
    umbral = tabla.y_pred_umbral.map({c: i for i, c in enumerate(CLASES)}).to_numpy(dtype=np.int8)
    return {"y": y, "argmax": pred, "umbral": umbral, "p_mel": tabla.p_mel.to_numpy(dtype=np.float64)}


def replicas(tabla: pd.DataFrame, n: int, semilla: int) -> list[np.ndarray]:
    """Índices de fila de cada réplica. La lesión es la unidad de remuestreo."""
    if tabla.groupby("lesion_id").y_true.nunique().max() != 1:
        raise ValueError("Hay una lesión con más de una clase.")
    lesion = tabla.lesion_id.to_numpy()
    y = tabla.y_true.to_numpy()
    bloques = {}
    for clase in CLASES:
        lesiones = sorted(set(lesion[y == clase]))
        if not lesiones:
            raise ValueError(f"La clase {clase} no tiene lesiones.")
        bloques[clase] = [np.flatnonzero(lesion == les) for les in lesiones]
    rng = np.random.default_rng(semilla)
    salida = []
    for _ in range(n):
        partes = []
        for clase in CLASES:
            elegidas = rng.integers(0, len(bloques[clase]), size=len(bloques[clase]))
            partes.extend(bloques[clase][i] for i in elegidas)
        salida.append(np.concatenate(partes))
    return salida


def _medir(y: np.ndarray, pred: np.ndarray, p_mel: np.ndarray) -> np.ndarray:
    cm = np.bincount(y.astype(np.int32) * 7 + pred, minlength=49).reshape(7, 7).astype(float)
    tp = np.diag(cm)
    soporte = cm.sum(axis=1)
    predichos = cm.sum(axis=0)
    sens = np.divide(tp, soporte, out=np.zeros(7), where=soporte > 0)
    prec = np.divide(tp, predichos, out=np.zeros(7), where=predichos > 0)
    f1 = np.divide(2 * prec * sens, prec + sens, out=np.zeros(7), where=(prec + sens) > 0)
    auc = roc_auc_score(y == IDX_MEL, p_mel)
    return np.array([
        f1.mean(), sens.mean(), sens[IDX_MEL], prec[IDX_MEL], sens[IDX_DF], auc,
    ])


def bootstrap_prueba(tablas: dict[str, pd.DataFrame], n: int = N_REPLICAS,
                     semilla: int = SEMILLA) -> dict:
    """Puntual en la muestra completa e intervalo percentil de `n` réplicas."""
    base = tablas["resnet50"]
    for arch, tabla in tablas.items():
        if not tabla.image_id.equals(base.image_id) or not tabla.y_true.equals(base.y_true):
            raise ValueError(f"{arch} no está alineada con resnet50.")
    codigos = {arch: _codificar(tabla) for arch, tabla in tablas.items()}
    puntos = {
        arch: {
            "argmax": _medir(c["y"], c["argmax"], c["p_mel"]),
            "umbral": _medir(c["y"], c["umbral"], c["p_mel"]),
        }
        for arch, c in codigos.items()
    }
    muestras = {
        arch: {"argmax": np.empty((n, len(METRICAS))), "umbral": np.empty((n, len(METRICAS)))}
        for arch in tablas
    }
    for i, idx in enumerate(replicas(base, n, semilla)):
        for arch, c in codigos.items():
            y, p = c["y"][idx], c["p_mel"][idx]
            muestras[arch]["argmax"][i] = _medir(y, c["argmax"][idx], p)
            muestras[arch]["umbral"][i] = _medir(y, c["umbral"][idx], p)
    intervalos = {}
    for arch in tablas:
        intervalos[arch] = {}
        for criterio in ("argmax", "umbral"):
            lo = np.percentile(muestras[arch][criterio], 2.5, axis=0)
            hi = np.percentile(muestras[arch][criterio], 97.5, axis=0)
            intervalos[arch][criterio] = {
                nombre: {"punto": float(puntos[arch][criterio][j]),
                         "bajo": float(lo[j]), "alto": float(hi[j])}
                for j, nombre in enumerate(METRICAS)
            }
    comparaciones = []
    for otra in ("resnet50", "densenet121"):
        for nombre, j in (("macro_f1", 0), ("mel_sensitivity", 2)):
            delta = muestras["efficientnetb0"]["argmax"][:, j] - muestras[otra]["argmax"][:, j]
            lo, hi = np.percentile(delta, [2.5, 97.5])
            punto = float(puntos["efficientnetb0"]["argmax"][j] - puntos[otra]["argmax"][j])
            incluye_cero = lo <= 0 <= hi
            comparaciones.append({
                "contra": otra,
                "metrica": nombre,
                "punto": punto,
                "bajo": float(lo),
                "alto": float(hi),
                "incluye_cero": bool(incluye_cero),
                "frase": "no significativa" if incluye_cero else "diferencia significativa",
            })
    n_lesiones = int(base.lesion_id.nunique())
    return {
        "n": n,
        "semilla": semilla,
        "n_imagenes": len(base),
        "n_lesiones": n_lesiones,
        "intervalos": intervalos,
        "comparaciones": comparaciones,
    }


def _fmt(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


def a_markdown(resultado: dict) -> str:
    """Texto del artefacto `bootstrap_test.md`."""
    lineas = [
        "# Intervalos de confianza en prueba",
        "",
        f"Bootstrap estratificado por clase y agrupado por `lesion_id`: {resultado['n']} réplicas, "
        f"semilla {resultado['semilla']}, percentil 2,5–97,5. "
        f"{resultado['n_imagenes']} imágenes, {resultado['n_lesiones']} lesiones. "
        "Las tres redes comparten las réplicas. El puntual es el de la muestra completa, no la media del bootstrap.",
        "",
        "Runs de origen, sin modificar: "
        + ", ".join(f"`{arch}` `{RUNS_PRUEBA[arch]}`" for arch in ARQUITECTURAS)
        + ".",
        "",
        "| Red | Criterio | macro-F1 | Balanced accuracy | Sens. mel | Prec. mel | Sens. df | AUC mel |",
        "|---|---|---|---|---|---|---|---|",
    ]
    nombres = {
        "macro_f1": "macro-F1",
        "balanced_accuracy": "Balanced accuracy",
        "mel_sensitivity": "Sens. mel",
        "mel_precision": "Prec. mel",
        "df_sensitivity": "Sens. df",
        "auc_mel": "AUC mel",
    }
    for arch in ARQUITECTURAS:
        for criterio in ("argmax", "umbral"):
            celdas = []
            for metrica in METRICAS:
                m = resultado["intervalos"][arch][criterio][metrica]
                celdas.append(f"{_fmt(m['punto'])} [{_fmt(m['bajo'])}; {_fmt(m['alto'])}]")
            lineas.append(f"| {arch} | {criterio} | " + " | ".join(celdas) + " |")
    lineas += [
        "",
        "El AUC de melanoma usa la probabilidad, así que el intervalo de argmax y el de umbral coinciden.",
        "",
        "## Diferencias pareadas con argmax",
        "",
        "EfficientNetB0 menos la otra red, en las mismas réplicas. "
        "«Diferencia significativa» quiere decir que el intervalo del 95 % no contiene el 0.",
        "",
        "| Comparación | Métrica | Diferencia | Intervalo 95 % | Lectura |",
        "|---|---|---|---|---|",
    ]
    for c in resultado["comparaciones"]:
        etiqueta = nombres[c["metrica"]]
        lineas.append(
            f"| EfficientNetB0 − {c['contra']} | {etiqueta} | {_fmt(c['punto'])} | "
            f"[{_fmt(c['bajo'])}; {_fmt(c['alto'])}] | {c['frase']} |"
        )
    lineas.append("")
    return "\n".join(lineas)


def lanzar_bootstrap(cfg: dict, autor: str = "alex", n: int = N_REPLICAS,
                     semilla: int = SEMILLA) -> dict:
    """Crea el run `m2-bootstrap-test` y deja `bootstrap_test.md`. No toca los runs de prueba."""
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.tracking import setup_mlflow, standard_tags

    setup_mlflow(cfg)
    client = MlflowClient()
    experimento = cfg["mlflow"]["experiments"]["m2_test"]
    exp = client.get_experiment_by_name(experimento)
    if exp is None:
        raise RuntimeError(f"No existe el experimento {experimento}.")
    previos = client.search_runs([exp.experiment_id], max_results=100)
    for run in previos:
        nombre = run.info.run_name or run.data.tags.get("mlflow.runName")
        if nombre == "m2-bootstrap-test" and run.data.tags.get("estado") == "final":
            raise RuntimeError(
                f"m2-bootstrap-test ya está en final ({run.info.run_id}). No se repite.")
    tablas = {}
    for arch, run_id in RUNS_PRUEBA.items():
        ruta = client.download_artifacts(run_id, "predicciones_test.csv")
        tablas[arch] = pd.read_csv(ruta)
    resultado = bootstrap_prueba(tablas, n=n, semilla=semilla)
    texto = a_markdown(resultado)
    mlflow.set_experiment(experimento)
    tags = standard_tags(cfg, modulo="m2", fase="evaluacion-test", autor=autor, seed=semilla)
    with mlflow.start_run(run_name="m2-bootstrap-test", tags=tags):
        mlflow.log_params({
            "n_replicas": n,
            "semilla": semilla,
            "metodo": "estratificado_por_clase_agrupado_por_lesion",
            "run_id_resnet50": RUNS_PRUEBA["resnet50"],
            "run_id_densenet121": RUNS_PRUEBA["densenet121"],
            "run_id_efficientnetb0": RUNS_PRUEBA["efficientnetb0"],
        })
        mlflow.log_text(texto, "bootstrap_test.md")
        mlflow.set_tag("estado", "final")
        resultado["run_id"] = mlflow.active_run().info.run_id
    resultado["markdown"] = texto
    return resultado
