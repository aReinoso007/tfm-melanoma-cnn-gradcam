"""Selección del Módulo 2 (OE3), sobre los 9 runs ya entrenados.

La regla está en `docs/01_objetivos.md`. Todo se calcula con la media de las tres semillas
en validación. Este módulo no entrena: lee las métricas de MLflow, escribe `seleccion.json`
y `seleccion.md`, y registra la semilla representativa de cada red.

Desempate: si la diferencia de la métrica principal entre la líder y otra candidata es menor
que la mayor de sus desviaciones estándar entre semillas, esa diferencia no se distingue del
ruido y se usa la métrica de desempate (balanced accuracy). Si ninguna arquitectura cumple
los dos requisitos, la métrica principal pasa a ser la sensibilidad en melanoma.
"""
from __future__ import annotations

import json
import re
from statistics import mean, stdev

ARQUITECTURAS = ("resnet50", "densenet121", "efficientnetb0")
SEMILLAS = (42, 123, 2024)
# Nombres sin el prefijo `val_`: así se leen de MLflow (`val_macro_f1`, …).
METRICAS = ("macro_f1", "mel_sensitivity", "mel_precision",
            "min_class_sensitivity", "balanced_accuracy")
UMBRAL_MEL = 0.80
UMBRAL_MIN_CLASE = 0.50
_NOMBRE = re.compile(r"^m2-(resnet50|densenet121|efficientnetb0)-s(42|123|2024)$")


def resumir(filas: list[dict]) -> dict:
    """Agrupa los 9 runs por arquitectura.

    Cada fila trae `arch`, `seed`, `run_id` y las cinco métricas (sin prefijo).
    Devuelve, por arquitectura, media y desviación estándar (n−1) de cada métrica
    y la semilla representativa: la de macro-F1 más cercano a la media. Si hay dos
    a la misma distancia, se queda la semilla menor.
    """
    if len(filas) != 9:
        raise ValueError(f"La selección necesita exactamente 9 runs, hay {len(filas)}.")
    vistos = {(f["arch"], int(f["seed"])) for f in filas}
    esperado = {(a, s) for a in ARQUITECTURAS for s in SEMILLAS}
    if vistos != esperado:
        raise ValueError(f"Faltan o sobran combinaciones de arquitectura y semilla: {vistos ^ esperado}.")

    resumen = {}
    for arch in ARQUITECTURAS:
        suyos = [f for f in filas if f["arch"] == arch]
        bloque = {"semillas": sorted(suyos, key=lambda f: int(f["seed"]))}
        for metrica in METRICAS:
            valores = [float(f[metrica]) for f in suyos]
            bloque[metrica] = {"media": mean(valores), "std": stdev(valores)}
        media_f1 = bloque["macro_f1"]["media"]
        representativa = min(suyos, key=lambda f: (abs(float(f["macro_f1"]) - media_f1), int(f["seed"])))
        bloque["semilla_representativa"] = int(representativa["seed"])
        bloque["run_id_representativo"] = representativa["run_id"]
        resumen[arch] = bloque
    return resumen


def _dentro_del_ruido(resumen: dict, lider: str, otra: str, metrica: str) -> bool:
    """True si la diferencia de medias no supera la variación entre semillas."""
    diferencia = abs(resumen[lider][metrica]["media"] - resumen[otra][metrica]["media"])
    ruido = max(resumen[lider][metrica]["std"], resumen[otra][metrica]["std"])
    return diferencia < ruido


def _elegir_entre(resumen: dict, candidatas: list[str], principal: str, desempate: str) -> tuple[str, str]:
    """Mayor `principal`. Si otra queda dentro del ruido de la líder, manda `desempate`."""
    ordenadas = sorted(candidatas, key=lambda a: resumen[a][principal]["media"], reverse=True)
    lider = ordenadas[0]
    banda = [lider] + [a for a in ordenadas[1:] if _dentro_del_ruido(resumen, lider, a, principal)]
    if len(banda) == 1:
        return lider, f"mayor {principal}"
    ganadora = max(banda, key=lambda a: (resumen[a][desempate]["media"], resumen[a][principal]["media"]))
    return ganadora, f"{principal} dentro del ruido entre semillas; desempate por {desempate} ({', '.join(banda)})"


def decidir(resumen: dict) -> dict:
    """Aplica la regla de `docs/01_objetivos.md` y devuelve la ganadora y el porqué."""
    pasos = []
    cumplen = []
    for arch in ARQUITECTURAS:
        mel = resumen[arch]["mel_sensitivity"]["media"]
        minima = resumen[arch]["min_class_sensitivity"]["media"]
        ok_mel, ok_min = mel >= UMBRAL_MEL, minima >= UMBRAL_MIN_CLASE
        pasos.append({"arch": arch, "mel_sensitivity_media": mel, "min_class_sensitivity_media": minima,
                      "cumple_melanoma": ok_mel, "cumple_min_clase": ok_min})
        if ok_mel and ok_min:
            cumplen.append(arch)

    if cumplen:
        ganadora, como = _elegir_entre(resumen, cumplen, "macro_f1", "balanced_accuracy")
        motivo = "cumple_ambos_requisitos"
    else:
        ganadora, como = _elegir_entre(resumen, list(ARQUITECTURAS), "mel_sensitivity", "balanced_accuracy")
        motivo = "ninguna_cumple_ambos_requisitos"
    return {"ganadora": ganadora, "motivo": motivo, "como": como, "cumplen": cumplen, "pasos": pasos}


def clase_de_la_minima(fila: dict) -> str:
    """Clase o clases cuya sensibilidad es la mínima de esa semilla."""
    from src.metrics import CLASES
    valores = [(c, float(fila[f"sensitivity_{c}"])) for c in CLASES]
    peor = min(v for _, v in valores)
    return ", ".join(c for c, v in valores if v == peor)


def filas_desde_runs(runs, extras: tuple[str, ...] = ()) -> list[dict]:
    """Convierte runs de MLflow (`search_runs`) en las filas que lee `resumir`.

    Solo acepta el nombre `m2-<arch>-s<semilla>` de un run real ya marcado `estado=final`.
    """
    filas = []
    for run in runs:
        nombre = run.info.run_name or run.data.tags.get("mlflow.runName")
        tags = run.data.tags
        if not _NOMBRE.match(nombre or ""):
            continue
        if tags.get("modo") != "real" or tags.get("estado") != "final":
            continue
        arch, semilla = nombre.split("-s")
        arch = arch.removeprefix("m2-")
        fila = {"arch": arch, "seed": int(semilla), "run_id": run.info.run_id, "run_name": nombre}
        faltan = []
        for metrica in METRICAS:
            clave = f"val_{metrica}"
            if clave not in run.data.metrics:
                faltan.append(clave)
            else:
                fila[metrica] = float(run.data.metrics[clave])
        for metrica in extras:
            clave = "val_frac_nv_limpio" if metrica == "frac_nv" else f"val_{metrica}"
            if clave not in run.data.metrics:
                faltan.append(clave)
            else:
                fila[metrica] = float(run.data.metrics[clave])
        if faltan:
            raise KeyError(f"{nombre} no tiene las métricas {faltan}.")
        filas.append(fila)
    return filas


def _tabla_md(resumen: dict) -> str:
    cabecera = "| Arquitectura | " + " | ".join(METRICAS) + " |"
    sep = "|---|" + "---|" * len(METRICAS)
    lineas = [cabecera, sep]
    for arch in ARQUITECTURAS:
        celdas = [f"{resumen[arch][m]['media']:.3f} ± {resumen[arch][m]['std']:.3f}" for m in METRICAS]
        lineas.append(f"| {arch} | " + " | ".join(celdas) + " |")
    return "\n".join(lineas)


def a_markdown(resumen: dict, decision: dict) -> str:
    """Texto que se archiva como `seleccion.md` y se puede copiar a la memoria."""
    ganadora = decision["ganadora"]
    if decision["motivo"] == "ninguna_cumple_ambos_requisitos":
        cierre = (
            "Ninguna arquitectura alcanza a la vez sensibilidad en melanoma ≥ 0,80 y "
            "sensibilidad ≥ 0,50 en todas las clases. La regla elige entonces la de mayor "
            f"sensibilidad en melanoma: **{ganadora}** ({decision['como']}). "
            "Hay que explicar en la discusión por qué el requisito de melanoma no se cumple."
        )
    else:
        cierre = (
            f"Cumplen ambos requisitos: {', '.join(decision['cumplen']) or '—'}. "
            f"Se elige **{ganadora}** ({decision['como']})."
        )
    semillas = "\n".join(
        f"- `{arch}`: semilla {resumen[arch]['semilla_representativa']} "
        f"(run `{resumen[arch]['run_id_representativo']}`), alias `final`"
        + (" y `seleccionado`" if arch == ganadora else "")
        for arch in ARQUITECTURAS
    )
    return (
        "# Selección en validación (Módulo 2)\n\n"
        "Media ± desviación estándar de las tres semillas. El modelo guardado de cada run "
        "es el mejor de la fase 2.\n\n"
        f"{_tabla_md(resumen)}\n\n"
        f"## Decisión\n\n{cierre}\n\n"
        "## Semilla representativa\n\n"
        "De cada arquitectura se registra la semilla cuyo macro-F1 de validación está más cerca "
        "de la media de sus tres semillas.\n\n"
        f"{semillas}\n"
    )


def a_json(resumen: dict, decision: dict) -> dict:
    """Versión estructurada. Las medias van en número, no en texto."""
    tabla = {}
    for arch in ARQUITECTURAS:
        tabla[arch] = {m: resumen[arch][m] for m in METRICAS}
        tabla[arch]["semilla_representativa"] = resumen[arch]["semilla_representativa"]
        tabla[arch]["run_id_representativo"] = resumen[arch]["run_id_representativo"]
        tabla[arch]["semillas"] = [
            {"seed": int(f["seed"]), "run_id": f["run_id"], **{m: float(f[m]) for m in METRICAS}}
            for f in resumen[arch]["semillas"]
        ]
    return {"tabla": tabla, "decision": decision}


def lanzar_seleccion(cfg: dict, autor: str = "alex", registrar_alias: bool = True) -> dict:
    """Lee los 9 runs `final`, publica `seleccion.json` y `seleccion.md` y registra alias.

    El run de la decisión se llama `m2-seleccion`, en el experimento `m2-entrenamiento`.
    Si ya existe uno con `estado=final`, no crea otro: la selección se hace una vez.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.tracking import setup_mlflow, standard_tags

    setup_mlflow(cfg)
    client = MlflowClient()
    experimento = cfg["mlflow"]["experiments"]["m2_train"]
    exp = client.get_experiment_by_name(experimento)
    if exp is None:
        raise RuntimeError(f"No existe el experimento {experimento}.")
    previos = [
        r for r in client.search_runs([exp.experiment_id], max_results=100)
        if (r.info.run_name or r.data.tags.get("mlflow.runName")) == "m2-seleccion"
        and r.data.tags.get("estado") == "final"
    ]
    if previos:
        raise RuntimeError(
            "Ya hay un run m2-seleccion con estado=final "
            f"({previos[0].info.run_id}). La selección no se repite."
        )

    runs = client.search_runs([exp.experiment_id], max_results=100)
    filas = filas_desde_runs(runs)
    resumen = resumir(filas)
    decision = decidir(resumen)
    payload = a_json(resumen, decision)
    texto = a_markdown(resumen, decision)

    mlflow.set_experiment(experimento)
    tags = standard_tags(cfg, modulo="m2", fase="seleccion", autor=autor)
    with mlflow.start_run(run_name="m2-seleccion", tags=tags):
        mlflow.log_dict(payload, "seleccion.json")
        mlflow.log_text(texto, "seleccion.md")
        mlflow.set_tag("ganadora", decision["ganadora"])
        mlflow.set_tag("motivo", decision["motivo"])
        run_id = mlflow.active_run().info.run_id
        if registrar_alias:
            try:
                _registrar_alias(client, resumen, decision["ganadora"])
                mlflow.set_tag("alias", "ok")
            except Exception as exc:
                # La decisión ya está en los artefactos. Un fallo del registro de modelos
                # (DagsHub no siempre acepta alias) no debe dejar el run a medias.
                mlflow.set_tag("alias", "error")
                mlflow.log_text(f"{type(exc).__name__}: {exc}", "alias_error.txt")
                print("AVISO: la decisión quedó guardada, pero el alias no se pudo registrar.")
                print(type(exc).__name__, exc)
        mlflow.set_tag("estado", "final")

    payload["run_id_seleccion"] = run_id
    payload["markdown"] = texto
    return payload


EXTRAS_V2 = ("auc_mel", "auc_macro_ovr", "frac_nv",
             "sensitivity_akiec", "sensitivity_bcc", "sensitivity_bkl", "sensitivity_df",
             "sensitivity_mel", "sensitivity_nv", "sensitivity_vasc")


def _promediar(resumen: dict, claves: tuple[str, ...]) -> None:
    for arch in ARQUITECTURAS:
        for clave in claves:
            valores = [float(f[clave]) for f in resumen[arch]["semillas"]]
            resumen[arch][clave] = {"media": mean(valores), "std": stdev(valores)}


def a_markdown_v2(resumen: dict, decision: dict) -> str:
    """Informe de `m2-seleccion-v2`. La regla es la de v1; la tabla añade lo pedido para la memoria."""
    from src.metrics import CLASES
    columnas = ("macro_f1", "mel_sensitivity", "mel_precision", "min_class_sensitivity",
                "balanced_accuracy", "auc_mel", "auc_macro_ovr", "frac_nv")
    cabecera = "| Arquitectura | " + " | ".join(columnas) + " | Clase de la mínima |"
    sep = "|---|" + "---|" * len(columnas) + "---|"
    filas = [cabecera, sep]
    for arch in ARQUITECTURAS:
        celdas = [f"{resumen[arch][m]['media']:.3f} ± {resumen[arch][m]['std']:.3f}" for m in columnas]
        por_semilla = "; ".join(
            f"s{int(f['seed'])} {clase_de_la_minima(f)}" for f in resumen[arch]["semillas"])
        filas.append(f"| {arch} | " + " | ".join(celdas) + f" | {por_semilla} |")
    sens = ["| Arquitectura | " + " | ".join(CLASES) + " |",
            "|---|" + "---|" * len(CLASES)]
    for arch in ARQUITECTURAS:
        celdas = [f"{resumen[arch][f'sensitivity_{c}']['media']:.3f} ± {resumen[arch][f'sensitivity_{c}']['std']:.3f}"
                  for c in CLASES]
        sens.append(f"| {arch} | " + " | ".join(celdas) + " |")
    base = a_markdown(resumen, decision)
    return (
        base.replace("# Selección en validación (Módulo 2)",
                     "# Selección en validación v2 (aumento uniforme)")
        + "\n## Métricas ampliadas\n\n"
        + "Fracción nv es la predicha en validación limpia (`val_frac_nv_limpio`). "
        + "La clase de la mínima es la de cada semilla, no la media.\n\n"
        + "\n".join(filas) + "\n\n## Sensibilidad por clase\n\n" + "\n".join(sens) + "\n"
    )


def lanzar_seleccion_v2(cfg: dict, autor: str = "alex", registrar_alias: bool = True) -> dict:
    """Selección sobre `m2-entrenamiento-v2`. Misma regla. Mueve los alias desde la v1.

    El run se llama `m2-seleccion-v2` y no se repite si ya está en `estado=final`.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.metrics import CLASES
    from src.tracking import setup_mlflow, standard_tags

    setup_mlflow(cfg)
    client = MlflowClient()
    experimento = cfg["mlflow"]["experiments"]["m2_train_v2"]
    exp = client.get_experiment_by_name(experimento)
    if exp is None:
        raise RuntimeError(f"No existe el experimento {experimento}.")
    previos = [
        r for r in client.search_runs([exp.experiment_id], max_results=100)
        if (r.info.run_name or r.data.tags.get("mlflow.runName")) == "m2-seleccion-v2"
        and r.data.tags.get("estado") == "final"
    ]
    if previos:
        raise RuntimeError(
            "Ya hay un run m2-seleccion-v2 con estado=final "
            f"({previos[0].info.run_id}). La selección no se repite."
        )
    runs = client.search_runs([exp.experiment_id], max_results=100)
    extras = EXTRAS_V2
    filas = filas_desde_runs(runs, extras=extras)
    resumen = resumir(filas)
    _promediar(resumen, extras)
    decision = decidir(resumen)
    payload = a_json(resumen, decision)
    for arch in ARQUITECTURAS:
        for clave in extras:
            payload["tabla"][arch][clave] = resumen[arch][clave]
        for fila_json, fila in zip(payload["tabla"][arch]["semillas"], resumen[arch]["semillas"]):
            fila_json["clase_minima"] = clase_de_la_minima(fila)
            for clave in extras:
                fila_json[clave] = float(fila[clave])
    texto = a_markdown_v2(resumen, decision)
    mlflow.set_experiment(experimento)
    tags = standard_tags(cfg, modulo="m2", fase="seleccion", autor=autor, aumento="uniforme")
    with mlflow.start_run(run_name="m2-seleccion-v2", tags=tags):
        mlflow.log_dict(payload, "seleccion.json")
        mlflow.log_text(texto, "seleccion.md")
        mlflow.set_tag("ganadora", decision["ganadora"])
        mlflow.set_tag("motivo", decision["motivo"])
        run_id = mlflow.active_run().info.run_id
        if registrar_alias:
            try:
                _mover_alias_a_v2(client, resumen, decision["ganadora"])
                mlflow.set_tag("alias", "ok")
            except Exception as exc:
                mlflow.set_tag("alias", "error")
                mlflow.log_text(f"{type(exc).__name__}: {exc}", "alias_error.txt")
                print("AVISO: la decisión quedó guardada, pero el alias no se pudo mover.")
                print(type(exc).__name__, exc)
        mlflow.set_tag("estado", "final")
    payload["run_id_seleccion"] = run_id
    payload["markdown"] = texto
    payload["clases"] = list(CLASES)
    return payload


def _mover_alias_a_v2(client, resumen: dict, ganadora: str) -> None:
    """Quita `final` y `seleccionado` de la v1 y los pone en la semilla representativa v2."""
    import mlflow

    for arch in ARQUITECTURAS:
        nombre = f"ham10000-{arch}"
        for alias in ("final", "seleccionado"):
            try:
                client.delete_registered_model_alias(nombre, alias)
            except Exception:
                pass
        version = mlflow.register_model(
            f"runs:/{resumen[arch]['run_id_representativo']}/modelo", nombre
        ).version
        client.set_registered_model_alias(nombre, "final", version)
        if arch == ganadora:
            client.set_registered_model_alias(nombre, "seleccionado", version)


def _registrar_alias(client, resumen: dict, ganadora: str) -> None:
    """Registra `runs:/<id>/modelo` y pone el alias. La ganadora lleva dos alias."""
    import mlflow

    for arch in ARQUITECTURAS:
        nombre = f"ham10000-{arch}"
        version = mlflow.register_model(
            f"runs:/{resumen[arch]['run_id_representativo']}/modelo", nombre
        ).version
        client.set_registered_model_alias(nombre, "final", version)
        if arch == ganadora:
            client.set_registered_model_alias(nombre, "seleccionado", version)
