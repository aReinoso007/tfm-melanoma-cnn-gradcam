"""Utilidades comunes de trazabilidad y reproducibilidad (MLflow en DagsHub).

Uso típico al inicio de cada notebook:

    from src.tracking import load_config, setup_mlflow, set_seeds, standard_tags, log_config
    cfg = load_config("configs/base.yaml")
    setup_mlflow(cfg)
    mlflow.set_experiment(cfg["mlflow"]["experiments"]["m2_train"])
    set_seeds(42)
    with mlflow.start_run(run_name="m2-resnet50-s42",
                          tags=standard_tags(cfg, modulo="m2", fase="entrenamiento",
                                             arch="resnet50", seed=42, autor="alex")):
        log_config(cfg)
        ...
"""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import yaml


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

def load_config(path: str | Path = "configs/base.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten(d: dict, parent: str = "", sep: str = ".") -> dict:
    items: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else str(k)
        if isinstance(v, dict):
            items.update(_flatten(v, key, sep))
        else:
            items[key] = v
    return items


def log_config(cfg: dict, exclude_prefixes: tuple[str, ...] = ("paths.", "mlflow.")) -> None:
    """Registra toda la configuración (aplanada) como parámetros del run activo."""
    flat = {k: str(v) for k, v in _flatten(cfg).items()
            if not k.startswith(exclude_prefixes)}
    # MLflow limita el número de parámetros por llamada: se envían por lotes
    keys = list(flat)
    for i in range(0, len(keys), 90):
        mlflow.log_params({k: flat[k] for k in keys[i:i + 90]})


# ---------------------------------------------------------------------------
# MLflow / DagsHub
# ---------------------------------------------------------------------------

def setup_mlflow(cfg: dict) -> str:
    """Apunta MLflow al servidor de DagsHub del repo.

    Las credenciales se leen de las variables de entorno
    MLFLOW_TRACKING_USERNAME y MLFLOW_TRACKING_PASSWORD (token de DagsHub),
    cargadas desde los secretos de Colab. Nunca escribirlas en el código.
    """
    owner = cfg["mlflow"]["repo_owner"]
    repo = cfg["mlflow"]["repo_name"]
    if not os.environ.get("MLFLOW_TRACKING_USERNAME") or not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        raise RuntimeError(
            "Faltan MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD. "
            "Cárgalos desde los secretos de Colab (DAGSHUB_USER / DAGSHUB_TOKEN)."
        )
    uri = f"https://dagshub.com/{owner}/{repo}.mlflow"
    mlflow.set_tracking_uri(uri)
    return uri


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def git_commit() -> str:
    return _git("rev-parse", "HEAD")


def git_dirty() -> bool:
    """True si hay cambios sin commit en archivos versionados."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return status not in ("", "unknown")


# ---------------------------------------------------------------------------
# Hashes
# ---------------------------------------------------------------------------

def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Semillas
# ---------------------------------------------------------------------------

def set_seeds(seed: int, deterministic: bool = True) -> None:
    """Fija las semillas de Python, NumPy y TensorFlow.

    Con deterministic=True se activa el determinismo de operaciones de TF.
    Algunas operaciones de GPU no tienen versión determinista y lanzarán un
    error; en ese caso usar deterministic=False y registrarlo en el run.
    """
    import numpy as np
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


# ---------------------------------------------------------------------------
# Runs que sobreviven a desconexiones de Colab
# ---------------------------------------------------------------------------

def start_or_resume_run(run_name: str, state_dir: str | Path,
                        tags: dict[str, str] | None = None) -> mlflow.ActiveRun:
    """Inicia un run nuevo o retoma el anterior si Colab se desconectó.

    Guarda el run_id en <state_dir>/mlflow_run_id.txt (en Drive). Si el archivo
    existe y el run no está terminado, lo retoma en lugar de crear otro.
    Usar junto con tf.keras.callbacks.BackupAndRestore(<state_dir>/backup).
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    id_file = state_dir / "mlflow_run_id.txt"

    if id_file.exists():
        run_id = id_file.read_text().strip()
        try:
            status = mlflow.get_run(run_id).info.status
        except Exception:
            status = None
        if status in ("RUNNING", "FAILED", "KILLED"):
            print(f"Retomando run {run_name} ({run_id})")
            return mlflow.start_run(run_id=run_id)
        if status == "FINISHED":
            raise RuntimeError(
                f"El run {run_name} ({run_id}) ya terminó. Si quieres repetirlo, "
                f"borra {id_file} y su carpeta de backup a propósito."
            )

    run = mlflow.start_run(run_name=run_name, tags=tags)
    id_file.write_text(run.info.run_id)
    return run


# ---------------------------------------------------------------------------
# Tags estándar
# ---------------------------------------------------------------------------

def standard_tags(cfg: dict, modulo: str, fase: str, autor: str,
                  arch: str | None = None, seed: int | None = None,
                  **extra: Any) -> dict[str, str]:
    """Tags obligatorios de cada run (ver docs/03_mlflow_convenciones.md §3)."""
    split_csv = Path(cfg["paths"]["split_csv"])
    tags = {
        "modulo": modulo,
        "fase": fase,
        "autor": autor,
        "split_version": cfg["data"]["split_version"],
        "split_sha256": file_sha256(split_csv) if split_csv.exists() else "pendiente",
        "git_commit": git_commit(),
        "git_dirty": str(git_dirty()).lower(),
        "estado": "en_curso",
    }
    if arch is not None:
        tags["arch"] = arch
    if seed is not None:
        tags["seed"] = str(seed)
    tags.update({k: str(v) for k, v in extra.items()})
    return tags
