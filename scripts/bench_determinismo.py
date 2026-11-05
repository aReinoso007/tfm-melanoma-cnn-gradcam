"""Comprueba, para UNA arquitectura y UN modo, si el entrenamiento se repite y cuánto cuesta.

    python scripts/bench_determinismo.py --arch resnet50 --det 1
    python scripts/bench_determinismo.py --arch resnet50 --det 0

Qué hace: entrena N lotes reales de validación en la fase 1 y N en la fase 2 (con los helpers de
src/models.py y los hiperparámetros de configs/base.yaml), DOS veces seguidas con la misma
semilla, y compara la pérdida de cada paso y un hash de todos los pesos finales.
Imprime una línea `RESULT {json}` con:
    error            None, o el mensaje si `enable_op_determinism()` rechaza alguna operación
    repetible        True si las dos pasadas dan exactamente las mismas pérdidas y los mismos pesos
    ms_por_paso      tiempo medio por paso (sin el primero de cada fase, que incluye la compilación)

Por qué un script aparte y no una celda del notebook:
  - `tf.config.experimental.enable_op_determinism()` no se puede desactivar dentro de un proceso.
    Para comparar con y sin determinismo hace falta un proceso nuevo por modo.
  - Los lotes se cargan en memoria ANTES de medir, así el tiempo es el del paso de entrenamiento
    en GPU y no el de leer JPEG del disco.
No guarda nada y no usa MLflow: es solo una comprobación.
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())          # se ejecuta desde la raíz del repo

ap = argparse.ArgumentParser()
ap.add_argument("--arch", required=True)
ap.add_argument("--det", type=int, required=True, help="1 = deterministic=True, 0 = False")
ap.add_argument("--n", type=int, default=10, help="lotes por fase")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--images-dir", default=None, help="solo para pruebas fuera de Colab")
args = ap.parse_args()

res = {"arch": args.arch, "det": args.det, "error": None, "repetible": None, "ms_por_paso": None}
try:
    import numpy as np
    import tensorflow as tf
    import yaml
    from src.data import make_dataset
    from src.models import build_model, congelar_base, descongelar_fraccion
    from src.tracking import set_seeds

    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))["training"]
    kw = {"images_dir": args.images_dir} if args.images_dir else {}
    lotes = list(make_dataset("val", args.arch, training=False, cache=False, **kw)
                 .take(args.n).as_numpy_iterator())
    assert len(lotes) == args.n, f"solo hay {len(lotes)} lotes"

    def una_pasada():
        set_seeds(args.seed, deterministic=bool(args.det))
        model = build_model(args.arch, dropout=cfg["dropout"])
        perdidas, tiempos = [], []
        for fase in ("f1", "f2"):
            if fase == "f1":
                congelar_base(model)
                lr = cfg["phase1"]["lr"]
            else:
                descongelar_fraccion(model, cfg["phase2"]["unfreeze_fraction"],
                                     cfg["phase2"]["freeze_batchnorm"])
                lr = cfg["phase2"]["lr"]
            model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="categorical_crossentropy")
            for i, (x, y) in enumerate(lotes):
                t0 = time.perf_counter()
                perdidas.append(float(model.train_on_batch(x, y)))
                if i > 0:                      # el primer paso de cada fase incluye la compilación
                    tiempos.append(time.perf_counter() - t0)
        h = hashlib.sha256()
        for w in model.get_weights():
            h.update(np.ascontiguousarray(w).tobytes())
        return perdidas, h.hexdigest(), tiempos

    p1, h1, t1 = una_pasada()
    p2, h2, t2 = una_pasada()
    res["repetible"] = bool(p1 == p2 and h1 == h2)
    res["max_dif_perdida"] = float(np.max(np.abs(np.array(p1) - np.array(p2))))
    res["ms_por_paso"] = round(1000 * float(np.mean(t2)), 1)       # la segunda pasada ya está "caliente"
except Exception as e:                       # noqa: BLE001
    res["error"] = f"{type(e).__name__}: {str(e)[:400]}"
print("RESULT " + json.dumps(res))
