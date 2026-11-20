"""Cierre de la v3, cuando los nueve entrenamientos ya estén en estado=final.

No entrena y no abre la prueba salvo que se pida explícitamente.

    python scripts/m2_v3_cierre.py --seleccion
    python scripts/m2_v3_cierre.py --prueba

La selección pone el alias `v3-representativo` y no toca `final` ni `seleccionado`.
La prueba es una vez por red, en `m2-evaluacion-test-v3`, con el bootstrap pareado
contra la v2 de esa misma arquitectura.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Cierre de la v3. No forma parte del entrenamiento.")
    parser.add_argument("--seleccion", action="store_true")
    parser.add_argument("--prueba", action="store_true")
    args = parser.parse_args()
    if not args.seleccion and not args.prueba:
        print("Sin --seleccion ni --prueba no se hace nada. Los nueve runs tienen que existir antes.")
        return
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.tracking import load_config, setup_mlflow
    cfg = load_config("configs/m2_v3.yaml")
    setup_mlflow(cfg)
    if args.seleccion:
        from src.selection import lanzar_seleccion_v3
        resultado = lanzar_seleccion_v3(cfg)
        print(resultado["run_id"])
        print(resultado["comparacion"])
    if args.prueba:
        from src.evaluation import lanzar_evaluacion_test_v3
        for arch in ("resnet50", "densenet121", "efficientnetb0"):
            print(lanzar_evaluacion_test_v3(arch, cfg)["run_id"])


if __name__ == "__main__":
    main()
