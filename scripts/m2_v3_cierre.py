"""Cierre de la v3, cuando los nueve entrenamientos ya estén en estado=final.

No entrena. La prueba solo se abre con `--prueba`.

    python scripts/m2_v3_cierre.py --seleccion
    python scripts/m2_v3_cierre.py --umbral
    python scripts/m2_v3_cierre.py --prueba

`--umbral` anota el corte de melanoma en validación y no abre la prueba.
`--prueba` anota ese corte si falta, evalúa una vez cada red en
`m2-evaluacion-test-v3` y publica el bootstrap entre las tres.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Cierre de la v3. No forma parte del entrenamiento.")
    parser.add_argument("--seleccion", action="store_true")
    parser.add_argument("--umbral", action="store_true")
    parser.add_argument("--prueba", action="store_true")
    args = parser.parse_args()
    if not args.seleccion and not args.umbral and not args.prueba:
        print("Sin --seleccion, --umbral ni --prueba no se hace nada.")
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
    if args.umbral or args.prueba:
        from src.evaluation import anotar_umbrales_v3
        for corte in anotar_umbrales_v3(cfg):
            print(
                f"{corte['arch']} t={corte['t']:.12g} sens={corte['mel_sensitivity']:.3f} "
                f"prec={corte['mel_precision']:.3f} f1={corte['macro_f1']:.3f} "
                f"min={corte['min_class_sensitivity']:.3f} {','.join(corte['clase_minima'])} "
                f"frac_mel={corte['frac_pred_mel']:.3f}"
            )
    if args.prueba:
        from src.evaluation import lanzar_evaluacion_test_v3, publicar_bootstrap_redes_v3
        for arch in ("resnet50", "densenet121", "efficientnetb0"):
            try:
                print(lanzar_evaluacion_test_v3(arch, cfg)["run_id"])
            except RuntimeError as exc:
                if "no se repite" not in str(exc):
                    raise
                print(exc)
        print(publicar_bootstrap_redes_v3(cfg))


if __name__ == "__main__":
    main()
