# Convenciones de MLflow (DagsHub)

El objetivo es que **cualquier número o figura de la memoria pueda rastrearse hasta un run concreto**: con qué código, con qué datos y con qué configuración se obtuvo.

## 1. Experimentos

| Experimento | Módulo | Contenido |
|---|---|---|
| `m1-datos` | M1 | EDA, partición, pesos de clase |
| `m2-entrenamiento` | M2 | 9 runs de entrenamiento (3 arquitecturas × 3 semillas), evaluados en validación |
| `m2-evaluacion-test` | M2 | 3 runs, uno por arquitectura, **ejecutados una sola vez** |
| `m3-gradcam` | M3 | Mapas Grad-CAM, análisis de FN y anotación ABCDE |
| `sandbox` | Todos | Pruebas y depuración. **Nada de aquí se cita en la memoria.** |

## 2. Nombres de runs

```
m1-split-v1
m2-<arch>-s<seed>            p. ej. m2-resnet50-s42
m2-test-<arch>               p. ej. m2-test-densenet121
m3-gradcam-<arch>
```

Las fases 1 y 2 del fine-tuning van en **el mismo run** (`m2-<arch>-s<seed>`). La fase se distingue por el prefijo de la métrica (`f1_` / `f2_`) y el paso es la época acumulada.

## 3. Tags obligatorios (los pone `standard_tags()`)

| Tag | Ejemplo | Para qué |
|---|---|---|
| `modulo` | `m2` | Filtrar por módulo |
| `fase` | `entrenamiento` | Fase de la metodología |
| `arch` | `resnet50` | Arquitectura |
| `seed` | `42` | Semilla |
| `split_version` | `v1` | Versión de la partición |
| `split_sha256` | `3fa2…` | Prueba de que los tres módulos usan los mismos datos |
| `git_commit` | `a1b2c3d` | Código exacto |
| `git_dirty` | `false` | Si es `true`, el run no vale como resultado final |
| `autor` | `alex` | Quién lo lanzó |
| `estado` | `final` / `descartado` | Marcar a mano los runs que se usan en la memoria |

## 4. Parámetros

Se registra **toda la configuración** de `configs/base.yaml` (aplanada) más lo específico del run: `img_size`, `batch_size`, `optimizer`, `f1_lr`, `f1_epochs`, `f2_lr`, `f2_epochs`, `f2_unfreeze_fraction`, `class_weighting`, `augmentation`, `early_stopping_patience` y `monitor`.

## 5. Métricas

**Durante el entrenamiento** (por época, `step=epoch`):
`f1_loss`, `f1_val_loss`, `f1_val_macro_f1`, `f2_loss`, `f2_val_loss`, `f2_val_macro_f1`

**Al final de cada run en validación:**
```
val_macro_f1, val_balanced_accuracy, val_auc_macro_ovr
val_mel_sensitivity, val_mel_precision, val_mel_f1, val_mel_ap
val_min_class_sensitivity
val_sensitivity_<clase>, val_precision_<clase>, val_f1_<clase>,
val_specificity_<clase>, val_auc_<clase>, val_ap_<clase>
```
`<clase>` ∈ `akiec, bcc, bkl, df, mel, nv, vasc`

**En prueba** (solo en `m2-evaluacion-test`): las mismas métricas con prefijo `test_`.

## 6. Artefactos

| Run | Artefactos |
|---|---|
| `m1-split-v1` | `split_v1.csv`, `class_weights.json`, figuras de EDA, `split_report.json` (conteos por clase y subconjunto, comprobación de no fuga) |
| `m2-<arch>-s<seed>` | `curvas.png`, `confusion_val_norm.png`, `predicciones_val.csv`, `classification_report_val.json`, modelo (`mlflow.tensorflow.log_model`) |
| `m2-test-<arch>` | `confusion_test_norm.png`, `predicciones_test.csv`, `classification_report_test.json`, `bootstrap_ci.json` (si se hace) |
| `m3-gradcam-<arch>` | Mapas por imagen (`gradcam/<clase>/<image_id>.png`), `fn_melanoma.csv`, `anotacion_abcde.csv`, `resumen_atajos.json` |

## 7. Registro de modelos

- Un modelo registrado por arquitectura: `ham10000-resnet50`, `ham10000-densenet121`, `ham10000-efficientnetb0`.
- Se registra la versión de la **semilla representativa**: la que tenga el F1 macro de validación más cercano a la media de las tres semillas.
- **Alias** (MLflow 3 usa alias en lugar de *stages*):
  - `final`: el modelo de cada arquitectura que se evalúa en prueba y analiza el Módulo 3.
  - `seleccionado`: el ganador según la regla de selección (solo uno de los tres).
- El Módulo 3 **siempre carga por alias**, nunca desde Drive:
  ```python
  model = mlflow.tensorflow.load_model("models:/ham10000-resnet50@final")
  ```

## 8. Checklist de reproducibilidad (OE7)

Un resultado puede ir en la memoria solo si:

- [ ] El run está en `m1-datos`, `m2-*` o `m3-gradcam`, no en `sandbox`.
- [ ] `git_dirty=false` y el `git_commit` existe en `main`.
- [ ] `split_sha256` coincide con el de `m1-split-v1`.
- [ ] Todos los parámetros de `base.yaml` están registrados.
- [ ] Las figuras o tablas que se citan son artefactos de ese run.
- [ ] El run tiene `estado=final`.
- [ ] La prueba se ejecutó una sola vez por arquitectura: hay exactamente un run por arquitectura en `m2-evaluacion-test`.
