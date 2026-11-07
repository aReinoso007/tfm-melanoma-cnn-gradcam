# Módulo 2 — Fine-tuning, selección y evaluación (OE2, OE3)

**Responsable:** Alex Reinoso · **Experimentos MLflow:** `m2-entrenamiento`, `m2-evaluacion-test` · **Depende de:** M1 · **Bloquea a:** M3

## Objetivo

Entrenar las tres arquitecturas bajo condiciones idénticas, seleccionar el mejor modelo en validación con la regla acordada y evaluarlo una sola vez en prueba.

## Mientras M1 termina la partición (no hay que esperar)
- [x] Escribir `src/models.py` con `build_model(arch, num_classes=7)` usando como base ResNet50, DenseNet121 y EfficientNetB0 (`weights="imagenet"`, `include_top=False`) + `GlobalAveragePooling2D` + `Dropout` + `Dense(7, softmax)`. Incluye `congelar_base()` (fase 1) y `descongelar_fraccion()` (fase 2). Rama `m2/modelos-metricas`; razones en [modulo2_registro_decisiones.md](modulo2_registro_decisiones.md) D3–D6.
- [x] Escribir `src/metrics.py` con todas las métricas de `03_mlflow_convenciones.md` §5, a partir de `y_true` y `y_prob`. Incluye `tabla_predicciones()` para `predicciones_<val|test>.csv` (D7–D8).
- [x] Pruebas en `tests/` (`python -m pytest tests -q`, 22 pruebas, sin GPU) (D9).
- [x] Probar el pipeline completo en el experimento `sandbox` (2+2 épocas, datos completos, T4, 7 nov 2026): las tres redes terminan, DagsHub acepta el modelo y la proyección de D10 queda en 83 min por cuenta. Notebook `m2_01_entrenamiento.ipynb`, decisiones D12–D16.

## Pasos

### 1. Configuración común
- [ ] Hiperparámetros base en `configs/base.yaml`, **iguales para las tres arquitecturas**. Cualquier cambio pasa por un PR aprobado por los tres.
- [x] Semillas: `[42, 123, 2024]`. Al inicio de cada run, `set_seeds(seed)` (con `training.deterministic: true`, decisión D10: medido en las tres redes, se repite exacto y cuesta 1,4–1,9× por paso). Registrar `deterministic`, `gpu` y `tf_version` en cada run. Implementado en `lanzar_run` (D16).

### 2. Fase 1 — base congelada (Fase 2 de la metodología)
- [x] `base.trainable = False`; solo se entrena la cabeza.
- [x] Pérdida `categorical_crossentropy` con `class_weight` de `class_weights.json`.
- [x] Optimizador Adam con `f1_lr`, `EarlyStopping` sobre `val_loss` con `restore_best_weights=True`. Parada y mejores pesos con `ControlFase` (persistente), no `EarlyStopping` de Keras: D13.
- [x] Registrar por época `f1_loss`, `f1_val_loss` y `f1_val_macro_f1` (callback propio). (`EvaluarValidacion`, D14).

### 3. Fase 2 — descongelado progresivo (Fase 3 de la metodología)
- [x] Descongelar la fracción superior de la base (`f2_unfreeze_fraction`).
- [x] **Mantener las capas BatchNormalization congeladas** (llamar a la base con `training=False`). Si no, el fine-tuning con lotes pequeños degrada los pesos de ImageNet.
- [x] Recompilar con `f2_lr` (≈ 10–100 veces menor que `f1_lr`) y continuar desde la mejor época de la fase 1.
- [x] Registrar `f2_*` con `step` = época acumulada. (D12).
- [x] Checkpoints en Drive para poder retomar si Colab se desconecta. (D15; probado con una desconexión simulada).

### 4. Evaluación en validación (por run)
- [x] Predicciones en validación → `predicciones_val.csv` (probabilidades de las 7 clases).
- [x] Todas las métricas `val_*`, la matriz de confusión normalizada por filas y el `classification_report`.
- [x] Registrar el modelo de forma que `mlflow.tensorflow.load_model("runs:/<id>/modelo")` funcione en DagsHub. `log_model` de MLflow 3 no sirve ahí (D17); se usa `publicar_modelo()`. Falta repetir la celda de carga en Colab sobre los tres runs de `sandbox`.
- [ ] Repetir para 3 arquitecturas × 3 semillas = **9 runs**.

### 5. Selección (en validación, con la media de las 3 semillas)
- [ ] Tabla resumen: media ± desviación estándar por arquitectura de `val_macro_f1`, `val_mel_sensitivity`, `val_mel_precision`, `val_min_class_sensitivity` y `val_balanced_accuracy`.
- [ ] Aplicar la regla:
  1. ¿`val_mel_sensitivity` ≥ 0.80?
  2. ¿`val_min_class_sensitivity` ≥ 0.50?
  3. Entre las que cumplen ambas → mayor `val_macro_f1`.
  4. Si la diferencia de F1 macro < desviación estándar entre semillas → mayor `val_balanced_accuracy`.
  5. Si ninguna cumple → mayor `val_mel_sensitivity` y se analizan las causas en la discusión.
- [ ] Guardar la tabla y la decisión como artefacto (`seleccion.json` + `seleccion.md`).
- [ ] Registrar en el Model Registry la semilla representativa de cada arquitectura con alias `final`, y la ganadora además con alias `seleccionado`.

### 6. Evaluación final en prueba (UNA sola vez)
- [ ] **No ejecutar hasta haber cerrado la selección.** Si después se cambia algo del entrenamiento, se documenta como desviación.
- [ ] Un run por arquitectura en `m2-evaluacion-test` (cargando `models:/ham10000-<arch>@final`).
- [ ] Métricas `test_*`, matriz de confusión normalizada y `predicciones_test.csv`. **Este CSV es la entrada del M3.**
- [ ] Comprobar si el modelo seleccionado mantiene sensibilidad en melanoma ≥ 80% en prueba.
- [ ] Opcional: intervalos de confianza al 95% por bootstrap **remuestreando por `lesion_id`** (1,000 réplicas) para F1 macro y sensibilidad en melanoma.

## Entregable para M3
- `models:/ham10000-<arch>@final` para las 3 arquitecturas.
- `predicciones_test.csv` de cada arquitectura (artefacto de `m2-test-<arch>`).

## Para la memoria (capítulo 4, sección M2)
Configuración de entrenamiento (tabla de hiperparámetros), curvas de aprendizaje de las dos fases, tabla de validación (media ± desviación estándar), aplicación de la regla de selección paso a paso, tabla final de prueba, matrices de confusión y análisis de hacia qué clases se desvían los melanomas.
