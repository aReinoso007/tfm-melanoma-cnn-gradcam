# Módulo 2 — Fine-tuning, selección y evaluación (OE2, OE3)

**Responsable:** Alex Reinoso · **Experimentos MLflow:** `m2-entrenamiento`, `m2-evaluacion-test` · **Depende de:** M1 · **Bloquea a:** M3

## Objetivo

Entrenar las tres arquitecturas bajo condiciones idénticas, seleccionar el mejor modelo en validación con la regla acordada y evaluarlo una sola vez en prueba.

## Mientras M1 termina la partición (no hay que esperar)
- [ ] Escribir `src/models.py` con `build_model(arch, num_classes=7)` usando como base ResNet50, DenseNet121 y EfficientNetB0 (`weights="imagenet"`, `include_top=False`) + `GlobalAveragePooling2D` + `Dropout` + `Dense(7, softmax)`.
- [ ] Escribir `src/metrics.py` con todas las métricas de `03_mlflow_convenciones.md` §5, a partir de `y_true` y `y_prob`.
- [ ] Probar el pipeline completo con un **subconjunto pequeño** en el experimento `sandbox` (2 épocas) para detectar errores antes de gastar GPU.

## Pasos

### 1. Configuración común
- [ ] Hiperparámetros base en `configs/base.yaml`, **iguales para las tres arquitecturas**. Cualquier cambio pasa por un PR aprobado por los tres.
- [ ] Semillas: `[42, 123, 2024]`. Al inicio de cada run, `set_seeds(seed)`.

### 2. Fase 1 — base congelada (Fase 2 de la metodología)
- [ ] `base.trainable = False`; solo se entrena la cabeza.
- [ ] Pérdida `categorical_crossentropy` con `class_weight` de `class_weights.json`.
- [ ] Optimizador Adam con `f1_lr`, `EarlyStopping` sobre `val_loss` con `restore_best_weights=True`.
- [ ] Registrar por época `f1_loss`, `f1_val_loss` y `f1_val_macro_f1` (callback propio).

### 3. Fase 2 — descongelado progresivo (Fase 3 de la metodología)
- [ ] Descongelar la fracción superior de la base (`f2_unfreeze_fraction`).
- [ ] **Mantener las capas BatchNormalization congeladas** (llamar a la base con `training=False`). Si no, el fine-tuning con lotes pequeños degrada los pesos de ImageNet.
- [ ] Recompilar con `f2_lr` (≈ 10–100 veces menor que `f1_lr`) y continuar desde la mejor época de la fase 1.
- [ ] Registrar `f2_*` con `step` = época acumulada.
- [ ] Checkpoints en Drive para poder retomar si Colab se desconecta.

### 4. Evaluación en validación (por run)
- [ ] Predicciones en validación → `predicciones_val.csv` (probabilidades de las 7 clases).
- [ ] Todas las métricas `val_*`, la matriz de confusión normalizada por filas y el `classification_report`.
- [ ] Registrar el modelo con `mlflow.tensorflow.log_model`.
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
