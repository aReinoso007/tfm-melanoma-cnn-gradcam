# Módulo 3 — Explicabilidad con Grad-CAM y análisis clínico (OE4, OE5, OE6)

**Responsable:** Christian Zhirzhan · **Experimento MLflow:** `m3-gradcam` · **Depende de:** M2

## Objetivo

Valorar si las decisiones de los modelos se apoyan en la lesión y en criterios dermatológicos, o en elementos ajenos a ella, con especial atención a los falsos negativos de melanoma.

## Mientras M2 entrena (no hay que esperar)
- [ ] Escribir `src/gradcam.py` con `gradcam(model, image, class_idx, layer_name)` y `overlay(image, heatmap)`.
- [ ] Identificar la **última capa convolucional** de cada arquitectura y fijarla en `base.yaml`:
  - ResNet50 → `conv5_block3_out`
  - DenseNet121 → `relu`
  - EfficientNetB0 → `top_activation`
  (Verificar con `model.summary()`: si la base está anidada dentro del modelo, hay que acceder a la capa a través de la base.)
- [ ] Probarlo con un modelo de `sandbox` y comprobar que el mapa cambia al cambiar `class_idx`.
- [ ] Diseñar la **plantilla de anotación** (`anotacion_abcde.csv`), ver el paso 4.

## Pasos

### 1. Carga de modelos y predicciones (siempre desde MLflow)
- [ ] `mlflow.tensorflow.load_model("models:/ham10000-<arch>@final")` para las 3 arquitecturas.
- [ ] Descargar `predicciones_test.csv` del run `m2-test-<arch>`.
- [ ] Registrar como tags la versión del modelo cargado y el `split_sha256`. Así queda probado que se analiza el mismo modelo que se evaluó.

### 2. Falsos negativos de melanoma (OE4)
- [ ] Filtrar `y_true == mel` y `y_pred != mel` → `fn_melanoma.csv` (`image_id, lesion_id, y_pred, p_mel, p_pred`).
- [ ] Contar los FN por modelo y por clase de destino (¿se confunden sobre todo con `nv`? ¿con `bkl`?).
- [ ] Ver qué FN son comunes a los tres modelos y cuáles exclusivos de uno.

### 3. Mapas Grad-CAM (OE5)
- [ ] Para las 7 clases: una muestra de aciertos y errores por clase (p. ej. 10 + 10 por clase y modelo; fijar el número en `base.yaml`, con semilla).
- [ ] Para **todos** los FN de melanoma: dos mapas por imagen, uno para la clase predicha y otro para `mel` (¿dónde habría mirado para decir melanoma?).
- [ ] Guardar como artefactos: `gradcam/<arch>/<clase>/<image_id>.png`.

### 4. Contraste con ABCDE y atajos espurios (OE6)
- [ ] Anotar cada FN de melanoma en `anotacion_abcde.csv`:

  | Columna | Valores |
  |---|---|
  | `image_id`, `arch` | — |
  | `activacion_en_lesion` | sí / parcial / no |
  | `A_asimetria`, `B_bordes`, `C_color` | ¿la activación cubre la zona relevante? sí / no / no aplica |
  | `atajo` | ninguno / regla / vello / tinta / borde dermatoscopio / otro |
  | `comentario` | texto libre |

- [ ] Solo se evalúan A, B y C: D (diámetro) y E (evolución) no se pueden valorar en imágenes estáticas sin escala.
- [ ] Recomendable: que **dos integrantes** anoten de forma independiente y se calcule el acuerdo (kappa de Cohen). Esto da robustez a un análisis cualitativo.
- [ ] Resumen por modelo en `resumen_atajos.json`: número y % de FN apoyados en elementos ajenos a la lesión.

### 5. Registro en MLflow (`m3-gradcam-<arch>`)
- **Parámetros:** `layer_name`, tamaño de muestra, semilla de muestreo.
- **Tags:** versión del modelo cargado, `split_sha256`, `git_commit`.
- **Métricas:** `n_fn_mel`, `pct_fn_atajo`, `pct_fn_activacion_en_lesion`, `kappa_anotadores` (si se hace).
- **Artefactos:** mapas, `fn_melanoma.csv`, `anotacion_abcde.csv`, `resumen_atajos.json`.

## Para la memoria (capítulo 4, sección M3)
Figuras representativas por clase (aciertos y errores), tabla de FN de melanoma por modelo y clase de destino, tabla de anotación ABCDE resumida, recuento de atajos por modelo y discusión de si el modelo seleccionado por métricas es también el más interpretable.

## Pendiente de decisión
- [ ] ¿Se añade Grad-CAM++? Si se añade, se aplica a los mismos casos y se registra como `method` en los parámetros.
