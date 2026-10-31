# Módulo 1 — Preparación del conjunto de datos (OE1)

**Responsable:** Adrián Arias · **Experimento MLflow:** `m1-datos` · **Bloquea a:** M2 y M3

## Objetivo

Entregar un conjunto de datos preparado **de forma idéntica** para las tres arquitecturas, de modo que las diferencias de rendimiento se deban a los modelos y no a los datos.

## Pasos

### 1. Obtención de los datos
- [ ] Descargar HAM10000 (imágenes + `HAM10000_metadata.csv`) desde ISIC Archive o Harvard Dataverse a Drive.
- [ ] Calcular el SHA-256 del CSV de metadatos y guardarlo como `dataset_sha256`.
- [ ] Comprobar que hay 10,015 imágenes y que todas las de los metadatos existen.

### 2. Análisis exploratorio (EDA)
- [ ] Distribución de las 7 clases (conteo y %).
- [ ] Número de lesiones únicas (`lesion_id`) y de imágenes por lesión.
- [ ] Distribución de edad, sexo y localización por clase.
- [ ] Muestra visual de imágenes por clase y presencia de artefactos (reglas, vello, tinta, bordes del dermatoscopio). Esto sirve después al M3.
- [ ] Registrar todas las figuras en MLflow.

### 3. Partición (decisión crítica, no se cambia después)
- [ ] `StratifiedGroupKFold` con `groups=lesion_id` y `y=dx`, con una semilla fija (`split_seed` en `base.yaml`), para obtener ~70/15/15.
  - Una forma sencilla: `StratifiedGroupKFold(n_splits=20)` y asignar 3 folds a prueba, 3 a validación y 14 a entrenamiento (15/15/70).
- [ ] **Test de no fuga:** ningún `lesion_id` aparece en más de un subconjunto (`assert`).
- [ ] Informe con conteos por clase y subconjunto (`split_report.json`). Las clases pequeñas (df, vasc) deben tener representación en los tres.
- [ ] Guardar `data/splits/split_v1.csv` (`image_id, lesion_id, dx, split`), hacer commit en el repo y registrarlo como artefacto.
- [ ] Calcular `split_sha256` del CSV: **es la huella que M2 y M3 deben citar en cada run**.

### 4. Estrategia frente al desbalance
- [ ] Pesos de clase calculados **solo con entrenamiento** → `class_weights.json`. Decisión final: pesos suavizados (raíz cuadrada de `balanced`, normalizados a peso medio 1; `class_weighting: "sqrt"` en `base.yaml`). Se leen con `load_class_weights()` de `src/data.py`.
- [ ] Aumento de datos **solo en entrenamiento** y solo para clases minoritarias (definir cuáles; p. ej. todas menos `nv`). Transformaciones: rotación, volteo horizontal y vertical, zoom ligero, brillo y contraste ligeros.
- [ ] Documentar en `base.yaml` qué transformaciones y con qué parámetros.

### 5. Pipeline `tf.data`
- [ ] Función en `src/data.py`:
  ```python
  make_dataset(split: str, arch: str, training: bool) -> tf.data.Dataset
  ```
  - Lee de `split_v1.csv` y redimensiona a `img_size` (224).
  - Aplica el `preprocess_input` propio de cada arquitectura (ResNet50, DenseNet121 y EfficientNetB0 tienen preprocesados distintos; la diferencia se documenta, no se elimina).
  - Aumento y barajado solo si `training=True`.
  - `cache` / `prefetch(tf.data.AUTOTUNE)`.
- [ ] Probar que el mismo `image_id` produce el mismo tensor dos veces seguidas con `training=False`.

### 6. Registro en MLflow (`m1-split-v1`)
- **Parámetros:** `split_seed`, proporciones, `n_splits`, `img_size`, transformaciones de aumento.
- **Tags:** `dataset_sha256`, `split_sha256`, `split_version=v1`.
- **Artefactos:** `split_v1.csv`, `class_weights.json`, `split_report.json`, figuras EDA.

## Entregable para M2
- `data/splits/split_v1.csv` en `main` + `split_sha256` comunicado al grupo.
- `class_weights.json`.
- `make_dataset()` en `src/data.py`, revisado por M2 en un PR.

## Para la memoria (capítulo 4, sección M1)
Tabla de distribución de clases, tabla de la partición por clase y subconjunto, figura de ejemplos por clase, justificación de la agrupación por lesión y de la estrategia de desbalance.
