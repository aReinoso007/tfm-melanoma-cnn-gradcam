# Módulo 2 — Registro de decisiones

Responsable: Alex Reinoso. Aquí se anota **qué se decidió, por qué y qué evidencia lo respalda**, para que la sección M2 del capítulo 4 de la memoria se pueda escribir sin reconstruir nada de memoria. Cada entrada indica si fue acordada con el grupo o si es una decisión de implementación propia que el grupo debe revisar en el PR.

Estado de las tareas: ver las casillas de [modulo2_entrenamiento_evaluacion.md](modulo2_entrenamiento_evaluacion.md).

## Resumen de decisiones del paso 1

El PR #7 se fusionó en `main` el 4 nov 2026 (aprobado por Adrián; Christian confirmó D8 y el grupo ratificó D5 y D10 en la revisión). Las notas para redactar la memoria están en [modulo2_notas_memoria.md](modulo2_notas_memoria.md).

| # | Decisión | Origen | ¿Revisar en el PR? |
|---|---|---|---|
| D1 | Rama `m2/<tarea>` (esta: `m2/modelos-metricas`) | Acordada | No |
| D2 | Lógica en `src/`, notebooks solo la importan | Acordada | No |
| D3 | Misma cabeza en las 3 redes; capas del backbone aplanadas; prefijo `head_`; el modelo no normaliza | Estructura acordada; capas aplanadas, implementación | Revisada y aprobada en el PR #7 |
| D4 | Fase 1: base congelada, solo cabeza | Acordada | No |
| D5 | Fase 2: 30 % final de las capas, BatchNormalization siempre congelada | Fracción y BN acordadas; conteo por capas, implementación | **Ratificada (4 nov): se mantiene y se declara como limitación en la memoria** (73 % / 41 % / 76 % de parámetros; ver `modulo2_notas_memoria.md` §1.1) |
| D6 | Los helpers de congelado no compilan | Implementación | No |
| D7 | Métricas con los nombres de la convención; definiciones estándar | Nombres acordados | No |
| D8 | `predicciones_*.csv` con `y_true`/`y_pred` como nombre de clase | Columnas acordadas; nombres, implementación | **Confirmada por Christian (M3), 4 nov** |
| D9 | Pruebas con `pytest` (22) | Implementación | No |
| D10 | `deterministic=True` en los 9 runs (`training.deterministic` en `base.yaml`) | Medición propia | **Ratificada por el grupo (4 nov); el coste real por época se confirma en la prueba de `sandbox`** |
| D11 | `sandbox` para pruebas; `m2-entrenamiento` y `m2-evaluacion-test` para lo oficial; `estado` a mano | Acordada (convenciones) | No |

## Conclusiones del paso 1 (4 nov 2026)

1. **El código del M1 es consistente** entre repo, Drive y MLflow (hashes del split y del dataset, pesos, pipeline). M2 puede construir sobre él sin cambios.
2. **`src/models.py` y `src/metrics.py` funcionan** en el portátil (Python 3.12) y en Colab (Python 3.13, T4): 22 pruebas en verde, parámetros entrenables idénticos en ambos entornos, formato de `predicciones_*.csv` validado con datos reales y run de prueba creado en `sandbox`.
3. **El determinismo es viable en las tres redes**: ninguna lanza error con `deterministic=True` y las tres se repiten exactamente; sin él, DenseNet121 y EfficientNetB0 no se repiten ni entre dos pasadas idénticas. Cuesta 1,37× a 1,90× por paso (≈ 26 min más por cuenta en el peor caso). Se adopta `True`, pendiente de confirmar el coste con el pipeline completo en `sandbox`.
4. **Hallazgo metodológico para la memoria:** "30 % de las capas" descongela un porcentaje muy distinto de parámetros según la red (D5). Hay que mencionarlo como limitación del protocolo "mismos hiperparámetros base".
5. **Qué no se ha hecho:** ningún entrenamiento real. Los números de las métricas vistas hasta ahora (por ejemplo `val_macro_f1` = 0,178) son de modelos sin entrenar y no son resultados.

---

## Paso 0 — Revisión del estado heredado del Módulo 1 (3 nov 2026)

**Qué se hizo.** Se comprobó que lo que entrega M1 coincide entre el repositorio y la copia de Drive (`TFM/`, descargada a `documentos-drive/`, que no se versiona).

| Comprobación | Resultado |
|---|---|
| SHA-256 de `data/splits/split_v1.csv` (repo) = el de Drive = el de `split_report.json` = el fijado en `src/data.py` | Coincide: `98485423…e34b38` |
| SHA-256 de `HAM10000_metadata.csv` = `dataset_sha256` de `split_report.json` | Coincide: `06aee5e6…fb99a5` |
| `src/data.py`, `src/augmentation.py` y `class_weights.json` del repo frente a los de Drive | Idénticos |
| Imágenes en Drive | 10 015 `.jpg` (las esperadas) |
| Split anterior descartado (`descartado/split_v1.csv`, hash `00ee7ee6…`) | No coincide con el vigente, como debe ser: M1 lo reemplazó por una estratificación desequilibrada con scikit-learn 1.6.1 (`df`: 11 imágenes en test). Por eso `requirements.txt` fija `scikit-learn==1.9.1` |
| Carpeta `checkpoints/` de Drive | Vacía: nadie ha entrenado todavía |

**Por qué.** M2 hereda estos datos sin poder cambiarlos (acuerdo de octubre de 2026). Si el hash no coincidiera, los 9 runs no serían comparables con los de M3. Se verifica una vez aquí y `split_df()` lo vuelve a verificar en cada ejecución.

**Qué se vio de `tfm_common.py` y del notebook `02_entrenamiento_evaluacion.ipynb`.** Se escribieron antes de la entrega de M1 y se apartan del protocolo acordado (datos con `tc.make_dataset`, pesos `balanced`, tres fases, regla de selección sin el mínimo del 50 % por clase, resultados en JSON de Drive en vez de MLflow, prueba evaluada por semilla). **No se corrigen parcheando**: se reescribe por pasos en notebooks `m2_*` apoyados en módulos de `src/`. El notebook antiguo queda como referencia hasta que lo reemplacen.

---

## Paso 1 — `src/models.py` y `src/metrics.py`

Rama: `m2/modelos-metricas` (antes `modulo-2`).

### D1. Nombre de rama

**Decisión.** `m2/<tarea>`; esta rama es `m2/modelos-metricas`.
**Razón.** `docs/02_flujo_de_trabajo.md` §2 reserva `m2/<tarea>` para el Módulo 2 y los commits usan `[m2]`. Un nombre como `modulo-2` no dice qué contiene y no se distingue de la rama del siguiente paso. Una rama por tarea mantiene los PR pequeños, que es lo que permite que otro integrante los revise de verdad. La rama se renombró en local (`git branch -m`); aún no existía en GitHub.
**Origen.** Acordada (flujo de trabajo).

### D2. La lógica va en `src/`, no en el notebook

**Decisión.** Modelos en `src/models.py`, métricas en `src/metrics.py`; los notebooks solo los importan.
**Razón.** `docs/02_flujo_de_trabajo.md` §5. Además: (a) se pueden probar con `pytest` sin GPU ni Colab; (b) el código que se revisa en el PR es el mismo que se ejecuta; (c) M3 puede importar `src.metrics` o cargar los modelos sin copiar celdas.
**Origen.** Acordada.

### D3. Cabeza de clasificación y capas "aplanadas"

**Decisión.** `base ImageNet sin top → GlobalAveragePooling2D → Dropout(0.3) → Dense(7, softmax)`, idéntica en las tres redes. El backbone se construye con `input_tensor` para que sus capas queden directamente en el modelo, y todas las capas de la cabeza se llaman `head_*`.
**Razón.**
- Misma cabeza = la única diferencia entre runs es el backbone, que es la pregunta del TFM.
- Con las capas aplanadas, M3 usa `model.get_layer("conv5_block3_out")` (nombres de `configs/base.yaml: gradcam`). Con un sub-modelo anidado habría que ir a buscarlas dentro de él.
- El prefijo `head_` es la marca para distinguir base y cabeza al congelar, sin listas de nombres por arquitectura.
- El modelo **no normaliza** la entrada: `make_dataset()` ya aplica el `preprocess_input` oficial de cada red (contrato M1→M2). Normalizar dos veces degradaría los resultados sin dar ningún error.
- No se activa precisión mixta: `base.yaml` no la contempla y cambiaría la numérica de los runs. La capa de salida ya fuerza `float32` por si algún día se activa.
**Evidencia.** `tests/test_models.py`: salida `(N, 7)` con suma 1, nombres de cabeza, y existencia de la capa de Grad-CAM con el nombre de `base.yaml` en las 3 redes.
**Origen.** Estructura acordada (guía M2). Capas aplanadas y prefijo `head_`: decisión de implementación, heredada de `tfm_common.build_model`.

### D4. Fase 1: base congelada

**Decisión.** `congelar_base()` pone `trainable=False` en toda la base y `True` en la cabeza.
**Razón.** La cabeza empieza con pesos aleatorios; sus primeros gradientes son grandes y, si la base también se entrenara, deteriorarían los pesos de ImageNet antes de que la cabeza aprenda algo útil.
**Evidencia.** Parámetros entrenables medidos con pesos de ImageNet (= `features × 7 + 7`):

| Arquitectura | Entrenables fase 1 | Total |
|---|---|---|
| ResNet50 | 14 343 | 23 602 055 |
| DenseNet121 | 7 175 | 7 044 679 |
| EfficientNetB0 | 8 967 | 4 058 538 |

### D5. Fase 2: descongelar la fracción superior por número de capas, con BatchNormalization congelada

**Decisión.** `descongelar_fraccion(model, 0.3, congelar_bn=True)` descongela el 30 % final de las capas de la base (valor de `training.phase2.unfreeze_fraction`) pero deja **todas** las `BatchNormalization` congeladas.
**Razón.**
- *Final y no inicio:* las primeras capas detectan bordes y texturas genéricas que ya sirven; las últimas son las más específicas de ImageNet y las que más conviene adaptar a dermatoscopia.
- *BN congelada:* con lotes de 32 y un dominio muy distinto de ImageNet, actualizar la media y varianza móviles degrada los pesos preentrenados (`docs/modulo2_entrenamiento_evaluacion.md` §3). En Keras 3 una BN con `trainable=False` trabaja en modo inferencia, es decir, usa y conserva sus estadísticas guardadas.
- *Por número de capas:* regla única, simple y reproducible para las tres redes.
**Evidencia.**
- `test_bn_no_actualiza_estadisticas_al_entrenar`: tras un `fit` con la fase 2 activa, las estadísticas de una BN de la zona descongelada no cambian.
- Resultado con pesos de ImageNet:

| Arquitectura | Capas base | Descongeladas (30 %) | Primera capa descongelada | Entrenables fase 2 | % de parámetros |
|---|---|---|---|---|---|
| ResNet50 | 174 | 52 | `conv4_block5_1_conv` | 17 199 111 | 72,9 % |
| DenseNet121 | 426 | 128 | `conv4_block23_1_relu` | 2 862 087 | 40,6 % |
| EfficientNetB0 | 237 | 71 | `block6a_dwconv` | 3 063 587 | 75,5 % |

**Ratificado por el grupo (4 nov 2026): se mantiene la regla y se menciona como limitación en la memoria** (texto preparado en `modulo2_notas_memoria.md` §1.1). Contexto original: "30 % de las capas" no equivale a "30 % de los parámetros": en DenseNet121 se entrena el 41 % de los pesos y en ResNet50 y EfficientNetB0 más del 70 %, porque las últimas capas de estas redes concentran más parámetros. La regla es idéntica para las tres, así que cumple "mismos hiperparámetros base", pero es una limitación que hay que **mencionar en la memoria**. Si se prefiere igualar la cantidad de parámetros entrenables, habría que cambiar la regla de `base.yaml` y eso exige PR aprobado por los tres. No se cambia sin acuerdo.
**Origen.** Fracción y BN congelada: acordadas (`base.yaml`). Conteo por capas: decisión de implementación.

### D6. Los helpers no compilan el modelo

**Decisión.** `congelar_base()` y `descongelar_fraccion()` solo cambian `trainable`; no llaman a `compile()`.
**Razón.** Keras exige recompilar tras cambiar `trainable`, y el optimizador y el LR dependen de la fase (`f1_lr`, `f2_lr`). Ese lugar es el código de entrenamiento (paso siguiente); así un helper no puede dejar un modelo con un LR equivocado.

### D7. Métricas: nombres de la convención y definiciones

**Decisión.** `calcular_metricas(y_true, y_prob)` devuelve un diccionario plano con los nombres de `docs/03_mlflow_convenciones.md` §5; `prefijar(m, "val"|"test")` produce el nombre exacto de MLflow.
**Razón.** Si los nombres difieren entre validación y prueba o entre el notebook y la memoria, no se puede rastrear un número hasta su run (OE7). Con una sola función para ambos conjuntos los números son comparables por construcción.
**Definiciones usadas** (todas se prueban en `tests/test_metrics.py`):
- Por clase: uno contra el resto (OvR). Sensibilidad = TP/(TP+FN); precisión = TP/(TP+FP); especificidad = TN/(TN+FP); F1 = media armónica.
- `macro_f1`: promedio simple de los F1 por clase; cada clase pesa igual aunque tenga 17 imágenes (`df` en test).
- `balanced_accuracy` = media de las sensibilidades (igual que sklearn; se verifica en un test).
- `min_class_sensitivity` = la menor sensibilidad de las 7 clases (requisito 2 de la regla de selección).
- `mel_*`: las mismas métricas para la clase positiva `mel`, más `mel_ap` (área bajo precisión-recall).
- `accuracy`: se registra pero **no** se usa para elegir (con 67 % de `nv` engaña).
- Denominador 0 (p. ej. precisión de una clase que nunca se predice) → `0.0`, como `zero_division=0`.
- AUC o AP imposibles de calcular (falta una clase en `y_true`, solo ocurre con submuestras del sandbox) → `NaN`, **no 0**: un 0 sería un valor inventado.
**Origen.** Nombres acordados (convenciones); definiciones: estándar de scikit-learn.

### D8. `predicciones_<val|test>.csv` (contrato M2 → M3)

**Decisión.** `tabla_predicciones(split_df(...), y_prob)` genera `image_id, lesion_id, y_true, y_pred, p_akiec … p_vasc`, con `y_true` e `y_pred` como **nombre de clase** (`mel`, `nv`…).
**Razón.**
- El contrato M2→M3 fija las columnas pero no si las clases son índices o nombres. Con nombres el CSV se entiende solo y no depende de recordar el orden de `CLASES`; M3 filtra falsos negativos con `y_true == "mel"`.
- Las probabilidades solo se pueden emparejar con las imágenes si están en el orden de `split_df()`. Un desorden no da error, simplemente asigna probabilidades a la imagen equivocada. Por eso la función comprueba longitud y que el split venga ordenado por `image_id`.
**Confirmado por Christian (M3) el 4 nov 2026:** los nombres de clase le sirven.
**Origen.** Columnas acordadas; nombres frente a índices: decisión de implementación.

### D9. Pruebas automáticas

**Decisión.** `tests/` con `pytest` (22 pruebas). Los modelos se prueban con `weights=None`, sin descargar ImageNet.
**Razón.** Un error en la métrica o en el congelado no da ningún aviso: el entrenamiento corre igual y solo se nota semanas después, con 9 runs de GPU ya gastados. Probar antes en el portátil es gratis.
**Cómo se ejecuta.** `pip install -r requirements.txt pytest` y `python -m pytest tests -q` desde la raíz. Entorno de comprobación: Python 3.12, TensorFlow 2.21.0, Keras 3.15.1, scikit-learn 1.9.1. Colab usa Python 3.13 con las mismas versiones de TensorFlow, Keras y scikit-learn.

### D10. Determinismo: regla fijada antes de medir

**Decisión.** Antes del primer entrenamiento se mide, para las tres redes, (a) si `deterministic=True` lanza error, (b) si dos pasadas con la misma semilla dan **exactamente** las mismas pérdidas y pesos, y (c) cuánto tiempo por paso cuesta frente a `deterministic=False`. La regla de decisión es:

| Resultado de la medición | Decisión |
|---|---|
| Alguna red lanza error con `deterministic=True` | `deterministic=False` en **las tres** redes |
| `deterministic=True` no se repite en alguna red | No se entrena hasta entender por qué |
| Se repite en las tres y la penalización de tiempo es ≤ 1,3× | `deterministic=True` en los 9 runs |
| Se repite en las tres pero la penalización es > 1,3× | Decisión del grupo (cuota de GPU frente a repetibilidad) |

**Razón.**
- *Medir antes de entrenar:* el bloque 4 de la verificación solo comprobó que no hay error en EfficientNetB0. "No da error" no es "se repite": lo que interesa es lo segundo. Descubrir un fallo en ResNet50 o DenseNet121 en el run 4 de 9 gastaría cuota de GPU gratuita.
- *Fijar la regla antes de ver el número:* evita elegir el umbral de penalización a conveniencia una vez visto el resultado.
- *Las tres redes con el mismo modo:* mezclar modos rompería la condición de "condiciones idénticas" (OE2). Si una falla, las tres pasan a `False`.
- *Un proceso por combinación (`scripts/bench_determinismo.py`):* `enable_op_determinism()` no se puede desactivar dentro de un proceso, así que comparar los dos modos exige procesos separados. Los lotes se cargan en memoria antes de medir para que el tiempo sea el del paso de entrenamiento y no el de leer JPEG.
- *No sobrevalorar el determinismo:* es repetible solo con el mismo hardware y versiones, y los runs se reparten entre tres cuentas de Colab que pueden recibir GPUs distintas. La reproducibilidad que se defiende en la memoria es: semillas fijas + trazabilidad (`git_commit`, versiones, GPU, `split_sha256`) + media ± desviación estándar sobre 3 semillas. El aumento de datos ya es repetible por sí mismo (operaciones con semilla por imagen en `src/augmentation.py`).
- *Se registra siempre:* cada run guarda `deterministic`, la GPU y la versión de TensorFlow como parámetros, sea cual sea la decisión.
**Evidencia (Colab, T4, 4 nov 2026; 10 lotes por fase, dos pasadas con la misma semilla, `scripts/bench_determinismo.py`).**

| Arquitectura | `det=False`: ms/paso | repetible | dif. máx. de pérdida | `det=True`: ms/paso | repetible | Penalización |
|---|---|---|---|---|---|---|
| ResNet50 | 132,4 | sí | 0 | 181,3 | sí | 1,37× |
| DenseNet121 | 107,0 | **no** | 2,4·10⁻⁷ | 173,4 | sí | 1,62× |
| EfficientNetB0 | 48,9 | **no** (pesos distintos, pérdida igual) | 0 | 93,1 | sí | 1,90× |

Lectura:
- Ninguna red lanza error con `deterministic=True`, y las tres se repiten exactamente con él.
- Sin determinismo, DenseNet121 y EfficientNetB0 ya difieren entre dos pasadas idénticas, aunque a un nivel de redondeo (≈10⁻⁷ en la pérdida tras 20 pasos). Aquí no se midió cuánto se amplifica esa diferencia a lo largo de un entrenamiento completo.
- Las tres penalizaciones superan el umbral de 1,3× fijado antes de medir, así que **por la regla no hay decisión automática: la toma el grupo**.

**Decisión adoptada y ratificada por el grupo (4 nov 2026): `deterministic=True` en las tres redes**, sujeta solo a la comprobación del coste real por época en `sandbox` (si la proyección supera ≈ 2 h por cuenta se vuelve a plantear). Queda reflejada en `configs/base.yaml` (`training.deterministic: true`).

**Razonamiento.** El umbral de 1,3× era una aproximación a "el coste es asumible"; el coste absoluto, calculado con estos tiempos, es pequeño:

| (peor caso: 45 épocas por run, 220 lotes por época, sin parada anticipada) | `det=False` | `det=True` |
|---|---|---|
| ResNet50, un run | ≈ 22 min | ≈ 30 min |
| DenseNet121, un run | ≈ 18 min | ≈ 29 min |
| EfficientNetB0, un run | ≈ 8 min | ≈ 15 min |
| Una cuenta (las 3 redes, 1 semilla, según el reparto de `04_colab.md` §7) | ≈ 48 min | ≈ 74 min |

Son ≈ 26 min más por cuenta en el peor caso. Es un orden de magnitud, no una promesa: el cálculo usa el tiempo medio de las dos fases juntas (la fase 1 es más ligera) y **no incluye** la lectura de datos ni la predicción en validación de cada época. Con determinismo, `tf.data` además fija el orden de sus operaciones paralelas, lo que podría frenar la entrada de datos y no está medido aquí.
- *A favor de `True`:* resultado exactamente repetible en el mismo hardware, lo que respalda OE7 y permite que otro integrante o el director reproduzca un número de la memoria; sin errores en ninguna red.
- *En contra:* +26 min de GPU por cuenta, y la repetibilidad exacta solo vale con la misma GPU y versiones (los runs se reparten entre tres cuentas).
- **Condición para confirmarlo:** en la prueba de `sandbox` (paso siguiente) se mide el tiempo real por época con el pipeline completo. Si la proyección a 45 épocas supera ≈ 2 h por cuenta, se vuelve a plantear al grupo.
- Cualquiera que sea el modo, se registra en cada run como `deterministic` (junto con GPU y versión de TensorFlow). Si se confirma, se añadirá a `configs/base.yaml` (`training.deterministic: true`), cambio que necesita el PR aprobado por los tres.

**Origen.** Medición propia; ratificada por el grupo el 4 nov 2026. Cada run registra `deterministic`, `gpu` y `tf_version` (`docs/03_mlflow_convenciones.md` §4).

### D11. Dónde se registra cada run en MLflow

**Decisión.**

| Qué | Experimento | Nombre del run |
|---|---|---|
| Pruebas de entorno, verificaciones, ensayos de 2 épocas con subconjunto | `sandbox` | libre (p. ej. `m2-verificacion-paso1`) |
| Los 9 entrenamientos | `m2-entrenamiento` | `m2-<arch>-s<seed>` |
| Las 3 evaluaciones finales en prueba | `m2-evaluacion-test` | `m2-test-<arch>` |

**Razón.**
- `sandbox` es exactamente para esto: `docs/03_mlflow_convenciones.md` §1 lo define como "pruebas y depuración, nada de aquí se cita en la memoria". Adrián hizo lo mismo: su prueba de entorno (`m1-prueba-entorno`) está en `sandbox` y sus runs oficiales (`m1-eda`, `m1-split-v1`, `m1-pipeline`…) en `m1-datos`. Eso se comprobó en la API de DagsHub el 4 nov.
- Mantener los experimentos oficiales limpios hace que la checklist de reproducibilidad (§8) sea una consulta simple: "todos los runs de `m2-entrenamiento`" son exactamente los 9 que cuentan, sin tener que filtrar pruebas.
- `m2-entrenamiento` y `m2-evaluacion-test` no existen todavía: `mlflow.set_experiment()` los crea en el primer uso. No hace falta crearlos a mano.
- Todos los runs nacen con `estado=en_curso` (lo pone `standard_tags()`); al cerrar el paso hay que marcar `final` o `descartado` a mano, como hizo Adrián con su split anterior. Un run oficial que sale mal se marca `descartado`, no se borra: queda la evidencia de lo que se probó.
- Para evitar que un run de prueba acabe en el experimento oficial por descuido, el notebook de entrenamiento tendrá un único interruptor `MODO = "sandbox" | "real"` que elige el experimento. En modo `real` se negará a arrancar si `git_dirty=true`, porque esos runs no valen como resultado final (`docs/02_flujo_de_trabajo.md` §6).
**Origen.** Acordada (convenciones de MLflow). El interruptor `MODO` es decisión de implementación.

### Comprobación de integración con el pipeline de M1

Con pesos de ImageNet y las imágenes locales (`documentos-drive/HAM10000-002/images`), se pasaron 2 lotes de validación de `make_dataset("val", arch, training=False)` por cada una de las tres redes: salida `(64, 7)` con probabilidades que suman 1. Confirma que el modelo y el pipeline de M1 son compatibles (tamaño 224×224, normalización fuera del modelo).

### Verificación en Colab (4 nov 2026, `m2_00_verificacion_colab.ipynb`, rama `m2/modelos-metricas`)

| Bloque | Resultado | Lectura |
|---|---|---|
| 1. Sesión | GPU `Tesla T4` (15 360 MiB), Python 3.13.16, commit `0c19a11` de la rama; dataset descomprimido en 92 s; `[+] 10015 imágenes, metadatos y split v1 verificados` | Entorno igual al previsto (T4 y Python 3.13); los hashes de datos y split coinciden también en Colab |
| 2. Pruebas (Python 3.13, sklearn en `dist-packages` de Colab) | `22 passed` en 114 s; solo los 3 avisos esperados de AUC sin dos clases | El código funciona en el entorno real, no solo en Python 3.12 |
| 3. Redes con ImageNet y lotes de `make_dataset()` | Los 6 conteos de parámetros por fase y los % coinciden exactamente con los de local (ResNet50 14 343 → 17 199 111; DenseNet121 7 175 → 2 862 087; EfficientNetB0 8 967 → 3 063 587) | El congelado se comporta igual en Colab; confirma D4 y D5 |
| 4. `set_seeds(42, deterministic=True)` + 1 paso en cada fase (EfficientNetB0) | Sin error: fase 1 `loss=1.846`, fase 2 `loss=1.503` | `enable_op_determinism()` no falla en esa configuración. Cubre solo EfficientNetB0 y 3 lotes; ResNet50 y DenseNet121 se confirman en el primer entrenamiento real (ver riesgo) |
| 4b. `scripts/bench_determinismo.py` en las 3 redes, `det` False y True, dos pasadas con la misma semilla | Sin errores; con `det=True` las tres se repiten; con `det=False` DenseNet121 y EfficientNetB0 no; penalización 1,37× / 1,62× / 1,90× | Detalle y decisión en D10 |
| 5. Métricas y `tabla_predicciones` con 64 imágenes reales de validación (sin entrenar) | Corren sin error; columnas `image_id, lesion_id, y_true, y_pred, p_akiec … p_vasc`; `y_true`/`y_pred` como nombre de clase; probabilidades por fila suman 1; aviso de AUC por una sola clase (esperado con 64 imágenes consecutivas) | El formato del contrato M2 → M3 funciona con datos reales. Los valores (`val_macro_f1` 0,178, `val_mel_sensitivity` 0,1) son de un modelo sin entrenar y no significan nada |
| 6. MLflow/DagsHub | Run `m2-verificacion-paso1` creado; `git_dirty = False`. Se comprobó por la API de MLflow que el experimento de ese run (id 0) se llama `sandbox` | Secretos y token con permiso de escritura correctos; el código del run está limpio. Las convenciones de `sandbox` se respetan |

Las pérdidas del bloque 4 salen de 3 lotes de validación sin entrenar de verdad: solo indican que compila y que la pérdida baja, no sirven como resultado.

**Estado: verificación del paso 1 cerrada.** Los seis bloques se ejecutaron y coinciden con lo esperado.

**Riesgo residual del determinismo.** El bloque 4 solo ejercita EfficientNetB0. Las otras dos redes usan operaciones distintas (`ResNet50`: convoluciones y pooling; `DenseNet121`: concatenaciones). Si alguna falla con `deterministic=True`, la opción es `deterministic=False` registrado como parámetro del run, decisión del grupo porque afecta a la reproducibilidad (OE7).

### Qué NO se hizo todavía (siguientes pasos)

0. ~~Aprobación del PR #7~~ (hecho el 4 nov 2026: D5, D8 y D10 ratificadas).
1. Callback y bucle de entrenamiento de las dos fases (`f1_*`, `f2_*` en el mismo run, checkpoints en Drive, `BackupAndRestore`, interruptor `MODO`, registro de `deterministic`, `gpu` y `tf_version`).
2. Prueba en `sandbox` con subconjunto pequeño y 2 épocas, en Colab con GPU: confirma además el tiempo real por época con `deterministic=True` (D10).
3. Los 9 runs, la selección, la evaluación en prueba y el registro de modelos.
