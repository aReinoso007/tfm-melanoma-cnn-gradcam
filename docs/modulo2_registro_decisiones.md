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
| D10 | `deterministic=True` en los 9 runs (`training.deterministic` en `base.yaml`) | Medición propia | **Ratificada (4 nov) y confirmada en `sandbox` (7 nov): peor caso 83 min por cuenta, bajo las ≈ 2 h** |
| D11 | `sandbox` para pruebas; `m2-entrenamiento` y `m2-evaluacion-test` para lo oficial; `estado` a mano | Acordada (convenciones) | No |
| D12 | Las dos fases van en el mismo run, con `step` = época acumulada (fase 2 continúa tras la 1) | Estructura acordada (`f1_*`, `f2_*`); `step` acumulado, implementación | Sí (PR del paso 2) |
| D13 | Parada anticipada y mejores pesos por fase con un callback propio persistente (`ControlFase`), no `EarlyStopping` de Keras; monitor `val_loss`; el modelo final es el mejor de la fase 2 | Monitor y paciencia acordados; el callback propio, implementación | Sí |
| D14 | Una sola pasada de validación por época: `val_loss` sin pesos de clase y `val_macro_f1`, para parada y para el log | Implementación | Sí |
| D15 | Reanudación tras desconexión de Colab: `BackupAndRestore` + estado en JSON en Drive + marca `<fase>_completa.json` | Acordada la idea (checkpoints en Drive); el diseño, implementación | Sí |
| D16 | `lanzar_run(modo="sandbox"\|"real")`: en `real` no admite `epochs`/`max_lotes`, exige árbol limpio y confirmación explícita | Interruptor acordado en D11; las guardas, implementación | Sí |
| D17 | El modelo se sube con `save_model` + `log_artifacts` (`modelo/`), no con `log_model` de MLflow 3 | Implementación (DagsHub no sirve el Logged Model) | Sí. Comprobado en DagsHub el 7 nov: las tres redes se recargan y dan las mismas probabilidades |
| D19 | v2: el mismo aumento para las siete clases; experimento `m2-entrenamiento-v2` | Grupo, 7 nov 2026, tras la comprobación del atajo | Sí. Los runs de la v1 se conservan; los alias se mueven en D20 |
| D20 | Selección v2: EfficientNetB0, semilla 2024 en las tres redes; `umbral_mel` anotado y la prueba sigue cerrada | Alex, 7 nov 2026, run `905dd96f099c4a968e33f3247318081c` | Sí, en el PR de `m2/seleccion-v2` |
| D21 | Evaluación en prueba, una sola vez, argmax y umbral; sin cambiar modelo ni selección | Alex, 7 nov 2026, runs `m2-test-*` | Sí, en el mismo PR |

## Conclusiones del paso 1 (4 nov 2026)

1. **El código del M1 es consistente** entre repo, Drive y MLflow (hashes del split y del dataset, pesos, pipeline). M2 puede construir sobre él sin cambios.
2. **`src/models.py` y `src/metrics.py` funcionan** en el portátil (Python 3.12) y en Colab (Python 3.13, T4): 22 pruebas en verde, parámetros entrenables idénticos en ambos entornos, formato de `predicciones_*.csv` validado con datos reales y run de prueba creado en `sandbox`.
3. **El determinismo es viable en las tres redes**: ninguna lanza error con `deterministic=True` y las tres se repiten exactamente; sin él, DenseNet121 y EfficientNetB0 no se repiten ni entre dos pasadas idénticas. Cuesta 1,37× a 1,90× por paso (≈ 26 min más por cuenta en el peor caso). Se adopta `True`. El 7 nov, con el pipeline completo en `sandbox`, la proyección quedó en 83 min por cuenta (ver D10).
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

**Decisión adoptada y ratificada por el grupo (4 nov 2026): `deterministic=True` en las tres redes**, sujeta a que el coste real en `sandbox` no superara ≈ 2 h por cuenta. Esa condición se cumplió el 7 nov (83 min; ver la confirmación más abajo). Queda reflejada en `configs/base.yaml` (`training.deterministic: true`).

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
- **Condición para confirmarlo:** en la prueba de `sandbox` se mide el tiempo real por época con el pipeline completo. Si la proyección a 45 épocas supera ≈ 2 h por cuenta, se vuelve a plantear al grupo.
- Cualquiera que sea el modo, se registra en cada run como `deterministic` (junto con GPU y versión de TensorFlow). Quedó en `configs/base.yaml` (`training.deterministic: true`) en el PR #7.

**Confirmación (7 nov 2026, `m2_01_entrenamiento.ipynb`, T4, datos completos, 2+2 épocas, semilla 42).** Las tres redes terminaron sin error con `deterministic=True`. La proyección usa el tiempo por paso de la última época de cada fase (220 pasos, 15+30 épocas, sin parada anticipada) y no incluye la validación:

| Arquitectura | ms/paso fase 1 | ms/paso fase 2 | Peor caso |
|---|---|---|---|
| ResNet50 | 136 | 199 | 29,3 min |
| DenseNet121 | 335 | 180 | 38,2 min |
| EfficientNetB0 | 79 | 103 | 15,6 min |
| Total por cuenta (3 redes, 1 semilla) | | | **83,1 min** |

La validación (1 502 imágenes, solo hacia delante) no entra en `s_por_paso`. Añadirle un 10 % deja el total en torno a 90 min, por debajo de las ≈ 2 h. **La condición se cumple y se mantiene `deterministic=True`.** El cálculo previo del banco, sin lectura de datos, daba ≈ 74 min: el pipeline completo no cambia el orden de magnitud.

En DenseNet121 la última época de la fase 1 salió más lenta que la de la fase 2 (335 frente a 180 ms/paso). Lo esperable es lo contrario, porque la fase 2 entrena más capas. Con solo dos épocas, esa última de la fase 1 puede seguir incluyendo el calentamiento de `tf.data`, así que 83 min es un techo. Los `val_macro_f1` de estos runs (0,426, 0,428 y 0,313) son de cuatro épocas y no se citan. Runs en `sandbox`: `51776efe5975433eb0f473a30c7b8989`, `9e97c522ffc949e5b3d94ae32275a10d`, `ad3fdffd95e04be4a6484f1019a77e35`. La carga del modelo en esa misma sesión falló: ver D17.

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
2. ~~Prueba en `sandbox` en Colab~~ (hecha el 7 nov 2026: las tres redes, 2+2 épocas, datos completos; D10 confirmada, 83 min por cuenta).
3. Los 9 runs, la selección, la evaluación en prueba y el registro de modelos.

---

## Paso 2 — `src/training.py`, `tests/test_training.py` y `notebooks/m2_01_entrenamiento.ipynb`

Rama `m2/entrenamiento`. Entrega: el bucle de entrenamiento de las dos fases, la reanudación tras desconexión, el registro en MLflow y el notebook que lo lanza en modo `sandbox` o `real`. La prueba de `sandbox` en Colab está hecha (7 nov); **todavía no hay ningún entrenamiento real**.

### D12. Las dos fases en el mismo run, con `step` acumulado

**Decisión.** Un solo run por (arquitectura, semilla) con métricas `f1_*` y `f2_*`. El `step` de MLflow es la **época acumulada**: si la fase 1 dura 15 épocas, la primera de la fase 2 se registra en el `step` 16.

**Razón.**
- Es lo que dicen las convenciones (`docs/03_mlflow_convenciones.md`): 9 runs oficiales, no 18.
- Con el `step` acumulado las dos fases se dibujan en la misma gráfica sin que la fase 2 pise a la 1; la subida de la pérdida al descongelar (se ve incluso en la prueba local: 2,02 → 2,46 con 2 lotes) queda visible y se puede explicar en la memoria.

### D13. `ControlFase` en vez de `EarlyStopping` de Keras

**Decisión.** La parada anticipada y la restauración de los mejores pesos se implementan en un callback propio (`ControlFase`) que guarda su estado (mejor `val_loss`, mejor época, épocas sin mejora) en `<fase>_control.json` y los mejores pesos en `<fase>_mejor.weights.h5`. Monitor `val_loss`, paciencia 5 (`configs/base.yaml`). Al terminar la fase se restauran los mejores pesos. **El modelo final de cada run es el mejor de la fase 2**; el mejor `val_loss` de cada fase se registra (`f1_mejor_val_loss`, `f2_mejor_val_loss`) y la mejor época de cada fase también.

**Razón.**
- `EarlyStopping` guarda su contador y sus mejores pesos **en memoria**. Si Colab se desconecta y `BackupAndRestore` reanuda en la época 12, el `EarlyStopping` nuevo empieza de cero: contaría la paciencia desde ahí y `restore_best_weights` devolvería la mejor época *posterior al reinicio*, no la mejor de toda la fase. El resultado dependería de si hubo desconexión, y eso rompe la reproducibilidad (OE7).
- Con el estado en disco, un run interrumpido y reanudado llega exactamente a la misma decisión que uno sin interrupción. Está probado en `tests/test_training.py` (simulando la desconexión).
- El modelo final es el de la fase 2 porque es el que se evalúa y el que usa M3. Registrar también `f1_mejor_val_loss` permite ver si descongelar mejoró algo (comparación que irá en la memoria).

### D14. Una sola pasada de validación por época

**Decisión.** Un callback (`EvaluarValidacion`) recorre `val` una vez por época, calcula `val_loss` (entropía cruzada **sin pesos de clase**) y `val_macro_f1`, los inyecta en `logs` (así los ve `ControlFase`) y los registra en MLflow junto con `loss` de entrenamiento y `s_por_paso`.

**Razón.**
- Evita pasar dos veces por validación (una para Keras y otra para el F1), que son 1 502 imágenes por época.
- `val_loss` sin pesos de clase: el `loss` de entrenamiento sí lleva pesos (`class_weight`), así que no es comparable con el de validación. La parada se hace con la pérdida sin pesos porque mide la calidad real de las probabilidades sobre la distribución real de las clases; el macro-F1 se registra para decidir (selección) pero no para parar, siguiendo la convención.
- `s_por_paso` (tiempo por lote) se registra cada época: es lo que alimenta la proyección de coste de D10.

### D15. Reanudación tras desconexión

**Decisión.** Tres capas, todas en `checkpoints_dir` (Drive):
1. `BackupAndRestore` de Keras por fase (`backup_<fase>/`): reanuda en la época interrumpida.
2. Estado propio en JSON escritos de forma atómica (archivo temporal + `os.replace`, para que una desconexión a mitad de escritura no deje un JSON corrupto): `<fase>_historial.json` (métricas por época, sin duplicar épocas al reanudar), `<fase>_control.json` y `<fase>_mejor.weights.h5`.
3. Marca `<fase>_completa.json`: si existe, la fase no se vuelve a entrenar y solo se recargan los mejores pesos. Así, una desconexión durante la fase 2 no repite la fase 1.

**Razón.** Los runs largos (hasta ≈ 2 h por cuenta en el peor caso, D10) sobre Colab gratuito se van a interrumpir. Sin reanudación fiable, perder 1 h de GPU por una desconexión obligaría a repetir y comprometería el presupuesto de cuota. Probado con datos sintéticos 64×64 en `tests/test_training.py` (no duplica épocas, no repite fases completas).

**Límite honesto.** `BackupAndRestore` restaura pesos y optimizador, pero no el estado del generador aleatorio del `shuffle`: un run reanudado a mitad es válido, pero **no es idéntico bit a bit** a uno ininterrumpido. Si un run se reanuda, hay que dejarlo anotado (el notebook lo muestra) y mencionarlo si afecta a un resultado oficial.

### D16. `lanzar_run`: modos `sandbox` y `real`

**Decisión.**

| | `sandbox` | `real` |
|---|---|---|
| Experimento MLflow | `sandbox` | `m2-entrenamiento` |
| Nombre del run | `m2-<arch>-s<seed>-sandbox` | `m2-<arch>-s<seed>` |
| Estado en Drive | `checkpoints/sandbox/…` | `checkpoints/…` |
| `epochs` / `max_lotes` | permitidos | **`ValueError`** |
| `git_dirty=true` | permitido | **`RuntimeError`** |
| Confirmación | no | el notebook exige `CONFIRMO_REAL = True` |

**Razón.** Es el interruptor prometido en D11. Las guardas hacen imposible, por un descuido, que un run acortado o hecho con código sin commitear acabe en el experimento oficial. Los estados en carpetas distintas evitan que un ensayo `sandbox` "complete" una fase que luego el run real daría por entrenada. Con `max_lotes`, el dataset se recorre entero una vez antes de medir para que la caché de `tf.data` esté caliente y el `s_por_paso` sea el real y no el del primer pase (que incluye decodificar JPEG).

### D17. El modelo va como artefacto del run, no como Logged Model

**Decisión.** `publicar_modelo()` guarda el modelo con `mlflow.tensorflow.save_model` y lo sube con `MlflowClient.log_artifacts(..., artifact_path="modelo")`. Se carga con `mlflow.tensorflow.load_model("runs:/<id>/modelo")`, que es la ruta que el registro de modelos convierte luego en `models:/ham10000-<arch>@final`.

**Razón.** En MLflow 3, `log_model(..., name="modelo")` no escribe en los artefactos del run: crea un *Logged Model* aparte. Con un servidor completo (SQLite local, 4 nov) `runs:/<id>/modelo` sí resuelve. DagsHub no: el 7 nov, con los tres runs de `sandbox` ya creados, la sección 4 del notebook falló con `Failed to download artifacts from path 'modelo'`. Los demás artefactos del mismo run (`curvas.png`, el CSV) sí estaban. El formato en disco es el mismo (`MLmodel` + pesos); cambia solo dónde se sube, y ese sitio es el que DagsHub ya sirve.

**Cómo se recupera lo ya entrenado.** Los pesos de la fase 2 siguen en Drive (`f2_mejor.weights.h5`). La sección 4, si no encuentra `modelo/`, reconstruye la red, carga esos pesos y los publica en el run cerrado. No hace falta repetir las 2+2 épocas.

**Comprobado en DagsHub (7 nov 2026, misma sesión de Colab).** Tras publicar desde Drive, `load_model("runs:/<id>/modelo")` descargó 7 artefactos en las tres redes y `predict` coincidió con las probabilidades guardadas del entrenamiento (`atol=1e-4`): ResNet50, DenseNet121 y EfficientNetB0, las tres `True`, salida `(None, 7)`. El aviso de MLflow de que el modelo se guarda sin *signature* no afecta a esta carga: el M3 usa `mlflow.tensorflow.load_model`, no la inferencia por `pyfunc`. La primera tentativa de descarga, cuando `modelo/` aún no existía, espera unos 4 minutos antes de fallar; en un run nuevo el artefacto ya está y esa espera no se produce.

### Verificación local del paso 2 (4 nov 2026)

- `pytest tests -q`: 28 pruebas en verde (22 de modelos y métricas y 6 nuevas de entrenamiento), 56 s en el portátil.
- **Ejecución de extremo a extremo en el portátil** (CPU, sin GPU, imágenes reales de `documentos-drive/`, EfficientNetB0 con ImageNet, 2 lotes por época, 1+1 épocas, MLflow local con SQLite): el run se crea con las etiquetas esperadas (`modulo=m2`, `fase=entrenamiento`, `modo=sandbox`, `arch`, `seed`), registra 61 métricas, los parámetros `deterministic=True`, `tf_version=2.21.0`, `gpu=sin GPU`, `n_capas_descongeladas=71`, y los 4 artefactos (`curvas.png`, `confusion_val_norm.png`, `classification_report_val.json`, `predicciones_val.csv`). Con el servidor SQLite local, `mlflow.tensorflow.log_model(..., name="modelo")` se vuelve a cargar con `mlflow.tensorflow.load_model("runs:/<id>/modelo")`: `Functional`, salida `(None, 7)`. Ese atajo no existe en DagsHub (D17): allí el modelo tiene que ir como artefacto del run.
- Los tiempos de esa ejecución (≈ 2,5 y 3,3 s/paso) son de CPU y **no valen para nada** salvo para comprobar que el campo se registra; el coste real se mide en la GPU de Colab.
- La celda de carga contra DagsHub falló el 7 nov por la mañana y, con `publicar_modelo`, quedó verificada esa misma tarde: las tres redes, `True` (D17).

### Verificación en Colab del paso 2 (7 nov 2026, `m2_01_entrenamiento.ipynb`, `MODO = "sandbox"`)

| Qué | Resultado |
|---|---|
| Tres redes, 2+2 épocas, datos completos, semilla 42, `deterministic=True` | Sin error. Tiempos de pared: ResNet50 6,3 min, DenseNet121 7,0 min, EfficientNetB0 4,4 min (incluyen el primer epoch en frío y la validación) |
| Proyección de peor caso | 29,3 + 38,2 + 15,6 = **83,1 min por cuenta**. Condición de D10 cumplida |
| Registro del modelo en DagsHub | `log_model` no dejó un artefacto cargable. Tras `publicar_modelo`, las tres redes se recargan y las probabilidades coinciden (D17) |

**No hecho en esta sesión:** la prueba de reanudación de `modulo2_guia_colab.md` §6.3 (parar a mitad y volver a lanzar). La lógica está cubierta por `tests/test_training.py` con una desconexión simulada; en Colab queda por ver una sola vez, y no bloquea el PR.

### Qué falta del paso 2

1. ~~Prueba `sandbox` y proyección de D10~~ (hechas el 7 nov).
2. ~~PR #8, aprobado y fusionado el 7 nov.~~
3. Opcional: la prueba de reanudación en Colab (§6.3 de la guía). No bloquea los runs reales.

## Runs reales — las 9 ejecuciones (7 nov 2026)

Experimento `m2-entrenamiento`. Modo `real`, `main`, `deterministic=True`. Semilla 42 Alex, 123 Adrián, 2024 Christian. El `val_macro_f1` y la sensibilidad son los de la línea `[+]`: el modelo guardado, que es el mejor de la fase 2. La selección oficial, con el resto de métricas, está en D18.

| Run | run_id | Tiempo | Épocas f1 / f2 | Mejor `val_loss` f1 (época de la fase) | Mejor `val_loss` f2 | `val_macro_f1` del modelo guardado | `val_mel_sensitivity` |
|---|---|---|---|---|---|---|---|
| `m2-resnet50-s42` | `b07eb7d9e4dc4526a31f9b7868255c16` | 14,2 min | 6 / 6 | 0,825 (1) | 1,102 (1) | 0,412 | 0,042 |
| `m2-densenet121-s42` | `3031ddc150fd485b89a34b1be65d06f2` | 17,2 min | 10 / 6 | 0,804 (5) | 0,931 (1) | 0,446 | 0,084 |
| `m2-efficientnetb0-s42` | `41ab358735a348f8b89617fe31f75c77` | 9,7 min | 9 / 6 | 0,842 (4) | 1,050 (1) | 0,358 | 0,006 |
| `m2-resnet50-s123` | `52713b074e9d426c85978644990825c0` | 13,4 min | 6 / 6 | 0,803 (1) | 0,951 (1) | 0,476 | 0,090 |
| `m2-densenet121-s123` | `c3db3c3ba1c745e18fef8ebe489a4daa` | 16,0 min | 9 / 6 | 0,819 (4) | 0,933 (1) | 0,431 | 0,127 |
| `m2-efficientnetb0-s123` | `dfe71994b6bc4a3a8fa07a589631b966` | 7,5 min | 6 / 6 | 0,854 (1) | 0,895 (1) | 0,337 | 0,030 |
| `m2-resnet50-s2024` | `03b89fa431db439188600861def72f31` | 14,2 min | 6 / 6 | 0,813 (1) | 0,933 (1) | 0,462 | 0,114 |
| `m2-densenet121-s2024` | `c6de3f9f46e04f8e973958881f7ffda0` | 15,5 min | 10 / 6 | 0,849 (5) | 0,938 (1) | 0,449 | 0,151 |
| `m2-efficientnetb0-s2024` | `84fab734e2a841a78460ab45d486e5d1` | 7,6 min | 8 / 6 | 0,810 (3) | 1,010 (1) | 0,348 | 0,012 |

Las épocas salen del log: la fase para cuando `val_loss` no mejora en 5 épocas (D13).

Media ± desviación estándar (n−1) del modelo guardado, con los tres decimales de la línea `[+]`:

| Arquitectura | `val_macro_f1` | `val_mel_sensitivity` |
|---|---|---|
| ResNet50 | 0,450 ± 0,034 | 0,082 ± 0,037 |
| DenseNet121 | 0,442 ± 0,010 | 0,121 ± 0,034 |
| EfficientNetB0 | 0,348 ± 0,011 | 0,016 ± 0,012 |

**Lectura.**

- La parada anticipada funcionó en los nueve. Ninguna fase agotó sus 15 o 30 épocas. Cada cuenta tardó entre 37 y 41 min, por debajo de los 83 min de peor caso de D10.
- En los nueve runs, el mejor `val_loss` de la fase 2 es el de su primera época, y es peor que el mejor de la fase 1. Al descongelar, la pérdida de entrenamiento baja y la de validación sube, y no se recupera dentro de la paciencia. El modelo guardado es ese por D13. El protocolo no se reescribe a posteriori: los nueve ya se entrenaron con la misma regla. `f1_mejor_val_loss` y `f2_mejor_val_loss` quedan en cada run para la memoria.
- La recarga del modelo quedó comprobada en las tres semillas (7 nov): `load_model("runs:/<id>/modelo")` da salida `(None, 7)` y las mismas probabilidades del entrenamiento.
- La selección con las métricas completas está en D18. Esta tabla, redondeada desde la consola, no es la decisión.

### D18. Cómo se aplica la regla de selección

**Decisión.** `src/selection.py` lee los 9 runs con `estado=final` y nombre `m2-<arch>-s<semilla>`. La media y la desviación estándar (n−1) se calculan por arquitectura. Un candidato entra si la media de sensibilidad en melanoma es ≥ 0,80 y la media de la sensibilidad mínima por clase es ≥ 0,50. Entre quienes entran, gana el mayor macro-F1. Si la diferencia de esa métrica con la líder es menor que la mayor de las dos desviaciones estándar entre semillas, se considera ruido y desempata el balanced accuracy. Si nadie entra, la misma regla se aplica sobre la sensibilidad en melanoma. La semilla representativa de cada red es la de macro-F1 más cercano a la media de sus tres semillas; esa versión recibe el alias `final`, y la de la ganadora también `seleccionado`. El resultado queda en un run `m2-seleccion`, con artefactos `seleccion.json` y `seleccion.md`.

**Razón.** Es la regla de `docs/01_objetivos.md`, escrita una sola vez para que la memoria y el registro de modelos salgan del mismo cálculo. Hacerlo a mano con los tres decimales de la consola redondea el desempate. No se repite: si `m2-seleccion` ya está en `final`, `lanzar_seleccion` se niega a crear otro.

**Resultado (7 nov 2026, run `cb7d4966bb7e4d58974c02420ed631d7`).** Ninguna arquitectura cumple los dos requisitos. Gana **DenseNet121** por sensibilidad en melanoma (0,120 ± 0,034), por delante de ResNet50 (0,082 ± 0,037) y EfficientNetB0 (0,016 ± 0,013). No hizo falta desempate. En ResNet50 la sensibilidad mínima por clase coincide con la de melanoma: es su peor clase. En DenseNet121 y EfficientNetB0 hay alguna clase todavía por debajo (mínima 0,115 y 0,006). Semillas representativas, alias `final`: ResNet50 semilla 2024, DenseNet121 semilla 42, EfficientNetB0 semilla 2024. DenseNet121 lleva además `seleccionado`. Las tres quedaron registradas como versión 1. Para la memoria hay que explicar por qué nadie llega al 80 % de sensibilidad en melanoma. Esos runs y esos alias se conservan. El segundo experimento, con el aumento igualado, es D19.

### D19. Aumento uniforme y experimento `m2-entrenamiento-v2`

**Causa.** En la v1 el aumento dependía de la clase: nv solo recibía volteos horizontal y vertical; las otras seis recibían además rotación ±90° con relleno por reflejo, acercamiento del 90–100 % y brillo y contraste de ×0,9 a ×1,1. Como nv es el 67 % del train, la red podía usar «esta imagen no está rotada ni cambiada de brillo» como pista de nv, sin mirar la lesión.

**Evidencia (7 nov 2026, validación, etiquetas sin cambiar).** Sobre los tres modelos guardados de las semillas representativas, forzar el aumento completo en todas las imágenes (media de las semillas 11, 22 y 33) baja la fracción predicha nv y sube la sensibilidad en melanoma. Los volteos solos no mueven la fracción. El conjunto de prueba no se abrió.

| Modelo guardado | Fracción predicha nv, limpia | Con aumento forzado | Sensibilidad en melanoma, limpia | Con aumento forzado |
|---|---|---|---|---|
| ResNet50, semilla 2024 (`03b89fa431db439188600861def72f31`) | 0,842 | 0,361 | 0,114 | 0,604 |
| DenseNet121, semilla 42 (`3031ddc150fd485b89a34b1be65d06f2`) | 0,841 | 0,390 | 0,084 | 0,576 |
| EfficientNetB0, semilla 2024 (`84fab734e2a841a78460ab45d486e5d1`) | 0,909 | 0,334 | 0,012 | 0,556 |

La diferencia de fracción nv es 0,45–0,58. El ruido entre las tres semillas de aumento es del orden de 0,03, y los volteos solos cambian la fracción en menos de 0,01. La fase 1 de ResNet50 s2024 y de DenseNet121 s42 ya tenía el mismo hueco, así que no es un efecto de la parada de la fase 2. Con el aumento a favor, la sensibilidad en melanoma se queda en 0,56–0,60: igualar el aumento es el experimento que toca, y no promete el umbral clínico de 0,80.

**Cambio.** Un solo cambio de protocolo. `augmentar` aplica `_fuerte` a las siete clases; `etiqueta` sigue en la firma porque el pipeline la pasa y no cambia la transformación. `configs/base.yaml` pasa a `config_version: 2` y `data.augmentation.politica: uniforme`. Se mantienen `split_v1`, `class_weights.json` (raíz cuadrada), la cabeza, las dos fases, las tasas, el 30 % de capas, la BatchNormalization congelada, la parada por `val_loss` sin pesos con paciencia 5, el modo determinista, el lote de 32 y 224 × 224.

Los runs nuevos van a `m2-entrenamiento-v2`, con el mismo nombre `m2-<arch>-s<semilla>`, tags `aumento=uniforme`, huella SHA-256 de `augmentation.py` y de `base.yaml`, `gpu` y `tf_version`. `estado=final` se pone al terminar sin errores. Los checkpoints van a `checkpoints-v2/`, incluido `f1_mejor.weights.h5`. Cada época registra `f1_val_sens_mel` y `f2_val_sens_mel`. Al cerrar el run se registran `val_frac_nv_limpio`, `val_frac_nv_aumentado`, `val_sens_mel_limpio`, `val_sens_mel_aumentado` y `val_delta_frac_nv`.

**Alarma.** `alarma_delta_frac_nv: 0,20`. Salta si la fracción predicha nv sin aumento menos la media con aumento forzado supera 0,20. Ese corte está por debajo de la mitad del hueco más pequeño de la v1 (0,45) y muy por encima del ruido de las semillas de aumento (≈ 0,03). Si salta, el run sigue en `estado=final` y el tag `alarma_atajo=true`; el piloto no pasa y no se lanzan los otros ocho. `m2-entrenamiento` y `m2-seleccion` no se tocan. En el momento de esta decisión los alias `final` y `seleccionado` seguían en la v1; D20 los mueve. El conjunto de prueba sigue cerrado.

### D20. Selección v2 y umbral de melanoma

**Resultado (7 nov 2026, run `905dd96f099c4a968e33f3247318081c`, experimento `m2-entrenamiento-v2`).** Los nueve runs de la v2 están en `estado=final` y `alarma_atajo=false`. El delta de fracción nv queda entre −0,021 y +0,011, dentro del corte 0,20 de D19. Ninguna arquitectura cumple a la vez sensibilidad en melanoma ≥ 0,80 y sensibilidad ≥ 0,50 en todas las clases. Gana **EfficientNetB0** por sensibilidad en melanoma (0,498 ± 0,049), por delante de ResNet50 (0,436 ± 0,055) y DenseNet121 (0,384 ± 0,030). La diferencia con ResNet50 es 0,062, mayor que la desviación más grande de las dos (0,055), así que no hay desempate por balanced accuracy. Macro-F1 de la ganadora: 0,644 ± 0,008.

Semilla representativa de las tres redes: **2024**. Alias `final` en la versión 2 de cada modelo registrado. EfficientNetB0 lleva además `seleccionado`. La versión 1, la de la v1, se queda en el registro sin alias.

| Arquitectura | Run | `umbral_mel` |
|---|---|---|
| ResNet50 | `c009467e38f04b22a0208f6732c58e1a` | 0,127295 |
| DenseNet121 | `a16c233b18b043f49a84e5afed0cf3ab` | 0,083365 |
| EfficientNetB0 | `de11d3148353403ba1c091e6b66f0abf` | 0,198781 |

El corte exacto, el que de verdad deja la sensibilidad en 0,801, es 0,12729512155056, 0,08336453884840012 y 0,1987808346748352. Redondear a seis decimales sube el de DenseNet121 y el de EfficientNetB0, y la sensibilidad baja a 0,795. Antes de abrir la prueba se anota ese corte exacto como `umbral_mel_preciso`. No es otro umbral.

**Umbral.** `umbral_mel` es el valor más alto de p(mel) con el que ese modelo, en validación, llega a sensibilidad de melanoma ≥ 0,80. Si p(mel) ≥ t se predice melanoma; si no, se mantiene el argmax. En los tres, ese punto deja la sensibilidad mínima por clase por debajo de 0,50 (df en ResNet50, akiec en las otras dos) y predice melanoma en el 27–33 % de la validación. El umbral no cambia la ganadora. La selección oficial sigue siendo el argmax. El parámetro queda en el run para que la evaluación en prueba pueda repetir los dos criterios. Esa evaluación no se ha lanzado.

**Fase 2.** En los nueve runs de la v2, el mejor `val_loss` de la fase 2 es menor que el mejor de la fase 1, y el macro-F1 del modelo guardado es mayor. La sensibilidad en melanoma sube en ocho y se queda igual (0,386) en ResNet50 semilla 42. El modelo guardado sigue siendo el mejor de la fase 2 por `val_loss` (D13). No se reentrena.

**Qué no se hace en esta decisión.** El conjunto de prueba sigue cerrado. El Módulo 3 carga `models:/ham10000-<arch>@final`, que ahora apunta a estas semillas 2024. La evaluación en prueba es D21.

### D21. Evaluación en prueba, una sola vez

**Resultado (7 nov 2026, experimento `m2-evaluacion-test`).** Una pasada por las 1503 imágenes de prueba, 168 de ellas melanoma, sobre la versión 2 de cada alias `final` (semilla 2024). El umbral aplicado es `umbral_mel_preciso`. Tag `test_evaluado=true` y `estado=final`. No se repite. No se cambia el modelo, el umbral ni la selección después de ver estas cifras.

| Arquitectura | Run de prueba | Modelo | Umbral | macro-F1 argmax | Sens. mel argmax | macro-F1 umbral | Sens. mel umbral | Mínima |
|---|---|---|---|---|---|---|---|---|
| ResNet50 | `3b89d4d64f6f46d390d74d257a31aa33` | `c009467e38f04b22a0208f6732c58e1a` | 0,127295 | 0,613 | 0,524 | 0,600 | 0,815 | df 0,353 |
| DenseNet121 | `364bcd927a4b46d4acf897452b611fe7` | `a16c233b18b043f49a84e5afed0cf3ab` | 0,083365 | 0,532 | 0,262 | 0,520 | 0,833 | mel 0,262 / df 0,294 con umbral |
| EfficientNetB0 | `fd8db8ac7a3140e294820740608294ed` | `de11d3148353403ba1c091e6b66f0abf` | 0,198781 | 0,638 | 0,506 | 0,623 | 0,726 | df 0,412 |

Con argmax, EfficientNetB0 sigue por delante en macro-F1 (0,638) y en balanced accuracy (0,640). Su sensibilidad en melanoma en prueba es 0,506. El umbral fijado en validación sube esa sensibilidad a 0,815, 0,833 y 0,726. EfficientNetB0 no llega a 0,80 en prueba. Ninguna red llega a 0,50 en todas las clases. La clase mínima con argmax es df, mel y df.

Falsos negativos de melanoma con argmax, artefacto `falsos_negativos_mel.csv` de cada run. De 168 melanomas:

| Arquitectura | FN | A dónde van | Recuperados con el umbral |
|---|---|---|---|
| ResNet50 | 80 | nv 44, bkl 24, akiec 8, vasc 2, df 1, bcc 1 | 49 |
| DenseNet121 | 124 | bkl 53, nv 51, akiec 8, df 6, bcc 3, vasc 3 | 96 |
| EfficientNetB0 | 83 | nv 39, bkl 35, bcc 4, akiec 2, df 2, vasc 1 | 37 |

Artefactos de cada run: `confusion_test_norm.png`, `confusion_test_umbral_norm.png`, `predicciones_test.csv`, `falsos_negativos_mel.csv`. El Módulo 3 parte de estos falsos negativos y del alias `final`, que sigue siendo el argmax.
