# Módulo 2 — Notas para redactar la memoria (capítulo 4, sección M2)

**Para qué sirve.** Reúne, ya ordenado, lo que hay que contar de M2 en la memoria: los textos que se pueden adaptar, las cifras con su procedencia y lo que todavía falta por rellenar. Las razones completas de cada decisión están en [modulo2_registro_decisiones.md](modulo2_registro_decisiones.md) (D1–D11); aquí solo se cita el número de decisión.

**Cómo usarlo.**
- Los bloques marcados **[TEXTO ADAPTABLE]** son borradores en tono de memoria: se ajustan al estilo del documento, no se pegan tal cual.
- Los marcados **[PENDIENTE]** se rellenan con resultados reales cuando existan los 9 runs. **Ninguna cifra de métricas de este documento es un resultado**: todavía no hay entrenamientos reales.
- Toda cifra de la memoria debe poder rastrearse a un run de MLflow (`docs/03_mlflow_convenciones.md` §8). Aquí se indica de dónde sale cada una.

---

## 1. Limitaciones que hay que mencionar

### 1.1 La fracción de descongelado no iguala los parámetros entrenados entre arquitecturas (D5)

**Decisión del grupo (4 nov 2026):** se mantiene el 30 % de las capas y se declara como limitación.

**Qué ocurre.** El protocolo de la fase 2 es el mismo para las tres redes (descongelar el 30 % final de las capas de la base, con la BatchNormalization congelada). Pero el número de capas y el reparto de parámetros entre ellas es distinto en cada arquitectura, así que el mismo 30 % de capas supone cantidades de parámetros muy distintas:

| Arquitectura | Capas de la base | Capas descongeladas (30 %) | Parámetros entrenables en fase 2 | % del total de parámetros |
|---|---|---|---|---|
| ResNet50 | 174 | 52 | 17 199 111 | 72,9 % |
| DenseNet121 | 426 | 128 | 2 862 087 | 40,6 % |
| EfficientNetB0 | 237 | 71 | 3 063 587 | 75,5 % |

*Procedencia:* medido con los pesos de ImageNet, en local y en Colab (T4) con el mismo resultado; ver `docs/modulo2_registro_decisiones.md` D5 y el bloque 3 del notebook `notebooks/m2_00_verificacion_colab.ipynb`. Se puede volver a obtener con `descongelar_fraccion()` de `src/models.py`.

**Por qué se mantiene.** La regla única por número de capas es simple, reproducible y cumple "mismos hiperparámetros base" (OE2). Igualar parámetros obligaría a elegir una fracción distinta por red, es decir, un hiperparámetro específico de cada arquitectura que habría que ajustar con validación; eso queda fuera del alcance acordado (búsqueda extensa de hiperparámetros, `docs/01_objetivos.md`).

**Qué implica para interpretar los resultados.** Una parte de las diferencias entre arquitecturas puede deberse a cuánta capacidad se adapta en la fase 2 y no solo a la arquitectura. No se hizo un ablation para separar ambos efectos. Lo que sí controla el diseño: la fase 1, el optimizador, las tasas de aprendizaje, el tamaño de lote, la regla de parada y los datos son idénticos para las tres, y cada arquitectura se entrena con tres semillas, de modo que la variación entre semillas acota el ruido de cada resultado.

**[TEXTO ADAPTABLE]**
> En la segunda fase del ajuste fino se descongeló, en las tres arquitecturas, el 30 % final de las capas de la base, manteniendo congeladas las capas de normalización por lotes. Esta regla común no equivale a entrenar la misma proporción de parámetros: en DenseNet121 se actualizó el 40,6 % de los pesos del modelo, frente al 72,9 % en ResNet50 y el 75,5 % en EfficientNetB0, debido a que las últimas capas de estas dos redes concentran más parámetros. Se optó por una única regla por número de capas por ser la más simple de reproducir y por mantener idénticos los hiperparámetros base entre arquitecturas. Como consecuencia, parte de las diferencias observadas entre modelos podría atribuirse a la distinta capacidad de adaptación en esta fase y no únicamente a la arquitectura; este efecto no se aisló experimentalmente y debe tenerse en cuenta al interpretar la comparación.

### 1.2 Determinismo y su alcance (D10)

**Decisión (ratificada por el grupo):** `training.deterministic: true`, es decir, `tf.config.experimental.enable_op_determinism()` en los 9 entrenamientos.

**Qué contar.**
- Se midió en las tres redes, en una GPU T4: ninguna operación lanzó error y, con el determinismo activado, dos pasadas con la misma semilla dieron exactamente las mismas pérdidas y los mismos pesos. Sin él, DenseNet121 y EfficientNetB0 no se repitieron entre dos pasadas idénticas (diferencias de redondeo del orden de 10⁻⁷ en la pérdida tras 20 pasos).
- Coste medido: 1,37× (ResNet50), 1,62× (DenseNet121) y 1,90× (EfficientNetB0) más tiempo por paso.
- **Alcance:** la repetibilidad exacta es válida con el mismo hardware y las mismas versiones de software. Los entrenamientos se repartieron entre tres cuentas de Colab que podrían recibir GPUs distintas, por lo que se registra la GPU y la versión de TensorFlow en cada run y la reproducibilidad que se defiende es: semillas fijas + trazabilidad completa + media ± desviación estándar sobre tres semillas.
- *Procedencia:* bloque 4b de `m2_00_verificacion_colab.ipynb` y `scripts/bench_determinismo.py` (4 nov 2026); D10.

**[TEXTO ADAPTABLE]**
> Para favorecer la reproducibilidad se activó el modo determinista de TensorFlow en todos los entrenamientos. Una comprobación previa en las tres arquitecturas mostró que con este modo dos ejecuciones con la misma semilla producen resultados idénticos, mientras que sin él DenseNet121 y EfficientNetB0 presentaban diferencias de redondeo entre ejecuciones; el coste fue de entre 1,4 y 1,9 veces más tiempo por paso de entrenamiento. La repetibilidad exacta solo está garantizada con el mismo hardware y versiones de software, por lo que se registró en cada ejecución la GPU y la versión de TensorFlow, y los resultados se presentan como media y desviación estándar de tres semillas.

### 1.3 Otras decisiones que conviene citar como parte del protocolo

| Tema | Qué decir | Decisión |
|---|---|---|
| BatchNormalization congelada en la fase 2 | Con lotes de 32 y un dominio distinto de ImageNet, actualizar las estadísticas móviles degrada los pesos preentrenados. Verificado con una prueba que comprueba que no cambian durante `fit`. | D5 |
| Misma cabeza en las tres redes y sin normalización dentro del modelo | La normalización oficial de cada red se aplica en `make_dataset()`; el modelo no normaliza otra vez. | D3 |
| Métricas uno contra el resto, promedio macro | Cada clase pesa igual aunque tenga pocas imágenes (17 en prueba para `df`). `accuracy` se registra pero no se usa para elegir. | D7 |
| AUC y AP no calculables | Se devuelven como NaN y no como 0 cuando falta una clase (solo ocurre en submuestras). | D7 |
| Prueba evaluada una sola vez | Un run por arquitectura en `m2-evaluacion-test`. | `docs/01_objetivos.md` |
| Predicciones con `y_true`/`y_pred` por nombre de clase | Acordado con el Módulo 3. | D8 |

---

## 2. Tabla de hiperparámetros (para la memoria)

Fuente: `configs/base.yaml`, sección `training`, registrados en cada run. Los marcados con (*) eran "propuesta" y se confirman con la prueba en `sandbox`; si cambian, hay que actualizar esta tabla.

| Hiperparámetro | Valor |
|---|---|
| Tamaño de imagen | 224 × 224 |
| Optimizador | Adam |
| Tamaño de lote (*) | 32 |
| Dropout de la cabeza (*) | 0,3 |
| Fase 1 (base congelada): tasa de aprendizaje (*), épocas máximas | 1·10⁻³, 15 |
| Fase 2 (30 % final descongelado): tasa de aprendizaje (*), épocas máximas | 1·10⁻⁵, 30 |
| Parada anticipada | Sobre `val_loss`, paciencia 5, se conservan los pesos de la mejor época |
| Pesos de clase | Raíz cuadrada de la ponderación balanceada, normalizada a peso medio 1 (solo con entrenamiento) |
| Semillas | 42, 123, 2024 |
| Modo determinista | Sí (D10) |

---

## 3. Esqueleto de la sección M2 (capítulo 4)

Orden sugerido y de dónde sale cada elemento. Los elementos marcados **[PENDIENTE]** requieren los 9 runs.

1. **Configuración experimental.** Tabla de la sección 2 de este documento + descripción de las dos fases y de la cabeza (D3, D4, D5). ✔ listo para redactar.
2. **Limitaciones del protocolo.** Secciones 1.1 y 1.2. ✔ listo para redactar.
3. **Curvas de aprendizaje de las dos fases.** Figura `curvas.png` de cada run, una por arquitectura (semilla representativa). **[PENDIENTE]**
4. **Resultados en validación.** Tabla media ± desviación estándar por arquitectura de `val_macro_f1`, `val_mel_sensitivity`, `val_mel_precision`, `val_min_class_sensitivity` y `val_balanced_accuracy` (fuente: artefacto `seleccion.md`). **[PENDIENTE]**
5. **Aplicación de la regla de selección paso a paso.** Los cinco pasos de `docs/01_objetivos.md`, indicando en cuál se decidió. **[PENDIENTE]**
6. **Resultado final en prueba.** Tabla `test_*` por arquitectura, evaluada una sola vez; comprobar si el seleccionado mantiene sensibilidad en melanoma ≥ 80 %. **[PENDIENTE]**
7. **Matrices de confusión normalizadas.** `confusion_test_norm.png`; análisis de hacia qué clases se desvían los melanomas (conecta con el análisis de falsos negativos del Módulo 3). **[PENDIENTE]**
8. **Intervalos de confianza (opcional).** Bootstrap por `lesion_id`, 1 000 réplicas. **[PENDIENTE, si se hace]**

**Regla de oro al redactar:** las cifras de los puntos 3–7 se copian de los artefactos de los runs con `estado=final` y `git_dirty=false`; no se recalculan a mano ni se toman de la consola de un notebook.
