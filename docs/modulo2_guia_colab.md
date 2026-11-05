# Módulo 2 — Guía para ejecutar en Google Colab

Esta guía cubre el **paso 1** (modelos y métricas), con el notebook [`notebooks/m2_00_verificacion_colab.ipynb`](../notebooks/m2_00_verificacion_colab.ipynb) (ya ejecutado y cerrado), y el **paso 2** (entrenamiento), con [`notebooks/m2_01_entrenamiento.ipynb`](../notebooks/m2_01_entrenamiento.ipynb): ver la sección 6 al final. Para los límites de Colab gratuito, los secretos y el reparto de GPU, ver [04_colab.md](04_colab.md); aquí solo está lo específico de M2.

## Qué comprueba y por qué hace falta

En local ya pasan las 22 pruebas de `tests/`, pero con pesos aleatorios y en CPU. Hay cosas que solo se ven en Colab y que es mejor descubrir **antes** de gastar GPU en los 9 runs:

| Comprobación | Por qué importa |
|---|---|
| Las pruebas pasan con Python 3.13 y las versiones de Colab | El código se probó en Python 3.12; Colab usa 3.13. |
| Las tres redes cargan los pesos de ImageNet y aceptan los lotes de `make_dataset()` | Es el contrato M1 → M2: si no encaja, hay que saberlo ya. |
| Los parámetros entrenables coinciden con los medidos en local | Confirma que el congelado hace lo mismo en Colab. |
| `set_seeds(42, deterministic=True)` no falla en la GPU | `src/tracking.py` avisa de que algunas operaciones de GPU no tienen versión determinista. Si falla, hay que decidirlo antes del primer run. |
| Una fase 1 y una fase 2 mínimas compilan y entrenan | Keras exige recompilar al cambiar `trainable`. |
| Secretos y MLflow/DagsHub funcionan (run en `sandbox`) | Si el token está mal, se descubre aquí y no tras horas de entrenamiento. |

Dura unos 20 minutos (el bloque 4b añade 5–8) (en la práctica: ~92 s en descomprimir el dataset, ~2 min en las pruebas y el resto en descargar pesos), con la GPU encendida todo el rato.

## 1. Preparación (una sola vez)

1. **Subir la rama a GitHub desde el portátil.** Colab clona desde GitHub, no desde tu disco. La rama `m2/modelos-metricas` solo existe en local:
   ```bash
   git push -u origin m2/modelos-metricas
   ```
   El push va a una rama propia, no a `main`, que sigue protegida.
2. **Drive:** la carpeta compartida `TFM` tiene que estar como acceso directo en *Mi unidad*, de modo que exista `/content/drive/MyDrive/TFM/data/HAM10000.zip` (`docs/04_colab.md` §2.1). Ese zip lo genera `m1_00_entorno.ipynb` y debe contener `HAM10000_metadata.csv` y `images/`.
3. **Secretos de Colab** (icono de la llave): `DAGSHUB_USER` y `DAGSHUB_TOKEN`, con el interruptor *Acceso desde el notebook* activado **en este notebook**. El token de DagsHub debe tener permiso de escritura en el repo (*Settings → Collaborators*).

## 2. Abrir el notebook

Opción A (directa): abrir esta dirección en el navegador:

```
https://colab.research.google.com/github/aReinoso007/tfm-melanoma-cnn-gradcam/blob/m2/modelos-metricas/notebooks/m2_00_verificacion_colab.ipynb
```

Opción B: Colab → *Archivo → Abrir cuaderno → GitHub* → pegar `aReinoso007/tfm-melanoma-cnn-gradcam`, elegir la rama `m2/modelos-metricas` y el notebook.

Luego: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU*.

## 3. Ejecutar

*Entorno de ejecución → Ejecutar todo*, o celda a celda con Mayús+Intro. La primera vez Colab pide autorizar el acceso a Drive.

Qué debe verse en cada bloque:

| Bloque | Resultado esperado |
|---|---|
| 1. Configuración | `Tesla T4`, el último commit de la rama, `[+] 10015 imágenes, metadatos y split v1 verificados` |
| 2. Pruebas | `22 passed` |
| 3. Redes con ImageNet | Tabla con estos valores (si no coinciden, parar y avisar) |
| 4. Determinismo | `[+] fase 1: ...` y `[+] fase 2: ...` |
| 4b. Determinismo en las 3 redes | Una tabla con 6 filas (3 redes × `det` 0 y 1), sin `error`, con `repetible = True` en las filas `det = 1`, y al final `VEREDICTO: …`. Resultado obtenido el 4 nov: sin errores, las tres repetibles con `det = 1`, penalización de 1,37× / 1,62× / 1,90× (ResNet50 / DenseNet121 / EfficientNetB0), por lo que el veredicto automático fue «decidir en grupo» (ver D10 en el registro de decisiones) |
| 5. Métricas | Una fila de métricas y el encabezado `image_id, lesion_id, y_true, y_pred, p_akiec…` |
| 6. MLflow | `[+] Run creado en el experimento sandbox` |

Tabla esperada en el bloque 3 (medida en local con los mismos pesos):

| Arquitectura | Entrenables fase 1 | Entrenables fase 2 | Total | % fase 2 |
|---|---|---|---|---|
| resnet50 | 14 343 | 17 199 111 | 23 602 055 | 72,9 |
| densenet121 | 7 175 | 2 862 087 | 7 044 679 | 40,6 |
| efficientnetb0 | 8 967 | 3 063 587 | 4 058 538 | 75,5 |

El run aparece en DagsHub → *Experiments* → `sandbox` → `m2-verificacion-paso1`, con el tag `determinismo_gpu`. Los valores de las métricas **no significan nada** (el modelo no está entrenado y son solo 64 imágenes).

Al terminar: *Entorno de ejecución → Desconectar y eliminar el entorno de ejecución*.

## 4. Si algo falla

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| `error: pathspec 'm2/modelos-metricas' did not match` | La rama no está en GitHub | Hacer el `git push` del apartado 1 |
| `FileNotFoundError: .../TFM/data/HAM10000.zip` | Falta el acceso directo a `TFM` o el zip | Revisar `docs/04_colab.md` §2.1; si el zip no existe, ejecutar `m1_00_entorno.ipynb` (Adrián) |
| `SecretNotFoundError` o `NotebookAccessError` | Secreto sin crear o sin acceso para este notebook | Llave → crear el secreto y activar el interruptor del notebook |
| Colab pide reiniciar la sesión tras `pip install` | `requirements.txt` fija `scikit-learn==1.9.1` | Reiniciar y volver a ejecutar desde la celda del clon; no se pierde nada |
| `assert ... Metadatos distintos` o `El split del repo no es el v1` | Zip de otro origen o rama desactualizada | No continuar; avisar a Adrián. El hash del split y de los metadatos no se discute |
| Bloque 4 falla con un error de operación no determinista | Una operación de GPU no tiene versión determinista | Anotar el mensaje completo. Hay que decidir entre `deterministic=False` (registrado como parámetro del run) o cambiar la operación; es una decisión del grupo |
| Fallan pruebas que pasaban en local | Diferencia de versiones | Copiar la salida completa; no se entrena hasta resolverlo |
| MLflow `403` o `401` | Token sin permiso de escritura | Revisar *Collaborators* en DagsHub |

## 5. Después de esta verificación

Con el paso 1 en verde, se abre el PR de `m2/modelos-metricas` a `main`. Como toca `src/`, necesita la aprobación de los tres (`docs/02_flujo_de_trabajo.md` §3). Las dos decisiones que más conviene revisar están marcadas como "punto abierto" en [modulo2_registro_decisiones.md](modulo2_registro_decisiones.md) (D5 y D8).

El siguiente paso es el bucle de entrenamiento de las dos fases. Cuando exista, los runs reales se lanzarán desde un notebook basado en `00_plantilla_colab.ipynb`, con `BackupAndRestore` y `start_or_resume_run` para sobrevivir a desconexiones (`docs/04_colab.md` §5).

## 6. Paso 2 — Entrenamiento (`m2_01_entrenamiento.ipynb`)

Un solo notebook para la prueba y para los runs oficiales; el interruptor es `MODO` (celda de parámetros). Las razones de diseño están en D12–D16 del registro de decisiones.

### 6.1 Antes de abrirlo

1. Subir la rama desde el portátil: `git push -u origin m2/entrenamiento`.
2. Mismos requisitos que el paso 1: T4 GPU, acceso directo a `TFM` en Drive, secretos `DAGSHUB_USER` y `DAGSHUB_TOKEN` activados para el notebook nuevo (el interruptor *Acceso desde el notebook* es por notebook).
3. Abrir: `https://colab.research.google.com/github/aReinoso007/tfm-melanoma-cnn-gradcam/blob/m2/entrenamiento/notebooks/m2_01_entrenamiento.ipynb`. No usar "Guardar una copia en GitHub" (crea ramas y carpetas sueltas).

### 6.2 Prueba en `sandbox` (lo primero que hay que hacer)

Con los valores por defecto (`MODO = "sandbox"`, 3 redes, semilla 42, 2+2 épocas, datos completos): *Entorno de ejecución → Ejecutar todo*.

**Tiempo estimado:** el primer epoch de cada red incluye decodificar los 7 009 JPEG (≈ 1–2 min extra); luego salen de la caché. Orden de magnitud: de 10 a 25 minutos para las tres redes. Los runs aparecen en el experimento `sandbox` con nombre `m2-<arch>-s42-sandbox`.

| Qué mirar | Resultado esperado |
|---|---|
| Línea por época | `[f1] época 01  loss=…  val_loss=…  val_macro_f1=…  (… ms/paso)`; la fase 2 sigue con `[f2] época 03…` (numeración acumulada) |
| `[+] m2-<arch>-s42-sandbox …` | Sin errores en las tres redes, sobre todo ResNet50 y DenseNet121 con `deterministic=True` (riesgo residual de D10: hasta ahora solo EfficientNetB0 se había probado en un entrenamiento) |
| Sección 3, proyección de coste | Tabla con minutos de peor caso por red y `TOTAL por cuenta (3 redes)`. **Hay que copiar esa tabla**: confirma o refuta la condición de D10 (≤ ≈ 120 min por cuenta). Súmale un margen del 5–10 % por la validación de cada época |
| Sección 4, carga del modelo | `modelo cargado (None, 7) | mismas probabilidades que en el entrenamiento: True` en las tres |
| DagsHub → Experiments → `sandbox` | Curvas `f1_*` y `f2_*` en una sola gráfica; tags `modulo=m2`, `modo=sandbox`; parámetros `deterministic=True`, `gpu`, `tf_version`; artefactos `curvas.png`, `confusion_val_norm.png`, `classification_report_val.json`, `predicciones_val.csv`, y el modelo |

Los valores de `val_macro_f1` tras 2+2 épocas **no son resultados**: solo indican que algo se aprende.

### 6.3 Prueba de reanudación (recomendada, 5 minutos)

Comprueba que una desconexión no hace perder el avance:

1. Poner `ARCHS = ["efficientnetb0"]` y `SANDBOX_EPOCAS = (3, 3)`.
2. Ejecutar la celda de lanzar y **detenerla** (botón de parar) durante la fase 2, o después de ver `[f1] época 03`.
3. Poner `RETOMAR_SANDBOX = True` y volver a ejecutar esa misma celda de lanzar (sin reiniciar el notebook).
4. Debe aparecer `Retomando run…`, la fase 1 **no** se repite y la fase 2 sigue desde su última época. En DagsHub debe ser **un solo** run con las épocas sin duplicar.

Si el entorno se reinició de verdad (no solo se detuvo la celda), hay que volver a ejecutar las celdas de configuración antes de la de lanzar.

### 6.4 Runs reales (cuando el PR esté fusionado y la prueba, confirmada)

1. `BRANCH = "main"` (el código ya fusionado), `MODO = "real"`, `CONFIRMO_REAL = True`, `ARCHS` = las de tu cuenta y `SEED` según `docs/04_colab.md` §7.
2. El notebook se niega a arrancar con cambios sin commit y `lanzar_run` no acepta `epochs`/`max_lotes` en este modo.
3. Los runs quedan en `m2-entrenamiento` como `m2-<arch>-s<seed>`. Al terminar los 9, marcar a mano `estado = final` o `descartado`.

### 6.5 Si algo falla

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| `error: pathspec 'm2/entrenamiento' did not match` | Rama sin subir | `git push -u origin m2/entrenamiento` |
| `OOM` / `ResourceExhaustedError` en una red | Memoria de GPU o de RAM (caché de `tf.data`) | Copiar el mensaje completo; no cambiar `batch_size` por tu cuenta (es de `base.yaml` y afecta a las tres redes) |
| Error de operación no determinista en ResNet50 o DenseNet121 | Es el riesgo residual de D10 | Copiar el mensaje completo y avisar: la salida (`deterministic=False`) es del grupo |
| Error en `mlflow.tensorflow.log_model` contra DagsHub | DagsHub puede no aceptar el formato de modelo de MLflow 3 | Copiar el mensaje completo; la alternativa es registrar el modelo como artefacto y no como modelo de MLflow, decisión que afecta a M3 |
| Proyección de coste > 120 min por cuenta | Determinismo más caro de lo medido | No lanzar runs reales; llevarlo al grupo (D10) |
| `RuntimeError: Hay cambios sin commit` (modo real) | El árbol de Colab tiene cambios | Volver a clonar o `git checkout .`; los runs oficiales salen de código commiteado |
