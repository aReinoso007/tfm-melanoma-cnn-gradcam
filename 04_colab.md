# Trabajar desde Google Colab (versión gratuita)

## 1. Límites de la versión gratuita que hay que tener en cuenta

| Límite | Qué implica para nosotros |
|---|---|
| La GPU (normalmente T4) **no está garantizada** y tiene una cuota de uso variable. Tras varias horas de GPU, Colab puede negarla durante un tiempo. | Repartir los entrenamientos entre las **tres cuentas** y no gastar GPU depurando. |
| La sesión se corta por **inactividad** (unos 90 min sin interacción) y tiene una **duración máxima** (unas 12 h, a menudo menos). | Guardar checkpoints en Drive en cada época y poder **retomar** el entrenamiento. |
| El disco de la sesión (`/content`) se **borra** al desconectar. | Todo lo que importa va a Drive (checkpoints), a GitHub (código) o a MLflow (resultados). |
| Leer miles de imágenes pequeñas desde Drive es **muy lento**. | Copiar el dataset comprimido a `/content` al inicio de cada sesión. |
| ~12 GB de RAM. | Cargar las imágenes con `tf.data` desde disco y no meterlas todas en memoria con NumPy. |

Las cifras son aproximadas: Google no las publica y cambian según la demanda.

## 2. Preparación (una sola vez)

### 2.1 Drive compartido
1. Adrián crea en su Drive la carpeta `TFM/` y la comparte con Alex y Christian como **Editor**.
2. Cada uno, en *Compartido conmigo*, hace clic derecho en `TFM` → **Organizar → Añadir acceso directo → Mi unidad**.
   Así la ruta `/content/drive/MyDrive/TFM` es **la misma para los tres** y `base.yaml` funciona sin cambios.
3. Dentro de `TFM/`:
   ```
   TFM/
   ├── data/HAM10000.zip      # imágenes + metadatos en un único zip
   └── checkpoints/           # una subcarpeta por run
   ```

### 2.2 Secretos de Colab (cada integrante en su cuenta)
En Colab, icono de la **llave** (barra lateral) → *Añadir secreto*:

| Nombre | Valor |
|---|---|
| `DAGSHUB_USER` | Usuario de DagsHub |
| `DAGSHUB_TOKEN` | Token de DagsHub (*User Settings → Tokens*) |
| `GITHUB_TOKEN` | Token de GitHub *fine-grained* con permiso **Contents: Read and write** solo para este repo |
| `GIT_NAME` / `GIT_EMAIL` | Nombre y correo para los commits |

Los secretos se guardan en la cuenta, no en el notebook: al compartir el notebook, el token no viaja con él. En cada notebook nuevo hay que activar el acceso al secreto (interruptor junto al nombre).

## 3. Rutina de cada sesión

Abrir `notebooks/00_plantilla_colab.ipynb` (o un notebook que empiece con sus celdas):

1. **Entorno de ejecución → Cambiar tipo** → T4 GPU (solo para entrenar; para escribir y depurar código, CPU).
2. Ejecutar las celdas de configuración. Hacen lo siguiente:
   - comprueban la GPU;
   - montan Drive;
   - clonan o actualizan el repo y se cambian a tu rama;
   - instalan dependencias;
   - copian y descomprimen el dataset en `/content` (~1 min);
   - conectan MLflow con DagsHub.
3. Trabajar.
4. Al terminar: **commit y push** desde el propio notebook (última celda) y **desconectar el entorno** (*Entorno de ejecución → Desconectar y eliminar*) para no consumir cuota.

## 4. Dónde se edita el código

- **Lógica (`src/`)**: se puede editar en Colab con el editor de archivos (barra lateral → carpeta → doble clic en el `.py`) o en local con VS Code. Después de editar un módulo ya importado, ejecutar `%load_ext autoreload` y `%autoreload 2` para no tener que reiniciar la sesión.
- **Notebooks (`notebooks/`)**: se guardan en el repo con *Archivo → Guardar una copia en GitHub*, o se descargan y se copian a la carpeta del repo clonado antes del commit.
- Regla de siempre: **commit y push antes de entrenar**. Si no, el run queda con `git_dirty=true` y no vale como resultado final.

## 5. Entrenamientos que sobreviven a desconexiones

- **`tf.keras.callbacks.BackupAndRestore(backup_dir)`** guarda el estado al final de cada época en Drive. Si Colab se desconecta, al volver a lanzar la misma celda el entrenamiento **continúa desde la última época**, no desde cero.
- **El run de MLflow también se retoma:** se guarda el `run_id` en un archivo de Drive y, al reanudar, se hace `mlflow.start_run(run_id=...)` en lugar de crear uno nuevo. Así no quedan runs duplicados ni a medias.
- Ruta de backup por run: `/MyDrive/TFM/checkpoints/m2-<arch>-s<seed>/`.

## 6. Ahorrar GPU

- **Depurar en CPU** con 50–100 imágenes y 1–2 épocas en el experimento `sandbox`. La GPU solo se usa para los runs reales.
- **Precisión mixta** en T4 (acelera bastante el entrenamiento):
  ```python
  from tensorflow.keras import mixed_precision
  mixed_precision.set_global_policy("mixed_float16")
  # la capa final debe ir en float32: Dense(7, activation="softmax", dtype="float32")
  ```
  Si se usa, se usa **en las tres arquitecturas** y se registra como parámetro.
- `tf.data` con `cache()` y `prefetch(AUTOTUNE)` para que la GPU no espere a los datos.
- Desconectar el entorno en cuanto termine el entrenamiento.

## 7. Reparto de los 9 runs del Módulo 2

Como las condiciones son idénticas (mismo código, configuración y partición, comprobado con `git_commit` y `split_sha256`), **no importa desde qué cuenta se lance cada run**. Para repartir la cuota:

| Cuenta | Runs |
|---|---|
| Alex | `m2-resnet50-s42`, `m2-densenet121-s42`, `m2-efficientnetb0-s42` |
| Adrián | `m2-resnet50-s123`, `m2-densenet121-s123`, `m2-efficientnetb0-s123` |
| Christian | `m2-resnet50-s2024`, `m2-densenet121-s2024`, `m2-efficientnetb0-s2024` |

El responsable del Módulo 2 sigue siendo Alex: él prepara el código y los demás solo lo ejecutan con su semilla. El tag `autor` recoge quién lanzó cada run.

## 8. Plan B si la GPU de Colab se agota

**Kaggle Notebooks** también ofrece GPU gratuita con una cuota semanal fija de horas (consultar la cifra actual). Se puede usar con el mismo repo y la misma configuración de MLflow. Si se usa, se registra el tag `plataforma=kaggle` y se comprueba que las versiones de TensorFlow coinciden.
