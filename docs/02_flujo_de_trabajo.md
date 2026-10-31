# Flujo de trabajo del grupo

## 1. Repositorio

- **Un único repositorio** para los tres módulos: `tfm-melanoma-cnn-gradcam`.
- GitHub es el origen del código y DagsHub está conectado a él (DagsHub → *Connect a repository* → GitHub). Así el código vive en GitHub y DagsHub aloja MLflow con el mismo nombre de repo.
- La rama `main` está protegida: no se hace push directo, todo entra por Pull Request.

## 2. Ramas

| Rama | Uso |
|---|---|
| `main` | Código estable y revisado |
| `m1/<tarea>` | Módulo 1 (p. ej. `m1/eda`, `m1/particion`) |
| `m2/<tarea>` | Módulo 2 (p. ej. `m2/fase1-congelada`, `m2/evaluacion`) |
| `m3/<tarea>` | Módulo 3 (p. ej. `m3/gradcam`, `m3/analisis-fn`) |
| `comun/<tarea>` | Cambios que afectan a todos (`src/tracking.py`, `configs/base.yaml`) |

## 3. Pull Requests

- Cada PR la revisa **al menos otro integrante**.
- Los cambios en `configs/base.yaml`, `src/` o `data/splits/` necesitan la aprobación de **los tres**, porque cambian las condiciones experimentales.
- En la descripción del PR se enlaza el run o experimento de MLflow si el cambio produjo resultados.

## 4. Commits

Formato: `[m1|m2|m3|comun] verbo en infinitivo + qué`

```
[m1] generar partición estratificada agrupada por lesion_id
[m2] registrar curvas de la fase 2 en MLflow
[comun] fijar tamaño de imagen 224 en base.yaml
```

## 5. Notebooks y código

- La **lógica va en `src/`** y los notebooks solo la importan y ejecutan. Así el mismo código sirve a los tres módulos y se puede revisar en los PR.
- Nombres de notebook: `m1_01_eda.ipynb`, `m2_02_entrenamiento.ipynb`, `m3_01_gradcam.ipynb`.
- Antes de hacer commit se limpian las salidas (`nbstripout`, ver abajo), para que los diffs sean legibles y no se suban imágenes pesadas. Los resultados quedan en MLflow, no en el notebook.

```bash
pip install nbstripout && nbstripout --install
```

## 6. Ejecución en Colab

1. Montar Drive, clonar el repo y hacer `pip install -r requirements.txt` (ver README).
2. Trabajar en la rama propia: `!git checkout -b m2/fase1-congelada`.
3. **Antes de lanzar un entrenamiento**, hacer commit y push del código. El commit queda registrado en MLflow (tag `git_commit`), y si hay cambios sin commit el run se marca como `git_dirty=true`. Los runs con `git_dirty=true` **no valen como resultados finales**.

## 7. Credenciales

- Cada integrante usa **su propio token** de DagsHub (*User Settings → Tokens*).
- El token se guarda en los **secretos de Colab** (`DAGSHUB_USER`, `DAGSHUB_TOKEN`) y **nunca** se escribe en un notebook ni se sube al repo.
- Los tres deben tener permiso de escritura en el repo de DagsHub (*Settings → Collaborators*).

## 8. Datos y artefactos pesados

| Qué | Dónde |
|---|---|
| Imágenes HAM10000 | Google Drive compartido (`/MyDrive/TFM/data/HAM10000/`) |
| Checkpoints intermedios | Google Drive (`/MyDrive/TFM/checkpoints/`) |
| Partición (`split_v1.csv`) y su hash | Repo (`data/splits/`) y artefacto en MLflow |
| Modelos finales | Registro de modelos de MLflow |
| Métricas, curvas, figuras, mapas Grad-CAM | Artefactos de MLflow |

## 9. Interfaces entre módulos (contratos)

Lo que cada módulo entrega al siguiente. Si un contrato cambia, se avisa al grupo y se pasa por PR común.

| De → A | Entrega | Formato |
|---|---|---|
| M1 → M2 | Partición fija | `data/splits/split_v1.csv` con columnas `image_id, lesion_id, dx, split` |
| M1 → M2 | Pesos de clase | `class_weights.json` (calculados solo con entrenamiento) |
| M1 → M2 | Carga de datos | `make_dataset(split, arch, training)` en `src/data.py` (aumento en `src/augmentation.py`) |
| M2 → M3 | Modelos finales | Registro MLflow: `ham10000-<arch>` con alias `final` |
| M2 → M3 | Predicciones de prueba | `predicciones_test.csv` con `image_id, lesion_id, y_true, y_pred, p_akiec … p_vasc` |

**Regla (acuerdo del grupo, octubre de 2026):** los Módulos 2 y 3 cargan los datos **solo** con el código del Módulo 1 y no lo reimplementan:

```python
from src.data import make_dataset, split_df, load_class_weights
train_ds = make_dataset("train", arch, training=True)     # aumento solo en train
val_ds   = make_dataset("val",   arch, training=False)
model.fit(train_ds, validation_data=val_ds, class_weight=load_class_weights())
# Orden de las predicciones de val/test: split_df("val") / split_df("test")
```

- No se regenera el split ni se recalculan los pesos.
- `make_dataset()` ya aplica la normalización oficial de cada red: el modelo no debe normalizar otra vez.
- Las secciones 2 y 3 de `src/tfm_common.py` (datos y pipeline) quedan obsoletas.

**Convenciones comunes fijadas:** semillas de entrenamiento `[42, 123, 2024]` (`configs/base.yaml`), MLflow 3.x (`requirements.txt`) y los nombres de experimentos de `docs/03_mlflow_convenciones.md`.

## 10. Reuniones

Dos reuniones semanales (Teams). Orden fijo:
1. Qué se terminó (con enlace al run o PR).
2. Qué bloquea a quién: sobre todo los contratos M1 → M2 → M3.
3. Decisiones que afectan a la memoria: se anotan y se trasladan al documento.
