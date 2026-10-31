# tfm-melanoma-cnn-gradcam

Comparativa empírica de **ResNet50, DenseNet121 y EfficientNetB0** para la clasificación de las 7 clases de **HAM10000**, con análisis de explicabilidad mediante **Grad-CAM** centrado en los falsos negativos de melanoma.

Trabajo Fin de Máster (grupal) — Máster Universitario en Inteligencia Artificial, UNIR.
Integrantes: Adrián Arias (Módulo 1), Alex Reinoso (Módulo 2), Christian Zhirzhan (Módulo 3).
Director: Pablo Gimeno Jordan.

## Documentación

| Documento | Contenido |
|---|---|
| [docs/01_objetivos.md](docs/01_objetivos.md) | Objetivo general, objetivos específicos, criterios de éxito y alcance |
| [docs/02_flujo_de_trabajo.md](docs/02_flujo_de_trabajo.md) | Git, ramas, revisiones, Colab, DagsHub y credenciales |
| [docs/03_mlflow_convenciones.md](docs/03_mlflow_convenciones.md) | Experimentos, nombres de runs, tags, parámetros, métricas, artefactos y registro de modelos |
| [docs/04_colab.md](docs/04_colab.md) | Cómo trabajar en Colab gratuito: Drive compartido, secretos, sesiones, checkpoints y reparto de GPU |
| [docs/modulo1_datos.md](docs/modulo1_datos.md) | Pasos del Módulo 1: EDA, desbalance, partición y pipeline `tf.data` |
| [docs/modulo2_entrenamiento_evaluacion.md](docs/modulo2_entrenamiento_evaluacion.md) | Pasos del Módulo 2: fine-tuning, selección y evaluación |
| [docs/modulo3_gradcam.md](docs/modulo3_gradcam.md) | Pasos del Módulo 3: Grad-CAM, falsos negativos y ABCDE |

## Estructura del repositorio

```
tfm-melanoma-cnn-gradcam/
├── configs/
│   └── base.yaml            # configuración común (rutas, semillas, hiperparámetros base)
├── data/
│   └── splits/              # CSV de la partición (sí se versiona; las imágenes NO)
├── docs/                    # objetivos, flujo de trabajo y guías por módulo
├── notebooks/               # un notebook por paso, con prefijo de módulo (m1_, m2_, m3_)
├── src/
│   ├── data.py              # Módulo 1: make_dataset(), split_df(), load_class_weights()
│   ├── augmentation.py      # Módulo 1: aumento selectivo (solo train)
│   ├── tfm_common.py        # modelos y métricas (M2); sus secciones 2-3 de datos están obsoletas
│   └── tracking.py          # utilidades comunes: MLflow, semillas, hashes, tags
├── requirements.txt
└── .gitignore
```

## Puesta en marcha (Google Colab)

Lo más cómodo es abrir [`notebooks/00_plantilla_colab.ipynb`](notebooks/00_plantilla_colab.ipynb), que hace todo esto (y más) en celdas listas. Detalle en [docs/04_colab.md](docs/04_colab.md). Resumen:

```python
# 1. Montar Drive (dataset y checkpoints)
from google.colab import drive
drive.mount('/content/drive')

# 2. Clonar el repo y entrar
!git clone https://github.com/aReinoso007/tfm-melanoma-cnn-gradcam.git
%cd tfm-melanoma-cnn-gradcam
!pip install -q -r requirements.txt

# 3. Credenciales de DagsHub desde los secretos de Colab (icono de llave)
import os
from google.colab import userdata
os.environ["MLFLOW_TRACKING_USERNAME"] = userdata.get("DAGSHUB_USER")
os.environ["MLFLOW_TRACKING_PASSWORD"] = userdata.get("DAGSHUB_TOKEN")

# 4. Configurar MLflow
from src.tracking import load_config, setup_mlflow
cfg = load_config("configs/base.yaml")
setup_mlflow(cfg)
```

## Entorno

Python 3.10 · TensorFlow 2.x / Keras · GPU T4 (Colab) · MLflow 3.x en DagsHub · scikit-learn · Matplotlib · Seaborn.

## Datos

HAM10000 (Tschandl et al., 2018), licencia CC BY-NC 4.0. Las imágenes **no se suben al repositorio**: se guardan en Google Drive. En el repo solo se versionan la partición (`data/splits/`) y su hash.
