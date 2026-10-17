# Objetivos del TFM

## Pregunta que responde el trabajo

¿Cuál de las arquitecturas ResNet50, DenseNet121 y EfficientNetB0, entrenadas bajo condiciones idénticas sobre HAM10000, detecta mejor el melanoma sin comprometer el resto de clases, y sus decisiones se apoyan en la lesión o en elementos ajenos a ella?

## Objetivo general

Identificar, mediante una comparativa empírica bajo condiciones experimentales idénticas, cuál de las arquitecturas ResNet50, DenseNet121 y EfficientNetB0 obtiene el mayor F1-score macro en la clasificación de las 7 clases de HAM10000 entre las que alcancen una sensibilidad en melanoma de al menos el 80%, sin comprometer el rendimiento en las demás clases, y valorar mediante Grad-CAM si sus decisiones se apoyan en la lesión y no en elementos ajenos a ella, analizando todos los falsos negativos de melanoma del conjunto de prueba.

## Objetivos específicos

| OE | Módulo | Objetivo | Entregable verificable |
|---|---|---|---|
| OE1 | M1 | Preparar HAM10000: EDA de las 7 clases, estrategia frente al desbalance (ponderación de clases y aumento selectivo solo en entrenamiento), partición estratificada y agrupada por `lesion_id` (~70/15/15, ninguna lesión compartida) y pipeline `tf.data` idéntico para las tres arquitecturas. | `split_v1.csv` + hash, `class_weights.json`, función `make_dataset()`, test de no fuga |
| OE2 | M2 | Fine-tuning de las tres arquitecturas preentrenadas en ImageNet en dos fases (base congelada → descongelado progresivo con LR reducida), mismos hiperparámetros base, **tres semillas por arquitectura**, curvas de ambas fases registradas. | 9 runs en MLflow (3 arquitecturas × 3 semillas) con curvas y modelos |
| OE3 | M2 | Evaluar con métricas por clase (uno contra el resto) resumidas por promedio macro, balanced accuracy y matriz de confusión normalizada; seleccionar en **validación** (media de 3 semillas) el modelo de mayor F1 macro entre los que tengan sensibilidad en melanoma ≥ 80% y ≥ 50% en cada clase; usar **prueba una sola vez** para el resultado final. | Tabla comparativa, modelo seleccionado en el registro, métricas finales de prueba |
| OE4 | M3 | Identificar y analizar todos los falsos negativos de melanoma de cada modelo en el conjunto de prueba. | Lista de FN por modelo con clase predicha |
| OE5 | M3 | Implementar Grad-CAM sobre los tres modelos y generar mapas para las 7 clases, en aciertos y errores. | Mapas guardados como artefactos asociados a cada modelo |
| OE6 | M3 | Contrastar las activaciones con los criterios A, B y C de la regla ABCDE en los FN de melanoma y detectar atajos espurios (reglas, vello, tinta, bordes del dermatoscopio), contando por modelo cuántos casos se apoyan en elementos ajenos a la lesión. | Tabla de anotación por caso y recuento por modelo |
| OE7 | Todos | Pipeline reproducible en Colab, código en GitHub y trazabilidad en MLflow: el 100% de entrenamientos y evaluaciones finales registrados con versión del dataset, hiperparámetros, métricas y modelo. | Checklist de reproducibilidad cumplido (ver `03_mlflow_convenciones.md`) |

## Regla de selección del modelo (resumen)

1. Se calcula todo en **validación**, con la **media de las 3 semillas** de cada arquitectura.
2. **Requisito 1:** sensibilidad en melanoma ≥ 80%.
3. **Requisito 2:** sensibilidad ≥ 50% en cada una de las 7 clases.
4. Entre las arquitecturas que cumplen ambos requisitos, se elige la de **mayor F1-score macro**.
5. **Desempate:** si la diferencia de F1 macro entre dos arquitecturas es menor que la variación entre semillas (desviación estándar), se elige la de mayor balanced accuracy.
6. **Si ninguna cumple ambos requisitos:** se elige la de mayor sensibilidad en melanoma y se analizan las causas en la discusión.
7. Después de seleccionar, se evalúa **una sola vez** sobre prueba para reportar el rendimiento final.

## Métricas

| Tipo | Métricas |
|---|---|
| Selección | F1-score macro, sensibilidad en melanoma, sensibilidad mínima por clase |
| Melanoma | Sensibilidad, precisión, F1, área bajo la curva precisión-recall |
| Resumen multiclase | Balanced accuracy, AUC-ROC macro one-vs-rest |
| Por clase (uno contra el resto) | Sensibilidad, precisión, F1, especificidad, AUC-ROC, área bajo la curva precisión-recall |
| Diagnóstico de errores | Matriz de confusión normalizada por filas |
| Complementaria (no se usa para elegir) | Accuracy, especificidad por clase |

## Alcance

**Dentro:** 7 clases de HAM10000, una partición fija común, 3 arquitecturas × 3 semillas, Grad-CAM, análisis de FN de melanoma con criterios A, B y C.

**Opcional, si hay tiempo:** intervalos de confianza por bootstrap sobre prueba (remuestreo por lesión) y prueba de significancia pareada entre modelos (McNemar o bootstrap).

**Fuera:** validación cruzada completa, búsqueda extensa de hiperparámetros, calibración y análisis de coste clínico, validación externa en otros datasets.

## Pendientes de decisión del grupo

- [ ] Llevar el mínimo del 50% por clase a la Fase 3 de la memoria (apartado 3.3).
- [ ] Decidir si se incorpora Grad-CAM++ además de Grad-CAM.
- [ ] Dejar explícito que los Vision Transformers quedan fuera del alcance.
- [ ] Confirmar si el bootstrap es obligatorio u opcional (la memoria lo presenta ahora como obligatorio).
