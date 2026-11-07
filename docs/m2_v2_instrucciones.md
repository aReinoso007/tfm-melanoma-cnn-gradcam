# M2 v2 — instrucciones para Adrián y Christian

Experimento `m2-entrenamiento-v2`. Aumento igual para las siete clases. Los runs de `m2-entrenamiento` y `m2-seleccion` se quedan como están. El conjunto de prueba sigue cerrado.

## Los dos

1. Esperar el aviso de Alex de que el piloto pasó (DenseNet121, semilla 42, `estado=final` y sin alarma de atajo). Si el piloto no pasa, no se lanza nada.
2. Abrir en Colab, con vuestra cuenta, el cuaderno de la etiqueta `m2-v2`:
   `notebooks/m2_04_entrenamiento_v2.ipynb`.
   Elegir una GPU T4 y activar los secretos `DAGSHUB_USER` y `DAGSHUB_TOKEN`.
3. En la celda de parámetros, poner la semilla y dejar las tres arquitecturas:
   - Adrián: `AUTOR = "adrian"`, `SEED = 123`
   - Christian: `AUTOR = "christian"`, `SEED = 2024`
4. Ejecutar todas las celdas. Son unas dos horas. Si Colab se desconecta, volver a ejecutar el cuaderno: las arquitecturas ya terminadas no se repiten.
5. No cambiar código ni configuración. Si una comprobación de huella falla, parar y avisar a Alex.
6. No evaluar en el conjunto de prueba y no mover alias en MLflow.

## Adrián, además (memoria, apartado 4.1.6)

En el cuaderno, poner `REVISION_NV = True` y ejecutar la última sección. Genera 5 semillas (42, 1, 2, 3 y 100) × 9 variantes de nv = 45 variantes, con el aumento completo, y guarda la figura en Drive.

Revisarlas con los mismos cuatro criterios de la revisión original: deja de parecer la misma lesión, corta el borde, cambia el color de forma visible, o muestra esquinas raras por la rotación.

En la memoria, actualizar:

- Tabla 13, columna «Clases»: «Todas».
- El párrafo que dice que los nevus solo reciben volteos.
- En «Implementación y comprobaciones», la propiedad de que en nv la imagen aumentada contiene los mismos píxeles.
- Las cifras de la revisión visual: ahora 315 variantes con aumento completo. Las 45 de nv de esta pasada sustituyen a las 45 que antes solo tenían volteos.

## Christian, además

Nada en la memoria por ahora. El Módulo 3 espera a `m2-seleccion-v2` y a la evaluación en prueba. Las semillas representativas pueden cambiar respecto a la v1. No preparar nada sobre los modelos v1.

## Qué pasarle a Alex, en un solo mensaje, al terminar

- El resumen que imprime la última celda de entrenamiento, para las tres arquitecturas: run ID, F1 macro, sensibilidad en melanoma, AUC de melanoma, fracción nv sin aumento y con aumento, minutos de GPU y estado.
- Confirmación de que los tres runs aparecen en `m2-entrenamiento-v2` con `estado=final`.
- Si se activó la alarma de regresión, cualquier error o desconexión, y si hubo que reanudar.
- Adrián, además: la figura de las 45 variantes de nv, cuántas tienen algún problema y de qué tipo, y el texto actualizado del 4.1.6 (o el aviso de que ya está en la memoria).
