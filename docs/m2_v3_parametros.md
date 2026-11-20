# Parámetros de las tres redes

`model.count_params()` con la cabeza de 7 clases, sin depender de los pesos de ImageNet. Coincide con las cifras calculadas a mano para la memoria.

| Arquitectura | Parámetros totales | Entrenables en la fase 2 de la v2 (30 %) | Entrenables en la fase 2 de la v3 (100 %, BN congelada) |
|---|---|---|---|
| ResNet50 | 23 602 055 | 17 199 111 | 23 495 815 |
| DenseNet121 | 7 044 679 | 2 862 087 | 6 877 383 |
| EfficientNetB0 | 4 058 538 | 3 063 587 | 3 974 499 |

La diferencia entre el total y los entrenables de la v3 son las BatchNormalization, que siguen congeladas, y sus estadísticas. En la v2 el resto de la base también está congelado.
