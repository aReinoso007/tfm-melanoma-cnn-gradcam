# Aumento uniforme (apartado 4.1.6)

Desde la v2 las siete clases reciben la misma cadena. La etiqueta no cambia ninguna transformación. Solo se aplica en entrenamiento: con `training=False`, `preparar` devuelve la imagen tal cual, en `[0, 255]`.

No se elige una transformación al azar. Sobre cada imagen se aplican todas, en este orden, y cada una usa su propia semilla derivada de la semilla de la imagen:

1. **Volteo horizontal.** Si está activo, probabilidad 0,5 (`stateless_random_flip_left_right`).
2. **Volteo vertical.** Si está activo, probabilidad 0,5, independiente del anterior.
3. **Rotación.** Siempre. Ángulo uniforme en ±90° (`rotation_factor` 0,25 de una vuelta). Interpolación bilineal. Las esquinas se rellenan por reflejo (`REFLECT`), no con negro.
4. **Acercamiento.** Siempre. El lado del recorte es uniforme entre el 90 % y el 100 % del lado (`zoom_factor` 0,1). La posición del recorte es aleatoria y luego se devuelve el tamaño a 224 × 224.
5. **Brillo.** Siempre. Factor multiplicativo uniforme en [0,9; 1,1]. No se cambia el tono ni la saturación.
6. **Contraste.** Siempre. Factor uniforme en [0,9; 1,1] respecto a la media de cada canal.
7. **Recorte de rango.** Los píxeles se limitan a [0, 255].

Los dos volteos son los únicos pasos con probabilidad de no aplicarse. La rotación, el acercamiento, el brillo y el contraste se aplican siempre; lo aleatorio es su magnitud, que puede quedar cerca de cero. La misma semilla repite exactamente la misma imagen.
