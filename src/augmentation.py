"""Aumento de datos del Módulo 1 (Etapa 6). Solo se aplica en entrenamiento.

Entrada: imagen float32 (H, W, 3) en [0, 255], ya recortada y redimensionada
(después de la caché del pipeline). Salida: misma forma y rango.

Todas las operaciones son "stateless": dado el mismo `seed` (tensor int64 de 2 valores)
la imagen aumentada es siempre la misma. El pipeline (Etapa 7) genera un seed distinto
por imagen y época, así que cada época ve variantes nuevas y el resultado es reproducible.

Dos intensidades (aumento selectivo):
  - fuerte (todas las clases salvo nv): volteos, rotación, zoom, brillo y contraste.
  - ligero (nv): solo volteos horizontal y vertical, que no pierden ni inventan píxeles.
"""
import math
import tensorflow as tf

CLASES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Parámetros por defecto = los de configs/base.yaml (data.augmentation).
# Revisión visual (315 variantes, 35 lesiones, revisor no dermatólogo): oscurecimiento leve en el 5,6 %
# sin pérdida de los rasgos de la lesión; el zoom toca la lesión en el 0,7 %. Se mantienen los valores.
CONFIG_POR_DEFECTO = {
    "apply_to": ["akiec", "bcc", "bkl", "df", "mel", "vasc"],   # intensidad fuerte
    "horizontal_flip": True,
    "vertical_flip": True,
    "rotation_factor": 0.25,    # fracción de vuelta: ±0,25 × 360° = ±90°. Con los dos volteos cubre los 360°
    "zoom_factor": 0.1,         # solo acercar: el lado visible pasa a ser 90–100 % (≈ 81–100 % del área)
    "brightness_delta": 0.1,    # brillo multiplicativo ×[0,9, 1,1]
    "contrast_factor": 0.1,     # contraste ×[0,9, 1,1] respecto a la media de cada canal
    "fill_mode": "REFLECT",     # relleno al rotar: reflejo (nunca negro: fabricaría un borde de dermatoscopio)
}

def _subseeds(seed, n):
    return tf.unstack(tf.random.experimental.stateless_split(tf.cast(seed, tf.int64), num=n))

def _voltear(img, seed, cfg):
    s1, s2 = _subseeds(seed, 2)
    if cfg["horizontal_flip"]:
        img = tf.image.stateless_random_flip_left_right(img, s1)
    if cfg["vertical_flip"]:
        img = tf.image.stateless_random_flip_up_down(img, s2)
    return img

def rotar(img, angulo, fill_mode="REFLECT"):
    """Rota `angulo` radianes alrededor del centro; las esquinas se rellenan según fill_mode."""
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    c, s = tf.cos(angulo), tf.sin(angulo)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    # Transformación inversa (salida -> entrada), formato de ImageProjectiveTransformV3
    t = tf.stack([c, -s, cx - c * cx + s * cy, s, c, cy - s * cx - c * cy, 0.0, 0.0])
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis], transforms=t[tf.newaxis], output_shape=tf.shape(img)[:2],
        fill_value=0.0, interpolation="BILINEAR", fill_mode=fill_mode)
    return out[0]

def _fuerte(img, seed, cfg):
    s_flip, s_rot, s_zoom, s_crop, s_bri, s_con = _subseeds(seed, 6)
    img = _voltear(img, s_flip, cfg)
    # Rotación continua
    if cfg["rotation_factor"] > 0:
        lim = cfg["rotation_factor"] * 2 * math.pi
        ang = tf.random.stateless_uniform([], s_rot, -lim, lim)
        img = rotar(img, ang, cfg["fill_mode"])
    # Zoom (solo acercar): recorte aleatorio de lado 90–100 % y vuelta al tamaño original
    if cfg["zoom_factor"] > 0:
        h, w = tf.shape(img)[0], tf.shape(img)[1]
        frac = tf.random.stateless_uniform([], s_zoom, 1.0 - cfg["zoom_factor"], 1.0)
        lado_h = tf.cast(tf.round(frac * tf.cast(h, tf.float32)), tf.int32)
        lado_w = tf.cast(tf.round(frac * tf.cast(w, tf.float32)), tf.int32)
        img = tf.image.stateless_random_crop(img, tf.stack([lado_h, lado_w, 3]), s_crop)
        img = tf.image.resize(img, tf.stack([h, w]), method="bilinear")
    # Brillo y contraste (no se toca tono ni saturación: el color es criterio diagnóstico)
    if cfg["brightness_delta"] > 0:
        d = cfg["brightness_delta"]
        img = img * tf.random.stateless_uniform([], s_bri, 1.0 - d, 1.0 + d)
    if cfg["contrast_factor"] > 0:
        f = cfg["contrast_factor"]
        img = tf.image.stateless_random_contrast(img, 1.0 - f, 1.0 + f, s_con)
    return tf.clip_by_value(img, 0.0, 255.0)

def _ligero(img, seed, cfg):
    return _voltear(img, seed, cfg)

def augmentar(img, etiqueta, seed, cfg=None):
    """Aumenta UNA imagen según su clase. `etiqueta` = índice de clase (orden de CLASES)."""
    cfg = {**CONFIG_POR_DEFECTO, **(cfg or {})}
    fuertes = tf.constant([CLASES.index(c) for c in cfg["apply_to"]], dtype=tf.int32)
    es_fuerte = tf.reduce_any(tf.equal(tf.cast(etiqueta, tf.int32), fuertes))
    img = tf.cast(img, tf.float32)
    return tf.cond(es_fuerte, lambda: _fuerte(img, seed, cfg), lambda: _ligero(img, seed, cfg))

def preparar(img, etiqueta, seed, training, cfg=None):
    """Punto de entrada del pipeline: aumenta solo si training=True; si no, devuelve la imagen tal cual."""
    if training:
        return augmentar(img, etiqueta, seed, cfg)
    return tf.cast(img, tf.float32)
