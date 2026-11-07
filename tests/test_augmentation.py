"""El aumento v2 es el mismo para las siete clases y no depende de la etiqueta."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.augmentation import CLASES, augmentar, preparar
from src.training import resumen_regresion_atajo


def _imagen():
    img = np.full((32, 32, 3), 40, np.float32)
    img[0, 0] = 255
    img[0, -1] = 0
    img[-1, 0] = 10
    return tf.constant(img)


def test_todas_las_clases_reciben_la_misma_transformacion():
    img = _imagen()
    seed = tf.constant([11, 4], tf.int64)
    salidas = [augmentar(img, i, seed).numpy() for i in range(len(CLASES))]
    for otra in salidas[1:]:
        np.testing.assert_allclose(salidas[0], otra)


def test_la_misma_semilla_repite_la_imagen():
    img = _imagen()
    seed = tf.constant([22, 1], tf.int64)
    a = augmentar(img, CLASES.index("nv"), seed)
    b = augmentar(img, CLASES.index("mel"), seed)
    np.testing.assert_allclose(a, b)


def test_en_evaluacion_la_imagen_no_cambia():
    img = _imagen()
    out = preparar(img, CLASES.index("nv"), tf.constant([1, 1], tf.int64), training=False)
    np.testing.assert_allclose(out, tf.cast(img, tf.float32))


def test_dentro_de_un_dataset():
    img = _imagen()
    ds = tf.data.Dataset.from_tensors((img, 5, tf.constant([11, 0], tf.int64)))
    ds = ds.map(lambda im, y, s: augmentar(im, y, s))
    out = next(iter(ds)).numpy()
    assert out.shape == (32, 32, 3)
    assert out.min() >= 0 and out.max() <= 255
    img = _imagen()
    out = augmentar(img, CLASES.index("nv"), tf.constant([11, 0], tf.int64)).numpy()
    assert out.min() >= 0 and out.max() <= 255
    assert not np.allclose(out, img.numpy())


def test_alarma_de_atajo():
    quieto = resumen_regresion_atajo(0.70, [0.68, 0.66, 0.67], 0.20, [0.22, 0.19, 0.21], 0.20)
    assert quieto["alarma_atajo"] is False
    v1 = resumen_regresion_atajo(0.84, [0.36, 0.39, 0.33], 0.08, [0.60, 0.58, 0.57], 0.20)
    assert v1["alarma_atajo"] is True
    assert v1["val_delta_frac_nv"] == pytest.approx(0.84 - np.mean([0.36, 0.39, 0.33]))
