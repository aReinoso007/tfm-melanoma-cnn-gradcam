"""Pruebas de src/explain.py. Usan weights=None y CPU (sin descargar ImageNet): son estructurales.

    pytest tests/test_explain.py
La prueba de cordura con los modelos entrenados reales se hace en Colab (notebooks/m3_01_gradcam.ipynb).
"""
import numpy as np
import pytest

keras = pytest.importorskip("keras")

from src.explain import (METODOS, Explicador, base_aleatorizada, correlacion_rangos,
                         n_capas_base_con_pesos, superponer)
from src.models import ARQUITECTURAS, build_model


def _imagen(seed=0):
    return np.random.default_rng(seed).random((224, 224, 3)).astype("float32") * 255


@pytest.fixture(scope="module", params=ARQUITECTURAS)
def caso(request):
    arch = request.param
    keras.utils.set_random_seed(0)
    modelo = build_model(arch, weights=None)
    return arch, modelo, Explicador(modelo, arch)


def test_logits_coinciden_con_la_softmax_del_modelo(caso):
    _, modelo, expl = caso
    x = _imagen()
    esperado = modelo.predict(x[None], verbose=0)[0]
    assert np.allclose(expl.probabilidades(x), esperado, atol=1e-4)


@pytest.mark.parametrize("metodo", METODOS)
def test_tamano_y_rango(caso, metodo):
    _, _, expl = caso
    x = _imagen()
    m = expl.mapa(x, int(np.argmax(expl.logits(x))), metodo)
    assert m.shape == (224, 224) and m.dtype == np.float32
    assert np.isfinite(m).all() and m.min() >= 0.0 and m.max() <= 1.0
    assert m.max() > 0.0, "mapa vacío: el gradiente no llega a la capa convolucional"


@pytest.mark.parametrize("metodo", METODOS)
def test_determinista(caso, metodo):
    _, _, expl = caso
    x = _imagen(1)
    c = int(np.argmax(expl.logits(x)))
    assert np.array_equal(expl.mapa(x, c, metodo), expl.mapa(x, c, metodo))


def test_el_mapa_cambia_con_la_clase(caso):
    _, _, expl = caso
    x = _imagen(2)
    primera, segunda = np.argsort(expl.logits(x))[::-1][:2]
    assert not np.allclose(expl.mapa(x, int(primera)), expl.mapa(x, int(segunda)))


def test_capa_inexistente_falla():
    modelo = build_model("resnet50", weights=None)
    with pytest.raises(ValueError):
        Explicador(modelo, "resnet50", capa="no_existe")


def test_correlacion_de_rangos():
    a = np.arange(100.0).reshape(10, 10)
    assert correlacion_rangos(a, a) == pytest.approx(1.0)
    assert correlacion_rangos(a, -a) == pytest.approx(-1.0)
    assert np.isnan(correlacion_rangos(a, np.zeros((10, 10))))


@pytest.mark.parametrize("arch", ARQUITECTURAS)
def test_cordura_adebayo_la_correlacion_cae_al_aleatorizar_la_base(arch):
    """Con ultimas=0 el modelo es idéntico (rho=1); con la base entera aleatorizada, rho cae."""
    keras.utils.set_random_seed(0)
    modelo = build_model(arch, weights=None)
    expl = Explicador(modelo, arch)
    imagenes = [_imagen(s) for s in range(4)]
    clases = [int(np.argmax(expl.logits(x))) for x in imagenes]   # clase con mapa no vacío
    for metodo in METODOS:
        orig = [expl.mapa(x, c, metodo) for x, c in zip(imagenes, clases)]
        assert all(o.max() > 0 for o in orig)
        with base_aleatorizada(modelo, 0, ultimas=0):
            r_id = np.mean([correlacion_rangos(o, expl.mapa(x, c, metodo)) for o, x, c in zip(orig, imagenes, clases)])
        with base_aleatorizada(modelo, 0):
            rs = [correlacion_rangos(o, expl.mapa(x, c, metodo)) for o, x, c in zip(orig, imagenes, clases)]
        # Un mapa aleatorizado vacío (constante) cuenta como mapa destruido: correlación 0.
        r_al = 0.0 if np.isnan(rs).all() else float(np.nanmean(rs))
        assert r_id == pytest.approx(1.0, abs=1e-4)
        # Umbral holgado: el "original" aquí es un modelo sin entrenar, sin estructura aprendida que perder.
        # La caída clara (rho ≪ 1) se comprueba con los modelos reales en el cuaderno de Colab.
        assert r_al < 0.9, f"{arch} {metodo}: rho={r_al:.2f} no cae tras aleatorizar la base"


@pytest.mark.parametrize("arch", ARQUITECTURAS)
def test_base_aleatorizada_restaura_los_pesos_y_no_toca_la_cabeza(arch):
    modelo = build_model(arch, weights=None)
    antes = [w.copy() for w in modelo.get_weights()]
    cabeza = modelo.get_layer("head_softmax").kernel.numpy().copy()
    capa = [l for l in modelo.layers if "kernel" in "".join(w.name for w in l.weights)
            and not l.name.startswith("head_")][-1]
    k_antes = capa.weights[0].numpy().copy()
    with base_aleatorizada(modelo, 3):
        assert not np.array_equal(capa.weights[0].numpy(), k_antes)
        assert np.array_equal(modelo.get_layer("head_softmax").kernel.numpy(), cabeza)
    assert all(np.array_equal(a, b) for a, b in zip(antes, modelo.get_weights()))
    assert n_capas_base_con_pesos(modelo) > 10


def test_superponer_devuelve_uint8():
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    out = superponer(img, np.random.default_rng(0).random((224, 224)))
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
