"""Pruebas de src/models.py. Usan weights=None (sin descargar ImageNet) y CPU: son estructurales.

    pytest tests/test_models.py
Las pruebas que dependen de los pesos reales de ImageNet se hacen en Colab (notebook m2_00).
"""
import numpy as np
import pytest
import yaml

keras = pytest.importorskip("keras")

from src.models import (ARQUITECTURAS, bn_entrenables, build_model, capas_base, congelar_base,
                        descongelar_fraccion)

with open("configs/base.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


@pytest.mark.parametrize("arch", ARQUITECTURAS)
def test_estructura_y_salida(arch):
    m = build_model(arch, weights=None)
    assert m.output_shape == (None, 7)
    x = np.random.default_rng(0).random((2, 224, 224, 3)).astype("float32") * 255
    p = m.predict(x, verbose=0)
    assert p.shape == (2, 7) and np.allclose(p.sum(1), 1.0, atol=1e-5)
    assert [l.name for l in m.layers if l.name.startswith("head_")] == ["head_gap", "head_dropout", "head_softmax"]


@pytest.mark.parametrize("arch", ARQUITECTURAS)
def test_capa_gradcam_existe_con_el_nombre_de_base_yaml(arch):
    """El M3 busca estas capas por nombre (configs/base.yaml: gradcam.last_conv_layer)."""
    m = build_model(arch, weights=None)
    capa = m.get_layer(CFG["gradcam"]["last_conv_layer"][arch])
    assert len(capa.output.shape) == 4


@pytest.mark.parametrize("arch", ARQUITECTURAS)
def test_fase1_solo_entrena_la_cabeza(arch):
    m = build_model(arch, weights=None)
    r = congelar_base(m)
    assert not any(l.trainable for l in capas_base(m))
    con_pesos_entrenables = {l.name for l in m.layers if l.trainable_weights}
    assert con_pesos_entrenables == {"head_softmax"}
    n_features = m.get_layer("head_gap").output.shape[-1]          # 2048 / 1024 / 1280
    assert r["params_entrenables"] == n_features * 7 + 7            # solo kernel + bias de la Dense


@pytest.mark.parametrize("arch", ARQUITECTURAS)
def test_fase2_descongela_la_fraccion_superior_y_deja_bn_congeladas(arch):
    m = build_model(arch, weights=None)
    congelar_base(m)
    f1 = congelar_base(m)["params_entrenables"]
    r = descongelar_fraccion(m, CFG["training"]["phase2"]["unfreeze_fraction"],
                             CFG["training"]["phase2"]["freeze_batchnorm"])
    assert r["params_entrenables"] > f1
    assert bn_entrenables(m) == 0
    base = capas_base(m)
    # Las capas iniciales siguen congeladas, y alguna de las finales con pesos está entrenable.
    assert not any(l.trainable for l in base[: r["n_capas_base"] - r["n_capas_descongeladas"]])
    assert any(l.trainable and l.weights for l in base[-r["n_capas_descongeladas"]:])


@pytest.mark.parametrize("arch", ARQUITECTURAS)
def test_fase2_al_100_descongela_la_base_salvo_bn(arch):
    m = build_model(arch, weights=None)
    congelar_base(m)
    r = descongelar_fraccion(m, 1.0, congelar_bn=True)
    base = capas_base(m)
    assert r["n_capas_descongeladas"] == len(base)
    assert bn_entrenables(m) == 0
    for capa in base:
        if isinstance(capa, keras.layers.BatchNormalization):
            assert capa.trainable is False
        else:
            assert capa.trainable is True


def test_la_configuracion_v3_solo_cambia_la_fraccion():
    from src.tracking import load_config
    base = load_config("configs/base.yaml")
    v3 = load_config("configs/m2_v3.yaml")
    assert base["training"]["phase2"]["unfreeze_fraction"] == 0.3
    assert v3["training"]["phase2"]["unfreeze_fraction"] == 1.0
    assert v3["training"]["phase2"]["freeze_batchnorm"] is True
    assert v3["data"]["augmentation"] == base["data"]["augmentation"]
    assert v3["mlflow"]["experiments"]["m2_train_v2"] == "m2-entrenamiento-v2"
    assert v3["mlflow"]["experiments"]["m2_train_v3"] == "m2-entrenamiento-v3"
    assert v3["paths"]["checkpoints_v2_dir"] == base["paths"]["checkpoints_v2_dir"]


def test_bn_no_actualiza_estadisticas_al_entrenar():
    """Una BN con trainable=False debe usar y conservar sus estadísticas móviles al hacer fit."""
    m = build_model("resnet50", weights=None)
    descongelar_fraccion(m, 0.3, congelar_bn=True)
    m.compile(optimizer=keras.optimizers.Adam(1e-4), loss="categorical_crossentropy")
    bn = next(l for l in capas_base(m) if isinstance(l, keras.layers.BatchNormalization) and l.name.startswith("conv5"))
    antes = [w.numpy().copy() for w in bn.non_trainable_weights]
    x = np.random.default_rng(0).random((8, 224, 224, 3)).astype("float32") * 255
    y = keras.utils.to_categorical(np.arange(8) % 7, 7)
    m.fit(x, y, batch_size=4, epochs=1, verbose=0)
    for a, w in zip(antes, bn.non_trainable_weights):
        assert np.allclose(a, w.numpy())
