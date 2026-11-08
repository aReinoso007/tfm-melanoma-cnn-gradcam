"""La regla de selección, sin MLflow ni GPU."""
import pytest

from src.selection import decidir, resumir

SEMILLAS = (42, 123, 2024)


def _filas(valores):
    """valores[arch] = lista de 3 dicts de métricas, en el orden de las semillas."""
    filas = []
    for arch, tres in valores.items():
        for seed, m in zip(SEMILLAS, tres):
            filas.append({"arch": arch, "seed": seed, "run_id": f"{arch}-{seed}", **m})
    return filas


def _plano(f1, mel, minima, prec=0.5, bal=0.5):
    return {"macro_f1": f1, "mel_sensitivity": mel, "mel_precision": prec,
            "min_class_sensitivity": minima, "balanced_accuracy": bal}


def test_ninguna_cumple_y_elige_la_mayor_sensibilidad_en_melanoma():
    filas = _filas({
        "resnet50": [_plano(0.45, 0.08, 0.10)] * 3,
        "densenet121": [_plano(0.44, 0.12, 0.10)] * 3,
        "efficientnetb0": [_plano(0.35, 0.02, 0.10)] * 3,
    })
    decision = decidir(resumir(filas))
    assert decision["motivo"] == "ninguna_cumple_ambos_requisitos"
    assert decision["ganadora"] == "densenet121"
    assert decision["cumplen"] == []


def test_desempate_por_balanced_accuracy_cuando_el_f1_cae_dentro_del_ruido():
    # Medias de macro-F1 0.50 y 0.52, con desviación 0.10: la diferencia es ruido.
    # EfficientNet queda fuera del requisito de melanoma.
    filas = _filas({
        "resnet50": [_plano(0.40, 0.90, 0.60, bal=0.80),
                     _plano(0.50, 0.90, 0.60, bal=0.80),
                     _plano(0.60, 0.90, 0.60, bal=0.80)],
        "densenet121": [_plano(0.42, 0.90, 0.60, bal=0.70),
                        _plano(0.52, 0.90, 0.60, bal=0.70),
                        _plano(0.62, 0.90, 0.60, bal=0.70)],
        "efficientnetb0": [_plano(0.90, 0.10, 0.90, bal=0.95)] * 3,
    })
    decision = decidir(resumir(filas))
    assert decision["cumplen"] == ["resnet50", "densenet121"]
    assert decision["ganadora"] == "resnet50"
    assert "desempate" in decision["como"]


def test_f1_claro_no_usa_el_desempate():
    filas = _filas({
        "resnet50": [_plano(0.70, 0.90, 0.60, bal=0.40)] * 3,
        "densenet121": [_plano(0.50, 0.90, 0.60, bal=0.90)] * 3,
        "efficientnetb0": [_plano(0.10, 0.10, 0.10)] * 3,
    })
    decision = decidir(resumir(filas))
    assert decision["ganadora"] == "resnet50"
    assert decision["como"] == "mayor macro_f1"


def test_semilla_representativa_es_la_mas_cercana_a_la_media():
    # macro-F1 0.40, 0.50, 0.60 → media 0.50 → semilla 123.
    filas = _filas({
        "resnet50": [_plano(0.40, 0.1, 0.1), _plano(0.50, 0.1, 0.1), _plano(0.60, 0.1, 0.1)],
        "densenet121": [_plano(0.2, 0.1, 0.1)] * 3,
        "efficientnetb0": [_plano(0.2, 0.1, 0.1)] * 3,
    })
    resumen = resumir(filas)
    assert resumen["resnet50"]["semilla_representativa"] == 123
    assert resumen["resnet50"]["run_id_representativo"] == "resnet50-123"


def test_umbral_elige_el_t_mas_alto_que_llega_a_0_80():
    from src.metrics import CLASES, barrer_umbral_mel
    # Tres imágenes: una melanoma con p=0.9, otra con p=0.2, y un nv con p=0.01.
    # Con t=0.9 solo se rescata 1 de 2 melanomas (sens 0.5). Con t=0.2 se rescatan los dos.
    y = [CLASES.index("mel"), CLASES.index("mel"), CLASES.index("nv")]
    probs = [
        [0.02, 0.02, 0.02, 0.02, 0.90, 0.01, 0.01],
        [0.10, 0.10, 0.10, 0.10, 0.20, 0.30, 0.10],
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.94, 0.01],
    ]
    r = barrer_umbral_mel(y, probs, objetivo=0.80)
    assert r["alcanza"] is True
    assert r["t"] == pytest.approx(0.20)
    assert r["mel_sensitivity"] == pytest.approx(1.0)
    assert r["cumple_min_clase"] is False


def test_los_falsos_negativos_de_melanoma_marcan_el_rescate():
    import pandas as pd
    from src.evaluation import falsos_negativos_mel
    tabla = pd.DataFrame({
        "image_id": ["a", "b", "c"],
        "lesion_id": ["l1", "l2", "l3"],
        "y_true": ["mel", "mel", "nv"],
        "y_pred": ["nv", "mel", "mel"],
        "p_mel": [0.4, 0.8, 0.7],
        "y_pred_umbral": ["mel", "mel", "mel"],
    })
    fn = falsos_negativos_mel(tabla)
    assert list(fn.image_id) == ["a"]
    assert fn.iloc[0].y_pred == "nv"
    assert bool(fn.iloc[0].recuperado_umbral) is True


def test_la_evaluacion_en_prueba_no_se_repite():
    from types import SimpleNamespace
    from src.evaluation import evaluacion_ya_hecha

    def run(nombre, etiqueta):
        return SimpleNamespace(info=SimpleNamespace(run_name=nombre),
                               data=SimpleNamespace(tags={"test_evaluado": etiqueta}))

    assert evaluacion_ya_hecha([run("m2-test-resnet50", "true")], "resnet50") is True
    assert evaluacion_ya_hecha([run("m2-test-resnet50", "false")], "resnet50") is False
    assert evaluacion_ya_hecha([run("m2-test-densenet121", "true")], "resnet50") is False


def test_rechaza_un_conjunto_que_no_es_de_nueve():
    with pytest.raises(ValueError, match="exactamente 9"):
        resumir(_filas({"resnet50": [_plano(0.1, 0.1, 0.1)] * 3,
                        "densenet121": [_plano(0.1, 0.1, 0.1)] * 3}))
