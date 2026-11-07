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


def test_rechaza_un_conjunto_que_no_es_de_nueve():
    with pytest.raises(ValueError, match="exactamente 9"):
        resumir(_filas({"resnet50": [_plano(0.1, 0.1, 0.1)] * 3,
                        "densenet121": [_plano(0.1, 0.1, 0.1)] * 3}))
