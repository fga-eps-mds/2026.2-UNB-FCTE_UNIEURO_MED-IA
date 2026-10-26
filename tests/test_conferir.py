import pytest

from exportacao.conferir import TOLERANCIA, conferir_escores
from exportacao.exportar import exportar_onnx
from exportacao.imagens import amostras_aleatorias


@pytest.mark.parametrize(
    ("semente_exportada", "tolerancia", "aprovada"),
    [(0, TOLERANCIA, True), (1, TOLERANCIA, False), (1, 1.0, True)],
    ids=["mesmos_pesos", "outros_pesos", "tolerancia_informada"],
)
def test_aprova_quando_a_diferenca_cabe_na_tolerancia(
    criar_modelo, lado, tmp_path, semente_exportada, tolerancia, aprovada
):
    caminho = exportar_onnx(criar_modelo(semente=semente_exportada), tmp_path / "modelo.onnx", lado)

    conferencia = conferir_escores(
        criar_modelo(), caminho, amostras_aleatorias(5, lado), tolerancia=tolerancia
    )

    assert conferencia.amostras == 5
    assert conferencia.tolerancia == tolerancia
    assert conferencia.aprovada is aprovada


def test_recusa_conjunto_de_conferencia_vazio(modelo, lado, tmp_path):
    caminho = exportar_onnx(modelo, tmp_path / "modelo.onnx", lado)

    with pytest.raises(ValueError, match="vazio"):
        conferir_escores(modelo, caminho, [])
