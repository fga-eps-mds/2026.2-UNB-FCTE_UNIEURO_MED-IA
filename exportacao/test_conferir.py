import pytest

from exportacao.conferir import TOLERANCIA, conferir_escores
from exportacao.exportar import exportar_onnx
from exportacao.imagens import amostras_aleatorias


def test_modelo_exportado_da_os_mesmos_escores(modelo, lado, tmp_path):
    caminho = exportar_onnx(modelo, tmp_path / "modelo.onnx", lado)

    conferencia = conferir_escores(modelo, caminho, amostras_aleatorias(5, lado))

    assert conferencia.amostras == 5
    assert conferencia.tolerancia == TOLERANCIA
    assert conferencia.maior_diferenca <= TOLERANCIA
    assert conferencia.aprovada


def test_reprova_arquivo_exportado_de_outros_pesos(criar_modelo, lado, tmp_path):
    caminho = exportar_onnx(criar_modelo(semente=1), tmp_path / "modelo.onnx", lado)

    conferencia = conferir_escores(criar_modelo(), caminho, amostras_aleatorias(5, lado))

    assert conferencia.maior_diferenca > TOLERANCIA
    assert not conferencia.aprovada


def test_respeita_a_tolerancia_informada(criar_modelo, lado, tmp_path):
    caminho = exportar_onnx(criar_modelo(semente=1), tmp_path / "modelo.onnx", lado)

    conferencia = conferir_escores(
        criar_modelo(), caminho, amostras_aleatorias(5, lado), tolerancia=1.0
    )

    assert conferencia.aprovada


def test_recusa_conjunto_de_conferencia_vazio(modelo, lado, tmp_path):
    caminho = exportar_onnx(modelo, tmp_path / "modelo.onnx", lado)

    with pytest.raises(ValueError, match="vazio"):
        conferir_escores(modelo, caminho, [])
