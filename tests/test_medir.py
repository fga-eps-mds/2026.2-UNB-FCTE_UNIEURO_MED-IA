import pytest

from exportacao.conferir import abrir_sessao, escore_exportado
from exportacao.exportar import exportar_onnx
from exportacao.imagens import amostras_aleatorias
from exportacao.medir import medir_execucao


def test_calcula_escores_e_mede_tempo_e_memoria_da_sessao(modelo, lado, tmp_path):
    caminho = exportar_onnx(modelo, tmp_path / "modelo.onnx", lado)
    amostras = amostras_aleatorias(3, lado)

    medicao = medir_execucao(caminho, amostras)

    assert medicao.escores == tuple(escore_exportado(abrir_sessao(caminho), amostra) for amostra in amostras)
    assert medicao.segundos_por_avaliacao > 0
    assert medicao.memoria_mb >= 0


def test_recusa_conjunto_de_medicao_vazio(modelo, lado, tmp_path):
    caminho = exportar_onnx(modelo, tmp_path / "modelo.onnx", lado)

    with pytest.raises(ValueError, match="vazio"):
        medir_execucao(caminho, [])
