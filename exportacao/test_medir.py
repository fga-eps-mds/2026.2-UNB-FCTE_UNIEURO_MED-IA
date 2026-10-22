import pytest

from exportacao.exportar import exportar_onnx
from exportacao.imagens import amostras_aleatorias
from exportacao.medir import medir_execucao


def test_mede_tempo_e_memoria_da_sessao(modelo, lado, tmp_path):
    caminho = exportar_onnx(modelo, tmp_path / "modelo.onnx", lado)

    medicao = medir_execucao(caminho, amostras_aleatorias(3, lado))

    assert medicao.segundos_por_avaliacao > 0
    assert medicao.memoria_mb >= 0


def test_recusa_conjunto_de_medicao_vazio(modelo, lado, tmp_path):
    caminho = exportar_onnx(modelo, tmp_path / "modelo.onnx", lado)

    with pytest.raises(ValueError, match="vazio"):
        medir_execucao(caminho, [])
