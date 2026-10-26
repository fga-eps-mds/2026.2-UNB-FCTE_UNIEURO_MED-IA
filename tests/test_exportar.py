import onnx
import torch
from torch import nn

from exportacao.conferir import abrir_sessao
from exportacao.exportar import ENTRADAS, SAIDA, exportar_onnx


def test_grava_o_modelo_em_um_unico_arquivo(modelo, lado, tmp_path):
    caminho = exportar_onnx(modelo, tmp_path / "modelo.onnx", lado)

    assert [arquivo.name for arquivo in tmp_path.iterdir()] == ["modelo.onnx"]
    onnx.checker.check_model(str(caminho))


def test_entradas_seguem_a_ordem_relogio_cubo_trilhas(modelo, lado, tmp_path):
    sessao = abrir_sessao(exportar_onnx(modelo, tmp_path / "modelo.onnx", lado))

    assert [entrada.name for entrada in sessao.get_inputs()] == ["relogio", "cubo", "trilhas"]
    assert [entrada.shape for entrada in sessao.get_inputs()] == [[1, 3, lado, lado]] * 3


def test_saida_e_uma_probabilidade_por_avaliacao(modelo, lado, tmp_path):
    sessao = abrir_sessao(exportar_onnx(modelo, tmp_path / "modelo.onnx", lado))

    (saida,) = sessao.get_outputs()
    assert saida.name == SAIDA == "probabilidade_ccl"
    assert saida.shape == [1]


def test_exporta_em_modo_de_avaliacao(lado, tmp_path):
    class ComDropout(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(0.9)

        def forward(self, relogio, cubo, trilhas):
            return self.dropout(relogio + cubo + trilhas).mean(dim=(1, 2, 3))

    modelo = ComDropout().train()
    sessao = abrir_sessao(exportar_onnx(modelo, tmp_path / "modelo.onnx", lado))
    imagem = torch.ones(1, 3, lado, lado).numpy()

    escore = sessao.run(None, dict.fromkeys(ENTRADAS, imagem))[0][0]

    assert not modelo.training
    assert escore == 3.0
