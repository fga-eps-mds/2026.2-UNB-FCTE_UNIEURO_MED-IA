import pytest
import torch
from PIL import Image

from exportacao import gerar
from exportacao.imagens import ARQUIVOS, amostras_aleatorias


@pytest.fixture
def modelo_pequeno_como_treinado(monkeypatch, modelo, lado):
    monkeypatch.setattr(gerar, "carregar_modelo", lambda caminho: modelo)
    monkeypatch.setattr(gerar, "LADO", lado)


def test_mede_parametros_tamanho_escore_e_tempo(modelo, lado, tmp_path):
    resultado = gerar.gerar(modelo, tmp_path / "modelo.onnx", amostras_aleatorias(3, lado), lado)

    assert resultado.parametros == sum(parametro.numel() for parametro in modelo.parameters())
    assert resultado.tamanho_mb == (tmp_path / "modelo.onnx").stat().st_size / 1e6
    assert resultado.conferencia.amostras == 3
    assert resultado.conferencia.aprovada
    assert resultado.segundos_por_avaliacao > 0


def test_main_exporta_e_confere_com_amostras_aleatorias(
    modelo_pequeno_como_treinado, capsys, tmp_path
):
    destino = tmp_path / "modelo.onnx"

    gerar.main(["model.pt", str(destino), "--quantidade", "2"])

    saida = capsys.readouterr().out
    assert destino.exists()
    assert "Maior diferença de escore em 2 amostras" in saida
    assert "Dentro da tolerância de 1e-04: sim" in saida


def test_main_confere_com_os_desenhos_da_pasta(modelo_pequeno_como_treinado, capsys, tmp_path):
    for paciente in ("a", "b", "c"):
        (tmp_path / "imagens" / paciente).mkdir(parents=True)
        for arquivo in ARQUIVOS:
            Image.new("L", (8, 8), 200).save(tmp_path / "imagens" / paciente / arquivo)

    gerar.main(["model.pt", str(tmp_path / "modelo.onnx"), "--imagens", str(tmp_path / "imagens")])

    assert "Maior diferença de escore em 3 amostras" in capsys.readouterr().out
