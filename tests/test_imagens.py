import pytest
import torch
from PIL import Image

from exportacao.imagens import ARQUIVOS, amostras_aleatorias, amostras_da_pasta, ler_desenho


@pytest.fixture
def pasta_de_pacientes(tmp_path):
    for paciente, cinza in (("b", 255), ("a", 0), ("c", 51)):
        (tmp_path / paciente).mkdir()
        for arquivo in ARQUIVOS:
            Image.new("L", (8, 8), cinza).save(tmp_path / paciente / arquivo)
    (tmp_path / "leia-me.txt").write_text("não é paciente")
    return tmp_path


def test_le_o_desenho_em_rgb_de_zero_a_um_no_tamanho_do_treino(pasta_de_pacientes):
    desenho = ler_desenho(pasta_de_pacientes / "c" / "clock.png", lado=4)

    assert desenho.shape == (1, 3, 4, 4)
    assert desenho.dtype == torch.float32
    assert torch.allclose(desenho, torch.full((1, 3, 4, 4), 0.2))


def test_le_os_primeiros_pacientes_em_ordem_de_nome(pasta_de_pacientes):
    amostras = amostras_da_pasta(pasta_de_pacientes, quantidade=2, lado=4)

    assert len(amostras) == 2
    assert [len(amostra) for amostra in amostras] == [3, 3]
    assert [amostra[0].max().item() for amostra in amostras] == [0.0, 1.0]


def test_amostras_aleatorias_se_repetem_com_a_mesma_semente():
    primeira = amostras_aleatorias(2, 4)
    segunda = amostras_aleatorias(2, 4)
    outra_semente = amostras_aleatorias(2, 4, semente=1)

    assert all(a.equal(b) for x, y in zip(primeira, segunda) for a, b in zip(x, y))
    assert not primeira[0][0].equal(outra_semente[0][0])
