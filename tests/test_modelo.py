import pytest
import torch

from exportacao.imagens import amostras_aleatorias
from exportacao.modelo import LADO, VGG16, Extrator, ModeloBase, carregar_modelo


def test_devolve_uma_probabilidade_por_avaliacao(modelo, lado):
    relogio, cubo, trilhas = (torch.rand(2, 3, lado, lado) for _ in range(3))

    probabilidades = modelo(relogio, cubo, trilhas)

    assert probabilidades.shape == (2,)
    assert ((probabilidades >= 0) & (probabilidades <= 1)).all()


def test_cada_desenho_passa_por_uma_torre_propria(modelo, lado):
    (relogio, cubo, trilhas), = amostras_aleatorias(1, lado)

    assert modelo(relogio, cubo, trilhas) != modelo(cubo, relogio, trilhas)


def test_vgg16_reduz_o_desenho_de_256_para_8x8_como_no_treino():
    with torch.no_grad():
        caracteristicas = Extrator(VGG16)(torch.zeros(1, 3, LADO, LADO))

    assert caracteristicas.shape == (1, 512, 8, 8)


def test_parametros_tem_os_nomes_do_arquivo_de_pesos():
    nomes = set(ModeloBase().state_dict())

    assert len(nomes) == 197
    assert {
        "towers.0.cls_token",
        "towers.1.backbone.features.28.weight",
        "towers.2.proj.bias",
        "towers.0.blocks.2.attn.qkv.weight",
        "towers.1.blocks.0.ff.3.bias",
        "towers.2.blocks.1.norm2.weight",
        "classifier.weight",
    } <= nomes


def test_modelo_completo_tem_a_quantidade_de_parametros_do_treino():
    assert sum(parametro.numel() for parametro in ModeloBase().parameters()) == 46_126_658


@pytest.mark.parametrize("cabecas", [1, 2, 4])
def test_numero_de_cabecas_nao_muda_os_parametros(criar_modelo, cabecas):
    assert set(criar_modelo(cabecas=cabecas).state_dict()) == set(criar_modelo().state_dict())


def test_carrega_os_pesos_gravados_pelo_treino(tmp_path):
    torch.manual_seed(0)
    treinado = ModeloBase()
    torch.save(treinado.state_dict(), tmp_path / "model.pt")

    carregado = carregar_modelo(tmp_path / "model.pt")

    assert not carregado.training
    assert all(
        torch.equal(peso, carregado.state_dict()[nome])
        for nome, peso in treinado.state_dict().items()
    )


def test_recusa_pesos_de_outra_arquitetura(modelo, tmp_path):
    torch.save(modelo.state_dict(), tmp_path / "model.pt")

    with pytest.raises(RuntimeError, match="size mismatch"):
        carregar_modelo(tmp_path / "model.pt")
