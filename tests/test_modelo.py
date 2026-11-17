import pytest
import torch
import torch.nn.functional as F

from exportacao.imagens import amostras_aleatorias
from exportacao.modelo import LADO, VGG16, Atencao, Extrator, ModeloBase, carregar_modelo, rollout

GRADE_DO_MODELO_PEQUENO = (8, 8)


def test_devolve_uma_probabilidade_por_avaliacao(modelo, lado):
    relogio, cubo, trilhas = (torch.rand(2, 3, lado, lado) for _ in range(3))

    probabilidades, _, _ = modelo(relogio, cubo, trilhas)

    assert probabilidades.shape == (2,)
    assert ((probabilidades >= 0) & (probabilidades <= 1)).all()


def test_cada_desenho_passa_por_uma_torre_propria(modelo, lado):
    (relogio, cubo, trilhas), = amostras_aleatorias(1, lado)

    assert modelo(relogio, cubo, trilhas)[0] != modelo(cubo, relogio, trilhas)[0]


@pytest.mark.parametrize("cabecas", [1, 2, 4])
def test_atencao_explicita_da_o_mesmo_resultado_que_a_do_pytorch(cabecas):
    torch.manual_seed(0)
    atencao = Atencao(16, cabecas)
    tokens = torch.rand(2, 5, 16)
    qkv = atencao.qkv(tokens).reshape(2, 5, 3, cabecas, 16 // cabecas)
    consulta, chave, valor = qkv.permute(2, 0, 3, 1, 4)
    esperada = atencao.proj(F.scaled_dot_product_attention(consulta, chave, valor).transpose(1, 2).reshape(2, 5, 16))

    saida, pesos = atencao(tokens)

    torch.testing.assert_close(saida, esperada)
    torch.testing.assert_close(pesos.sum(dim=-1), torch.ones(2, cabecas, 5))


def test_rollout_de_uma_camada_mistura_a_media_das_cabecas_com_a_identidade():
    cabecas = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]])

    resultado = rollout([cabecas])

    torch.testing.assert_close(resultado, torch.tensor([[[0.75, 0.25], [0.25, 0.75]]]))


def test_rollout_aplica_a_camada_mais_funda_por_ultimo():
    primeira = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    segunda = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    mistura_primeira = torch.tensor([[0.5, 0.5], [0.0, 1.0]])
    mistura_segunda = torch.tensor([[1.0, 0.0], [0.5, 0.5]])

    resultado = rollout([primeira, segunda])

    torch.testing.assert_close(resultado, (mistura_segunda @ mistura_primeira).unsqueeze(0))


def test_devolve_um_mapa_de_0_a_1_por_desenho(modelo, lado):
    relogio, cubo, trilhas = (torch.rand(2, 3, lado, lado) for _ in range(3))

    _, _, mapas = modelo(relogio, cubo, trilhas)

    assert mapas.shape == (2, 3, *GRADE_DO_MODELO_PEQUENO)
    planos = mapas.flatten(2)
    torch.testing.assert_close(planos.min(dim=2).values, torch.zeros(2, 3))
    torch.testing.assert_close(planos.max(dim=2).values, torch.ones(2, 3))


def test_contribuicoes_somadas_ao_vies_dao_a_probabilidade(modelo, lado):
    relogio, cubo, trilhas = (torch.rand(2, 3, lado, lado) for _ in range(3))
    vies_da_ccl = modelo.classifier.bias[1] - modelo.classifier.bias[0]

    probabilidades, contribuicoes, _ = modelo(relogio, cubo, trilhas)

    assert contribuicoes.shape == (2, 3)
    torch.testing.assert_close(torch.sigmoid(contribuicoes.sum(dim=1) + vies_da_ccl), probabilidades)


@pytest.mark.parametrize("desenho_trocado", [0, 1, 2], ids=["relogio", "cubo", "trilhas"])
def test_trocar_um_desenho_muda_so_a_contribuicao_e_o_mapa_dele(modelo, lado, desenho_trocado):
    original, outra = amostras_aleatorias(2, lado)
    trocada = list(original)
    trocada[desenho_trocado] = outra[desenho_trocado]

    _, contribuicoes, mapas = modelo(*original)
    _, contribuicoes_trocadas, mapas_trocados = modelo(*trocada)

    mantidos = [desenho for desenho in range(3) if desenho != desenho_trocado]
    torch.testing.assert_close(contribuicoes_trocadas[:, mantidos], contribuicoes[:, mantidos])
    torch.testing.assert_close(mapas_trocados[:, mantidos], mapas[:, mantidos])
    assert not torch.allclose(contribuicoes_trocadas[:, desenho_trocado], contribuicoes[:, desenho_trocado])


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
