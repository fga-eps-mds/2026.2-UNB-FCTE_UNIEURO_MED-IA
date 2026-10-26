import pytest
import torch

from exportacao.modelo import ModeloBase

BLOCOS_PEQUENOS = (8, "M", 16, "M")


@pytest.fixture
def lado():
    return 32


@pytest.fixture
def criar_modelo():
    def criar(semente: int = 0, cabecas: int = 1) -> ModeloBase:
        torch.manual_seed(semente)
        return ModeloBase(blocos=BLOCOS_PEQUENOS, dimensao=16, camadas=2, cabecas=cabecas).eval()

    return criar


@pytest.fixture
def modelo(criar_modelo):
    return criar_modelo()
