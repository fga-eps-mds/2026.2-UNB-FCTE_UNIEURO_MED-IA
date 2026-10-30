#!/usr/bin/env python
"""Compara o modelo do treino com a reconstrução em exportacao/modelo.py, com os mesmos pesos.

Uso, na raiz do repositório:
    uv run --with scikit-learn==1.7.2 --with pandas --with opencv-python-headless --with pyyaml \
        --with torchvision python treino/scripts/conferir_reconstrucao.py model.pt \
        --imagens images --rotulos label.csv --semente 1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parents[2]
# Scripts rodados por caminho não enxergam a raiz nem o pacote mci, que não é instalado aqui.
sys.path[:0] = [str(RAIZ), str(RAIZ / "treino" / "src")]

from exportacao.imagens import ler_desenho  # noqa: E402
from exportacao.modelo import LADO, carregar_modelo  # noqa: E402
from mci.dataset import TASKS, MCIDataset, build_transforms, load_records, make_splits  # noqa: E402
from mci.model import build_model  # noqa: E402
from mci.utils import compute_metrics, load_config  # noqa: E402


def modelo_do_treino(pesos: Path, configuracao: Path) -> torch.nn.Module:
    cfg = load_config(configuracao)
    # Os pesos do ImageNet seriam sobrescritos pelo arquivo do treino, então não há o que baixar.
    cfg["model"]["pretrained_backbone"] = False
    modelo = build_model(cfg)
    modelo.load_state_dict(torch.load(pesos, map_location="cpu", weights_only=True))
    return modelo.eval()


@torch.no_grad()
def escores_do_treino(modelo: torch.nn.Module, registros: list, imagens: Path) -> np.ndarray:
    _, transformacao = build_transforms()
    conjunto = MCIDataset(registros, imagens, TASKS, transformacao)
    return np.array([
        torch.softmax(modelo([imagem.unsqueeze(0) for imagem in amostra["images"]]), dim=1)[0, 1].item()
        for amostra in conjunto
    ])


@torch.no_grad()
def escores_reconstruidos(modelo: torch.nn.Module, registros: list, imagens: Path) -> np.ndarray:
    return np.array([
        modelo(*(ler_desenho(imagens / registro.patient_id / f"{tarefa}.png", LADO) for tarefa in TASKS)).item()
        for registro in registros
    ])


def main() -> None:
    leitor = argparse.ArgumentParser(description=__doc__)
    leitor.add_argument("pesos", type=Path)
    leitor.add_argument("--imagens", type=Path, required=True)
    leitor.add_argument("--rotulos", type=Path, required=True, help="label.csv do conjunto de dados")
    leitor.add_argument("--semente", type=int, required=True)
    leitor.add_argument("--configuracao", type=Path, default=RAIZ / "treino" / "configs" / "original.yaml")
    opcoes = leitor.parse_args()

    _, _, teste = make_splits(load_records(opcoes.rotulos), opcoes.semente)
    rotulos = np.array([registro.hard_label for registro in teste])
    treino = escores_do_treino(modelo_do_treino(opcoes.pesos, opcoes.configuracao), teste, opcoes.imagens)
    reconstruido = escores_reconstruidos(carregar_modelo(opcoes.pesos), teste, opcoes.imagens)

    print(f"Pacientes de teste da semente {opcoes.semente}: {len(teste)}, {int(rotulos.sum())} com CCL")
    print(f"Modelo do treino: {compute_metrics(rotulos, treino)}")
    print(f"Reconstrução: {compute_metrics(rotulos, reconstruido)}")
    print(f"Maior diferença de escore: {np.abs(treino - reconstruido).max():.2e}")


if __name__ == "__main__":
    main()
