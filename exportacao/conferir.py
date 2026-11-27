from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from torch import nn

from exportacao.exportar import ENTRADAS, SAIDAS
from exportacao.imagens import Amostra

TOLERANCIA = 1e-4


@dataclass(frozen=True)
class Conferencia:
    amostras: int
    maiores_diferencas: dict[str, float]
    tolerancia: float

    @property
    def aprovada(self) -> bool:
        return all(diferenca <= self.tolerancia for diferenca in self.maiores_diferencas.values())


def abrir_sessao(caminho: Path) -> onnxruntime.InferenceSession:
    return onnxruntime.InferenceSession(str(caminho), providers=["CPUExecutionProvider"])


def saidas_exportadas(sessao: onnxruntime.InferenceSession, amostra: Amostra) -> list[np.ndarray]:
    entradas = {nome: imagem.numpy() for nome, imagem in zip(ENTRADAS, amostra)}
    return sessao.run(list(SAIDAS), entradas)


def escore_exportado(sessao: onnxruntime.InferenceSession, amostra: Amostra) -> float:
    probabilidade, _, _ = saidas_exportadas(sessao, amostra)
    return float(probabilidade[0])


def conferir_saidas(
    modelo: nn.Module,
    caminho: Path,
    amostras: Iterable[Amostra],
    tolerancia: float = TOLERANCIA,
) -> Conferencia:
    """Compara cada saída do modelo original com a do arquivo exportado, amostra por amostra."""
    modelo.eval()
    sessao = abrir_sessao(caminho)
    diferencas: dict[str, list[float]] = {nome: [] for nome in SAIDAS}
    with torch.no_grad():
        for amostra in amostras:
            for nome, original, exportada in zip(SAIDAS, modelo(*amostra), saidas_exportadas(sessao, amostra)):
                diferencas[nome].append(float(np.abs(original.numpy() - exportada).max()))
    if not diferencas[SAIDAS[0]]:
        raise ValueError("O conjunto de conferência está vazio.")
    return Conferencia(
        len(diferencas[SAIDAS[0]]), {nome: max(valores) for nome, valores in diferencas.items()}, tolerancia
    )
