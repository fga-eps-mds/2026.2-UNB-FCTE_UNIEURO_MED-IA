from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import onnxruntime
import torch
from torch import nn

from exportacao.exportar import ENTRADAS
from exportacao.imagens import Amostra

TOLERANCIA = 1e-4


@dataclass(frozen=True)
class Conferencia:
    amostras: int
    maior_diferenca: float
    tolerancia: float

    @property
    def aprovada(self) -> bool:
        return self.maior_diferenca <= self.tolerancia


def abrir_sessao(caminho: Path) -> onnxruntime.InferenceSession:
    return onnxruntime.InferenceSession(str(caminho), providers=["CPUExecutionProvider"])


def escore_exportado(sessao: onnxruntime.InferenceSession, amostra: Amostra) -> float:
    entradas = {nome: imagem.numpy() for nome, imagem in zip(ENTRADAS, amostra)}
    return float(sessao.run(None, entradas)[0][0])


def conferir_escores(
    modelo: nn.Module,
    caminho: Path,
    amostras: Iterable[Amostra],
    tolerancia: float = TOLERANCIA,
) -> Conferencia:
    """Compara o escore do modelo original com o do arquivo exportado, amostra por amostra."""
    modelo.eval()
    sessao = abrir_sessao(caminho)
    diferencas = []
    with torch.no_grad():
        for amostra in amostras:
            original = float(modelo(*amostra)[0])
            diferencas.append(abs(original - escore_exportado(sessao, amostra)))
    if not diferencas:
        raise ValueError("O conjunto de conferência está vazio.")
    return Conferencia(len(diferencas), max(diferencas), tolerancia)
