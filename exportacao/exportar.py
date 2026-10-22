from pathlib import Path

import torch
from torch import nn

ENTRADAS = ("relogio", "cubo", "trilhas")
SAIDA = "probabilidade_ccl"


def exportar_onnx(modelo: nn.Module, destino: Path, lado: int) -> Path:
    """Grava o modelo em ONNX, com uma avaliação por chamada e entradas de tamanho fixo."""
    modelo.eval()
    exemplo = tuple(torch.zeros(1, 3, lado, lado) for _ in ENTRADAS)
    programa = torch.onnx.export(
        modelo,
        exemplo,
        input_names=list(ENTRADAS),
        output_names=[SAIDA],
        dynamo=True,
        verbose=False,
    )
    programa.save(str(destino))
    return destino
