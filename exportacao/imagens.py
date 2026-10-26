from pathlib import Path

import numpy as np
import torch
from PIL import Image

from exportacao.exportar import ENTRADAS

Amostra = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# Nomes dos arquivos de cada paciente no conjunto de dados, na ordem das entradas.
ARQUIVOS = ("clock.png", "copy.png", "trail.png")


def ler_desenho(caminho: Path, lado: int) -> torch.Tensor:
    """Lê o desenho em RGB, no tamanho do treino, com valores de 0 a 1."""
    imagem = Image.open(caminho).convert("RGB").resize((lado, lado), Image.Resampling.BILINEAR)
    pixels = torch.from_numpy(np.asarray(imagem).copy())
    return pixels.permute(2, 0, 1).float().div(255).unsqueeze(0)


def amostras_da_pasta(pasta: Path, quantidade: int, lado: int) -> list[Amostra]:
    """Lê os desenhos dos primeiros pacientes da pasta, que tem uma subpasta por paciente."""
    pacientes = sorted(caminho for caminho in pasta.iterdir() if caminho.is_dir())
    return [
        tuple(ler_desenho(paciente / arquivo, lado) for arquivo in ARQUIVOS)
        for paciente in pacientes[:quantidade]
    ]


def amostras_aleatorias(quantidade: int, lado: int, semente: int = 0) -> list[Amostra]:
    gerador = torch.Generator().manual_seed(semente)
    return [
        tuple(torch.rand(1, 3, lado, lado, generator=gerador) for _ in ENTRADAS)
        for _ in range(quantidade)
    ]
