import time
from dataclasses import dataclass
from pathlib import Path

import psutil

from exportacao.conferir import abrir_sessao, escore_exportado
from exportacao.imagens import Amostra


@dataclass(frozen=True)
class Medicao:
    segundos_por_avaliacao: float
    memoria_mb: float


def medir_execucao(caminho: Path, amostras: list[Amostra]) -> Medicao:
    """Mede o tempo por avaliação e a memória que a sessão do ONNX Runtime ocupa no processo."""
    if not amostras:
        raise ValueError("O conjunto de medição está vazio.")
    processo = psutil.Process()
    memoria_antes = processo.memory_info().rss

    sessao = abrir_sessao(caminho)
    inicio = time.perf_counter()
    for amostra in amostras:
        escore_exportado(sessao, amostra)
    segundos = (time.perf_counter() - inicio) / len(amostras)

    memoria_mb = (processo.memory_info().rss - memoria_antes) / 1e6
    return Medicao(segundos_por_avaliacao=segundos, memoria_mb=memoria_mb)
