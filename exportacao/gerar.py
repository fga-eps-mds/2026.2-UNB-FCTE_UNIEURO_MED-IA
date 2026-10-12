"""Exporta o modelo treinado e confere o arquivo gerado.

Uso, na raiz do repositório:
    uv run python -m exportacao.gerar model.pt modelo.onnx --imagens images
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from torch import nn

from exportacao.conferir import Conferencia, abrir_sessao, conferir_escores, escore_exportado
from exportacao.exportar import exportar_onnx
from exportacao.imagens import Amostra, amostras_aleatorias, amostras_da_pasta
from exportacao.modelo import LADO, carregar_modelo


@dataclass(frozen=True)
class Resultado:
    parametros: int
    tamanho_mb: float
    conferencia: Conferencia
    segundos_por_avaliacao: float


def gerar(modelo: nn.Module, destino: Path, amostras: list[Amostra], lado: int) -> Resultado:
    caminho = exportar_onnx(modelo, destino, lado)
    conferencia = conferir_escores(modelo, caminho, amostras)

    sessao = abrir_sessao(caminho)
    inicio = time.perf_counter()
    for amostra in amostras:
        escore_exportado(sessao, amostra)
    segundos = (time.perf_counter() - inicio) / len(amostras)

    return Resultado(
        parametros=sum(parametro.numel() for parametro in modelo.parameters()),
        tamanho_mb=caminho.stat().st_size / 1e6,
        conferencia=conferencia,
        segundos_por_avaliacao=segundos,
    )


def main(argumentos: list[str] | None = None) -> None:
    leitor = argparse.ArgumentParser(description="Exporta o modelo treinado para ONNX.")
    leitor.add_argument("pesos", type=Path, help="arquivo de pesos gerado pelo treino")
    leitor.add_argument("destino", type=Path, help="arquivo .onnx a gravar")
    leitor.add_argument("--imagens", type=Path, help="pasta com uma subpasta de desenhos por paciente")
    leitor.add_argument("--quantidade", type=int, default=20, help="avaliações usadas na conferência")
    opcoes = leitor.parse_args(argumentos)

    if opcoes.imagens:
        amostras = amostras_da_pasta(opcoes.imagens, opcoes.quantidade, LADO)
    else:
        amostras = amostras_aleatorias(opcoes.quantidade, LADO)
    resultado = gerar(carregar_modelo(opcoes.pesos), opcoes.destino, amostras, LADO)

    conferencia = resultado.conferencia
    print(f"Parâmetros: {resultado.parametros}")
    print(f"Arquivo exportado: {resultado.tamanho_mb:.1f} MB")
    print(f"Maior diferença de escore em {conferencia.amostras} amostras: "
          f"{conferencia.maior_diferenca:.2e}")
    print(f"Dentro da tolerância de {conferencia.tolerancia:.0e}: "
          f"{'sim' if conferencia.aprovada else 'não'}")
    print(f"Tempo por avaliação neste computador: {resultado.segundos_por_avaliacao:.2f} s")


if __name__ == "__main__":
    main()
