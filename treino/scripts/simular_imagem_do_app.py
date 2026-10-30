#!/usr/bin/env python
"""Mede o quanto o preparo da imagem do aplicativo muda o escore, sem usar o aplicativo.

Uso, na raiz do repositório:
    uv run --with scikit-learn==1.7.2 --with pandas --with opencv-python-headless --with pyyaml \
        --with torchvision python treino/scripts/simular_imagem_do_app.py modelo.onnx \
        --imagens images --rotulos label.csv --semente 1
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnxruntime
from PIL import Image, ImageFilter

RAIZ = Path(__file__).resolve().parents[2]
# Scripts rodados por caminho não enxergam a raiz nem o pacote mci, que não é instalado aqui.
sys.path[:0] = [str(RAIZ), str(RAIZ / "treino" / "src")]

from exportacao.exportar import ENTRADAS  # noqa: E402
from exportacao.modelo import LADO  # noqa: E402
from mci.dataset import TASKS, load_records, make_splits  # noqa: E402
from mci.utils import compute_metrics  # noqa: E402

# Quadro suposto a partir das posições dos círculos das trilhas no aplicativo. O real depende do aparelho.
LARGURA_DO_QUADRO, ALTURA_DO_QUADRO = 1168, 520
COR_DO_TRACO_NO_APP = np.array((0x0F, 0x17, 0x2A), dtype=np.float32) / 255
LIMIAR_DE_CINZA = 128
Imagem = Image.Image


def no_quadro(desenho: Imagem) -> Imagem:
    ampliado = desenho.resize((ALTURA_DO_QUADRO, ALTURA_DO_QUADRO), Image.Resampling.BILINEAR)
    quadro = Image.new("RGB", (LARGURA_DO_QUADRO, ALTURA_DO_QUADRO), "white")
    quadro.paste(ampliado, ((LARGURA_DO_QUADRO - ALTURA_DO_QUADRO) // 2, 0))
    return quadro


def recortar(quadro: Imagem) -> Imagem:
    margem = (quadro.width - quadro.height) // 2
    quadrado = quadro.crop((margem, 0, margem + quadro.height, quadro.height))
    return quadrado.resize((LADO, LADO), Image.Resampling.BILINEAR)


def esticar(quadro: Imagem) -> Imagem:
    return quadro.resize((LADO, LADO), Image.Resampling.BILINEAR)


def com_limiar(desenho: Imagem) -> Imagem:
    cinza = np.asarray(desenho.convert("L"))
    return Image.fromarray(np.where(cinza < LIMIAR_DE_CINZA, 0, 255).astype(np.uint8)).convert("RGB")


def com_cor_do_app(desenho: Imagem) -> Imagem:
    pixels = np.asarray(desenho).astype(np.float32) / 255
    return Image.fromarray((255 * (1 - (1 - pixels) * (1 - COR_DO_TRACO_NO_APP))).astype(np.uint8))


VARIACOES: dict[str, Callable[[Imagem], Imagem]] = {
    "Imagem do conjunto": lambda desenho: desenho,
    "Recorte central do quadro": lambda desenho: recortar(no_quadro(desenho)),
    "Quadro esticado": lambda desenho: esticar(no_quadro(desenho)),
    "Recorte central com limiar": lambda desenho: com_limiar(recortar(no_quadro(desenho))),
    "Traço 1 pixel mais grosso": lambda desenho: desenho.filter(ImageFilter.MinFilter(3)),
    "Traço na cor do aplicativo": com_cor_do_app,
}


def main() -> None:
    leitor = argparse.ArgumentParser(description=__doc__)
    leitor.add_argument("onnx", type=Path)
    leitor.add_argument("--imagens", type=Path, required=True)
    leitor.add_argument("--rotulos", type=Path, required=True, help="label.csv do conjunto de dados")
    leitor.add_argument("--semente", type=int, required=True)
    opcoes = leitor.parse_args()

    sessao = onnxruntime.InferenceSession(str(opcoes.onnx), providers=["CPUExecutionProvider"])
    _, _, teste = make_splits(load_records(opcoes.rotulos), opcoes.semente)
    rotulos = np.array([registro.hard_label for registro in teste])

    def escore(paciente: str, variacao: Callable[[Imagem], Imagem]) -> float:
        entradas = {}
        for nome, tarefa in zip(ENTRADAS, TASKS):
            imagem = variacao(Image.open(opcoes.imagens / paciente / f"{tarefa}.png").convert("RGB"))
            entradas[nome] = (np.asarray(imagem).astype(np.float32) / 255).transpose(2, 0, 1)[None]
        return float(sessao.run(None, entradas)[0][0])

    base = None
    for nome, variacao in VARIACOES.items():
        escores = np.array([escore(registro.patient_id, variacao) for registro in teste])
        base = escores if base is None else base
        mudaram = int(((escores >= 0.5) != (base >= 0.5)).sum())
        print(f"{nome}: {compute_metrics(rotulos, escores)}, diferença média "
              f"{np.abs(escores - base).mean():.3f}, mudaram de classe {mudaram}/{len(teste)}", flush=True)


if __name__ == "__main__":
    main()
